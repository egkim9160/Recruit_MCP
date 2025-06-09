#!/usr/bin/env python3

import os
import ssl
import json
import copy
import re
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
import tiktoken

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastMCP("의료진 채용 검색 서버")

try:
    token_encoder = tiktoken.encoding_for_model("gpt-4")
except Exception as e:
    logger.warning(f"GPT-4 인코더 로드 실패, cl100k_base 사용: {e}")
    token_encoder = tiktoken.get_encoding("cl100k_base")

def calculate_tokens(text: str) -> int:
    try:
        return len(token_encoder.encode(text))
    except Exception as e:
        logger.error(f"토큰 계산 오류: {e}")
        return len(text.split()) 

class OpenSearchService:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")

        self._connect()
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            request_timeout=30,
            openai_api_key=openai_api_key
        )
        self.recruit_index = "recruit_text-embedding-3-small_1536_100000_300_20250529_150924"
        self.resume_index = "resume_text-embedding-3-large_3072_100000_300_20250221_175445"

    def _connect(self):
        opensearch_host = os.getenv("OPENSEARCH_HOST", 'opensearch.medigate.net')
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", 9200))
        opensearch_user = os.getenv("OPENSEARCH_USER", 'medigate')
        opensearch_password = os.getenv("OPENSEARCH_PASSWORD", 'Soakaeofh12!@')

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_auth=(opensearch_user, opensearch_password),
            use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=30, ssl_context=ssl_context
        )

    async def create_embedding(self, text: str) -> List[float]:
        try:
            return await self.embeddings_model.aembed_query(text)
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return []

    def build_filter_conditions(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        filter_clauses = []
        for field, value in filters.items():
            if field == "region" and value:
                filter_clauses.append({"wildcard": {"metadata.REGION_NAME": f"*{value}*"}})
            elif field == "department" and value:
                filter_clauses.append({"wildcard": {"metadata.SPECIALTIES": f"*{value}*"}})
            elif field == "hospital_name" and value:
                filter_clauses.append({"wildcard": {"metadata.ORGANIZATION_NAME": f"*{value}*"}})
        return filter_clauses

    def build_search_query(self, index_hint: str, filters: Dict[str, Any] = None,
                          semantic_query: str = None, query_vector: List[float] = None,
                          exclude_filters: List[Dict[str, Any]] = None, size: int = 10,
                          sort_options: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        query = {"size": min(size, 50), "_source": {"excludes": ["vector_field"]}}
        bool_query = {"must": [], "filter": [], "should": [], "must_not": []}

        if filters:
            bool_query["filter"].extend(self.build_filter_conditions(filters))
        if exclude_filters:
            bool_query["must_not"].extend(exclude_filters)
        if query_vector:
            bool_query["must"].append({"knn": {"vector_field": {"vector": query_vector, "k": max(10, size)}}})
        if semantic_query and not query_vector: 
            bool_query["must"].append({"multi_match": {"query": semantic_query, "fields": ["text", "metadata.*"]}})
        
        if any(bool_query.values()):
            query["query"] = {"bool": bool_query}
        else: 
            if not query_vector : 
                 query["query"] = {"match_all": {}}

        if sort_options:
            query["sort"] = sort_options
            
        return query

    def _create_logging_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        logging_query = copy.deepcopy(query)
        if "query" in logging_query and "bool" in logging_query["query"]:
            bool_clauses = logging_query["query"]["bool"]
            for clause_type in ["must", "should"]:
                if clause_type in bool_clauses:
                    for i, condition in enumerate(bool_clauses[clause_type]):
                        if "knn" in condition:
                            for knn_field_name, knn_field_content in condition["knn"].items():
                                if isinstance(knn_field_content, dict) and "vector" in knn_field_content and isinstance(knn_field_content["vector"], list):
                                    vector_length = len(knn_field_content["vector"])
                                    logging_query["query"]["bool"][clause_type][i]["knn"][knn_field_name]["vector"] = f"<{vector_length}차원 벡터 데이터 생략>"
        return logging_query

    async def search(self, index_hint: str, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            index_name = self.recruit_index if index_hint in ["recruit", "job", "board"] else self.resume_index if index_hint in ["user", "resume", "profile"] else index_hint

            logging_query = self._create_logging_query(query)
            logger.info(f"OpenSearch Query to index '{index_name}':\n{json.dumps(logging_query, ensure_ascii=False, indent=2)}")

            return self.client.search(index=index_name, body=query)
        except opensearch_exceptions.NotFoundError:
            logger.error(f"Index '{index_name}' not found.")
            return {"hits": {"hits": [], "total": {"value": 0}}}
        except Exception as e:
            logger.error(f"OpenSearch 검색 오류: {e}", exc_info=True)
            return {"hits": {"hits": [], "total": {"value": 0}}}

os_service = OpenSearchService()

def _format_board_details_to_dict(hit: Dict[str, Any]) -> Dict[str, Any]:
    source = hit["_source"]
    metadata = source.get("metadata", {})
    text_content = source.get("text", "")
    title_from_text = ""
    if "TITLE =>" in text_content:
        title_from_text = text_content.split("||")[0].replace("TITLE => ", "").strip()
    
    title = title_from_text if title_from_text else metadata.get("TITLE", "")

    return {
        "board_id": metadata.get("BOARD_IDX", hit.get("_id")),
        "title": title,
        "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
        "department": metadata.get("SPECIALTIES", ""),
        "region": metadata.get("REGION_NAME", ""),
        "employment_type": metadata.get("REGULAR_STATUS", ""),
        "work_type": metadata.get("WORK_TYPE", ""),
        "pay_details": metadata.get("PAY_DETAILS", ""),
        "work_hours": metadata.get("WORK_HOUR_DETAILS", ""),
        "address": metadata.get("ADDRESS", ""),
        "benefits": metadata.get("MEAL_HOUSE_STATUS", ""),
        "description_full": text_content,
        "description_short": (text_content[:100] + "..." if len(text_content) > 100 else text_content) if text_content else "",
        "view_count": metadata.get("view_count") 
    }

@app.tool()
async def get_board_by_id(board_id: str) -> str:
    logger.info(f"도구 실행: get_board_by_id, ID: {board_id}")
    try:
        search_query = {"query": {"term": {"metadata.BOARD_IDX": board_id}}, "size": 1, "_source": {"excludes": ["vector_field"]}}
        result = await os_service.search("recruit", search_query)
        if not result["hits"]["hits"]:
            return json.dumps({"success": False, "error": f"공고 ID {board_id} 없음"}, ensure_ascii=False)

        formatted_result = _format_board_details_to_dict(result["hits"]["hits"][0])
        logger.info(f"get_board_by_id 성공: ID {board_id}")
        return json.dumps({"success": True, "data": formatted_result}, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"get_board_by_id 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"공고 조회 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def get_user_by_id(user_id: str) -> str:
    logger.info(f"도구 실행: get_user_by_id, ID: {user_id}")
    try:
        search_query = {"query": {"term": {"metadata.U_ID": user_id}}, "size": 1, "_source": {"excludes": ["vector_field"]}}
        result = await os_service.search("user", search_query)
        if not result["hits"]["hits"]:
            return json.dumps({"success": False, "error": f"사용자 ID {user_id} 없음"}, ensure_ascii=False)

        hit = result["hits"]["hits"][0]
        metadata = hit["_source"]["metadata"]
        profile_text = hit["_source"].get("text", "")

        formatted_result = {"success": True, "data": {
            "user_id": metadata.get("U_ID", user_id),
            "department": metadata.get("SPECIALTY", ""),
            "preferred_region": metadata.get("HOPE_LOCATION", ""),
            "experience_level": metadata.get("EXPERIENCE", ""),
            "work_type": metadata.get("WORK_TYPE", ""),
            "profile_text": profile_text[:300] + "..." if len(profile_text) > 300 else profile_text
        }}
        logger.info(f"get_user_by_id 성공: ID {user_id}")
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"get_user_by_id 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"사용자 조회 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def create_and_search_recruits(
    region: str = None,
    department: str = None,
    hospital_name: str = None,
    semantic_keywords: str = None,
    exclude_board_id: str = None,
    size: int = 10
) -> str:
    params = locals()
    log_params = {k: v for k, v in params.items() if v is not None and k != 'semantic_keywords'}
    if semantic_keywords:
        log_params['semantic_keywords_present'] = True
    logger.info(f"도구 실행: create_and_search_recruits, Params: {log_params}")

    try:
        filters = {k: v for k, v in {"region": region, "department": department, "hospital_name": hospital_name}.items() if v}
        exclude_filters = [{"term": {"metadata.BOARD_IDX": exclude_board_id}}] if exclude_board_id else []
        query_vector = await os_service.create_embedding(semantic_keywords) if semantic_keywords else None

        query = os_service.build_search_query("recruit", filters, semantic_keywords, query_vector, exclude_filters, size)
        result = await os_service.search("recruit", query)

        formatted_results = []
        board_ids = []

        for hit in result["hits"]["hits"]:
            board_detail = _format_board_details_to_dict(hit)
            board_ids.append(board_detail["board_id"])
            formatted_results.append({
                "board_id": board_detail["board_id"],
                "title": board_detail["title"],
                "hospital_name": board_detail["hospital_name"],
                "department": board_detail["department"],
                "region": board_detail["region"],
                "employment_type": board_detail["employment_type"],
                "work_type": board_detail["work_type"],
                "pay_details": board_detail["pay_details"], 
                "description": board_detail["description_short"]
            })

        result_data = {
            "success": True,
            "total_hits": result["hits"]["total"]["value"],
            "results": formatted_results,
            "board_ids": board_ids
        }

        logger.info(f"create_and_search_recruits 성공: 반환된 결과 {len(formatted_results)}개 (총 검색 {result['hits']['total']['value']}개)")
        final_json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
        token_count = calculate_tokens(final_json_str)
        logger.info(f"📊 create_and_search_recruits 응답 크기 - 토큰 수: {token_count}, 결과 수: {len(formatted_results)}")
        return final_json_str

    except Exception as e:
        logger.error(f"create_and_search_recruits 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"공고 검색 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def create_and_search_users(
    department: str = None,
    preferred_region: str = None,
    semantic_keywords: str = None,
    size: int = 10
) -> str:
    params = locals()
    log_params = {k: v for k, v in params.items() if v is not None and k != 'semantic_keywords'}
    if semantic_keywords:
        log_params['semantic_keywords_present'] = True
    logger.info(f"도구 실행: create_and_search_users, Params: {log_params}")

    try:
        user_query_bool_filter = []
        if department:
            user_query_bool_filter.append({"wildcard": {"metadata.SPECIALTY": f"*{department}*"}})
        if preferred_region:
            user_query_bool_filter.append({"wildcard": {"metadata.HOPE_LOCATION": f"*{preferred_region}*"}})

        user_query_bool_must = []
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)
            if query_vector:
                user_query_bool_must.append({"knn": {"vector_field": {"vector": query_vector, "k": max(10, size)}}})

        user_query_body = {"size": min(size, 50), "_source": {"excludes": ["vector_field"]}}
        user_query_bool_conditions = {}
        if user_query_bool_filter:
            user_query_bool_conditions["filter"] = user_query_bool_filter
        if user_query_bool_must:
            user_query_bool_conditions["must"] = user_query_bool_must
        
        if user_query_bool_conditions:
            user_query_body["query"] = {"bool": user_query_bool_conditions}
        else:
            user_query_body["query"] = {"match_all": {}}

        result = await os_service.search("user", user_query_body)
        formatted_results = []

        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            profile_text = source.get("text", "")

            formatted_results.append({
                "user_id": metadata.get("U_ID", hit["_id"]),
                "department": metadata.get("SPECIALTY", ""),
                "preferred_region": metadata.get("HOPE_LOCATION", ""),
                "experience_level": metadata.get("EXPERIENCE", ""),
                "work_type": metadata.get("WORK_TYPE", ""),
                "profile_text": profile_text[:200] + "..." if len(profile_text) > 200 else profile_text
            })

        result_data = {"success": True, "total_hits": result["hits"]["total"]["value"], "results": formatted_results}
        logger.info(f"create_and_search_users 성공: {len(formatted_results)}개 결과")
        final_json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
        token_count = calculate_tokens(final_json_str)
        logger.info(f"📊 create_and_search_users 응답 크기 - 토큰 수: {token_count}, 결과 수: {len(formatted_results)}")
        return final_json_str

    except Exception as e:
        logger.error(f"create_and_search_users 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"사용자 검색 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def create_formatted_recommendations(
    search_criteria: str,
    selected_board_ids: List[str]
) -> str:
    """
    사용자의 최초 요청에 맞는 공고를 LLM이 선별한 최대 5개 ID 목록을 받아서
    각 공고의 상세 정보를 조회하여 사용자 친화적인 텍스트 추천 목록을 생성하고,
    그 뒤에 관련 메타데이터를 JSON 문자열 형태로 추가하여 하나의 문자열로 반환합니다.

    Args:
        search_criteria (str): 원래 검색 조건/요구사항.
        selected_board_ids (List[str]): 사용자 요청에 부합한 순으로 LLM이 선별한 최대 5개의 공고 ID 목록.

    Returns:
        str: "텍스트 추천 목록\n\n<<METADATA_JSON_START>>\nJSON 메타데이터 문자열\n<<METADATA_JSON_END>>" 형태의 문자열.
             메타데이터 JSON 예시: {"datatype": "recommendation_metadata", "board_ids": [1172181, 1172325]}
             오류 시: {"success": false, "error": "오류 메시지"} 형태의 JSON 문자열.
    """
    logger.info(f"도구 실행: create_formatted_recommendations, 검색 조건: {search_criteria}, 선별된 ID 수: {len(selected_board_ids)}")

    if not selected_board_ids:
        return json.dumps({
            "success": False,
            "error": "선별된 공고 ID가 없습니다. `selected_board_ids`는 비어있을 수 없습니다."
        }, ensure_ascii=False)
    
    if len(selected_board_ids) > 5:
        logger.warning(f"선별된 공고 ID가 5개를 초과합니다 ({len(selected_board_ids)}개). 처음 5개만 처리합니다.")
        board_ids_to_process = selected_board_ids[:5]
    else:
        board_ids_to_process = selected_board_ids

    try:
        # timestamp = int(time.time()) # 필요시 메타데이터에 추가
        detailed_recommendations_data = []
        actual_processed_ids = [] 
        
        for board_id in board_ids_to_process:
            board_info_json_str = await get_board_by_id(board_id)
            try:
                board_info = json.loads(board_info_json_str)
                if board_info.get("success") and "data" in board_info:
                    detailed_recommendations_data.append(board_info["data"])
                    # board_id가 문자열일 수 있으므로 str()로 변환하여 일관성 유지
                    if board_info["data"].get("board_id"):
                        actual_processed_ids.append(str(board_info["data"]["board_id"])) 
                else:
                    logger.warning(f"추천 공고 ID {board_id} 정보 조회 실패: {board_info.get('error', '알 수 없는 오류')}")
            except json.JSONDecodeError:
                logger.error(f"공고 ID {board_id} 정보 조회 결과 JSON 파싱 실패: {board_info_json_str}")

        if not detailed_recommendations_data:
            return json.dumps({
                "success": False,
                "error": "선별된 공고들의 상세 정보를 조회할 수 없습니다. ID가 유효한지 확인하세요."
            }, ensure_ascii=False)

        text_output_lines = [f"'{search_criteria}' 조건에 따라 다음 {len(detailed_recommendations_data)}개 공고를 추천합니다:"]
        for i, r_detail in enumerate(detailed_recommendations_data, 1):
            text_output_lines.append(f"\n🏥 {i}. {r_detail.get('title', 'N/A')} (ID: {r_detail.get('board_id', 'N/A')})")
            text_output_lines.extend([
                f"   - 병원: {r_detail.get('hospital_name', 'N/A')}, 지역: {r_detail.get('region', 'N/A')}",
                f"   - 진료과: {r_detail.get('department', 'N/A')}, 고용형태: {r_detail.get('employment_type', 'N/A')}",
                f"   - 근무유형: {r_detail.get('work_type', 'N/A')}",
                f"   - 급여: {r_detail.get('pay_details', '정보 없음') if r_detail.get('pay_details') else '정보 없음'}",
                f"   - 간략 설명: {r_detail.get('description_short', 'N/A')}"
            ])
        
        formatted_text_part = "\n".join(text_output_lines)
        
        metadata_json_content = {
            "datatype": "recommendation_metadata", 
            "board_ids": actual_processed_ids,
            # "search_criteria": search_criteria, # 필요시 추가
            # "timestamp": timestamp # 필요시 추가
        }
        metadata_json_string = json.dumps(metadata_json_content, ensure_ascii=False) # indent 없이 한 줄로

        final_output_string = f"{formatted_text_part}\n\n<<METADATA_JSON_START>>\n{metadata_json_string}\n<<METADATA_JSON_END>>"
        
        token_count = calculate_tokens(final_output_string) 
        logger.info(f"📊 create_formatted_recommendations 전체 응답 토큰 수: {token_count}, 추천 수: {len(detailed_recommendations_data)}")
        
        return final_output_string

    except Exception as e:
        logger.error(f"create_formatted_recommendations 오류: {e}", exc_info=True)
        return json.dumps({ 
            "success": False,
            "error": f"추천 생성 및 포맷팅 오류: {str(e)}"
        }, ensure_ascii=False)

"""
@app.tool()
async def summarize_board_by_id(board_id: str) -> str:
    # ... (이 함수는 현재 주석 처리되어 있음) ...
"""
@app.tool()
async def recommend_popular_jobs(user_id: Optional[str] = None, size: int = 5) -> str:
    logger.info(f"도구 실행: recommend_popular_jobs, 사용자 ID: {user_id}, 개수: {size}")
    try:
        filters = {}
        user_department_for_log = "전체"
        if user_id:
            user_info_json_str = await get_user_by_id(user_id)
            user_info = json.loads(user_info_json_str) 
            if user_info.get("success") and user_info["data"].get("department"):
                user_department = user_info["data"]["department"]
                if user_department: 
                    filters["department"] = user_department
                    user_department_for_log = user_department
                    logger.info(f"사용자 {user_id}의 전문과 '{user_department}' 기준으로 인기 공고 검색")
                else:
                    logger.info(f"사용자 {user_id}의 전문과 정보가 비어있습니다. 전체 인기 공고를 검색합니다.")
            else:
                logger.warning(f"사용자 ID {user_id}의 프로필 또는 전문과 정보를 가져오지 못했습니다. 전체 인기 공고를 검색합니다.")
        
        sort_options = [{"metadata.view_count": {"order": "desc", "missing": "_last", "unmapped_type": "long"}}]

        query = os_service.build_search_query(
            index_hint="recruit",
            filters=filters, 
            size=min(size, 20),
            sort_options=sort_options
        )
        
        result = await os_service.search("recruit", query)

        if not result["hits"]["hits"]:
            return f"{user_department_for_log} 분야에서 추천할 인기 공고를 찾을 수 없습니다." # 일반 텍스트 반환

        output_lines = [f"'{user_department_for_log}' 분야의 인기 채용공고 {len(result['hits']['hits'])}개를 추천합니다:"]
        for i, hit in enumerate(result['hits']['hits'], 1):
            board_detail = _format_board_details_to_dict(hit)
            output_lines.append(
                f"\n🌟 {i}. {board_detail.get('title', 'N/A')} (ID: {board_detail.get('board_id', 'N/A')})"
            )
            output_lines.extend([
                f"   - 병원: {board_detail.get('hospital_name', 'N/A')}, 지역: {board_detail.get('region', 'N/A')}",
                f"   - 진료과: {board_detail.get('department', 'N/A')}",
                f"   - 인기도(조회수): {board_detail.get('view_count', '정보 없음') if isinstance(board_detail.get('view_count'), int) else '집계 중'}"
            ])
        
        logger.info(f"recommend_popular_jobs 성공: {len(result['hits']['hits'])}개 공고 추천 (분야: {user_department_for_log})")
        return "\n".join(output_lines) # 일반 텍스트 반환

    except opensearch_exceptions.RequestError as e:
        if "Field [metadata.view_count] is not a numeric type" in str(e) or \
           "No mapping found for [metadata.view_count]" in str(e) or \
           "can't load numeric doc values" in str(e):
             logger.error(f"인기 공고 추천 오류: 'metadata.view_count' 필드로 정렬할 수 없습니다. OpenSearch 매핑을 확인하세요. 에러: {e}", exc_info=False)
             return "인기 공고를 조회하는 중 문제가 발생했습니다. (인기 지표 설정 오류). 관리자에게 문의하세요." 
        logger.error(f"recommend_popular_jobs OpenSearch 요청 오류: {e}", exc_info=True)
        return f"인기 공고 추천 중 OpenSearch 오류 발생: {str(e)}" 
    except json.JSONDecodeError as e: 
        logger.error(f"recommend_popular_jobs 사용자 정보 파싱 오류: {e}", exc_info=True)
        return "사용자 정보를 처리하는 중 오류가 발생했습니다." 
    except Exception as e:
        logger.error(f"recommend_popular_jobs 일반 오류: {e}", exc_info=True)
        return f"인기 공고 추천 중 알 수 없는 오류 발생: {str(e)}" 

if __name__ == "__main__":
    app.run(transport="stdio")
