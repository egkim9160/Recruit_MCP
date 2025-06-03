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

# 토큰 계산용 인코더 초기화 (GPT-4 기준)
try:
    token_encoder = tiktoken.encoding_for_model("gpt-4")
except Exception as e:
    logger.warning(f"GPT-4 인코더 로드 실패, cl100k_base 사용: {e}")
    token_encoder = tiktoken.get_encoding("cl100k_base")

def calculate_tokens(text: str) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    try:
        return len(token_encoder.encode(text))
    except Exception as e:
        logger.error(f"토큰 계산 오류: {e}")
        return len(text.split()) # 대략적인 단어 수로 대체

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
        if semantic_query and not query_vector: # semantic_query가 있고, vector가 없을 때만 multi_match 사용
            bool_query["must"].append({"multi_match": {"query": semantic_query, "fields": ["text", "metadata.*"]}})
        
        # bool_query에 조건이 하나라도 있을 때만 query["query"]에 할당
        if any(bool_query.values()):
            query["query"] = {"bool": bool_query}
        else: # 아무 조건도 없으면 match_all 사용 (knn 검색 등 다른 조건이 must에 없을 경우)
            if not query_vector : # 벡터 검색이 아닌 경우에만 match_all
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
    # TITLE 필드가 메타데이터에 있을 수도 있고, text에 있을 수도 있음. 우선순위 적용.
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
        "view_count": metadata.get("view_count") # view_count는 숫자일 것으로 가정
    }

@app.tool()
async def get_board_by_id(board_id: str) -> str:
    """
    채용공고 ID로 특정 채용공고의 상세 정보를 조회합니다.
    Args:
        board_id (str): 조회할 채용공고의 고유 ID
    Returns:
        str: JSON 형태의 문자열로 채용공고 상세 정보 또는 오류 메시지를 포함.
    """
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
    """
    사용자 ID로 특정 사용자(의료진)의 프로필 정보를 조회합니다.
    Args:
        user_id (str): 조회할 사용자의 고유 ID
    Returns:
        str: JSON 형태의 문자열로 사용자 프로필 정보 또는 오류 메시지를 포함.
    """
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
    """
    다양한 조건으로 채용공고를 검색합니다. 필터링과 의미적 검색을 지원합니다.
    Args:
        region (str, optional): 지역 필터 (예: "서울", "경기도")
        department (str, optional): 진료과/전문과목 필터 (예: "내과", "외과")
        hospital_name (str, optional): 병원명 필터
        semantic_keywords (str, optional): 의미적 검색을 위한 키워드
        exclude_board_id (str, optional): 제외할 공고 ID
        size (int, optional): 반환할 결과 수 (기본값: 10, 최대: 50)
    Returns:
        str: JSON 형태의 문자열로 검색 결과 또는 오류 메시지를 포함. 검색 결과에는 board_id 목록과 각 공고의 간략한 정보가 포함됩니다.
    """
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
            # LLM이 다음 단계를 위해 필요한 최소한의 정보 + 요약 정보
            formatted_results.append({
                "board_id": board_detail["board_id"],
                "title": board_detail["title"],
                "hospital_name": board_detail["hospital_name"],
                "department": board_detail["department"],
                "region": board_detail["region"],
                "employment_type": board_detail["employment_type"],
                "work_type": board_detail["work_type"],
                "pay_details": board_detail["pay_details"], # 급여 정보는 LLM 판단에 중요할 수 있음
                "description": board_detail["description_short"] # 간략 설명
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
    """
    다양한 조건으로 사용자(의료진)를 검색합니다. 필터링과 의미적 검색을 지원합니다.
    Args:
        department (str, optional): 전문과목 필터 (예: "내과", "외과")
        preferred_region (str, optional): 선호 지역 필터 (예: "서울", "경기도")
        semantic_keywords (str, optional): 의미적 검색을 위한 키워드
        size (int, optional): 반환할 결과 수 (기본값: 10, 최대: 50)
    Returns:
        str: JSON 형태의 문자열로 사용자 검색 결과 또는 오류 메시지를 포함.
    """
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
    LLM이 선별한 최대 5개 공고 ID 목록을 받아, 각 공고의 상세 정보를 조회하고,
    사용자에게 보여줄 포맷팅된 추천 결과 문자열을 생성합니다.
    선별된 공고 ID 목록은 JSON 파일로도 저장됩니다.

    Args:
        search_criteria (str): 원래 검색 조건/요구사항 (요약 생성에 사용).
        selected_board_ids (List[str]): LLM이 선별한 최대 5개의 공고 ID 목록.

    Returns:
        str: 포맷팅된 추천 결과 문자열 (사용자 친화적).
             오류 발생 시, 오류 정보가 담긴 JSON 문자열 반환.
    """
    logger.info(f"도구 실행: create_formatted_recommendations, 검색 조건: {search_criteria}, 선별된 ID 수: {len(selected_board_ids)}")

    if not selected_board_ids:
        return json.dumps({"success": False, "error": "선별된 공고 ID가 없습니다. `selected_board_ids`는 비어있을 수 없습니다."}, ensure_ascii=False)
    
    if len(selected_board_ids) > 5:
        logger.warning(f"선별된 공고 ID가 5개를 초과합니다 ({len(selected_board_ids)}개). 처음 5개만 처리합니다.")
        board_ids_to_process = selected_board_ids[:5]
    else:
        board_ids_to_process = selected_board_ids

    try:
        timestamp = int(time.time())
        filename = f"recommended_board_ids_{timestamp}.json"
        try:
            # 현재 작업 디렉토리에 저장
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "search_criteria": search_criteria,
                    "selected_board_ids": board_ids_to_process,
                    "timestamp": timestamp
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 선별된 Board IDs 저장됨: {filepath}")
        except Exception as e:
            logger.error(f"❌ 추천 ID JSON 파일 저장 오류 ({filename}): {e}")
            filename = "저장 실패" # 파일 저장 실패 시 사용자에게 알릴 파일명

        detailed_recommendations = []
        for board_id in board_ids_to_process:
            # get_board_by_id는 JSON 문자열을 반환하므로 파싱 필요
            board_info_json_str = await get_board_by_id(board_id)
            try:
                board_info = json.loads(board_info_json_str)
                if board_info.get("success") and "data" in board_info:
                    detailed_recommendations.append(board_info["data"])
                else:
                    logger.warning(f"추천 공고 ID {board_id} 정보 조회 실패: {board_info.get('error', '알 수 없는 오류')}")
            except json.JSONDecodeError:
                logger.error(f"공고 ID {board_id} 정보 조회 결과 JSON 파싱 실패: {board_info_json_str}")


        if not detailed_recommendations:
            return json.dumps({"success": False, "error": "선별된 공고들의 상세 정보를 조회할 수 없습니다. ID가 유효한지 확인하세요."}, ensure_ascii=False)

        output_lines = [f"'{search_criteria}' 조건에 따라 다음 {len(detailed_recommendations)}개 공고를 추천합니다:"]
        for i, r_detail in enumerate(detailed_recommendations, 1):
            # _format_board_details_to_dict 에서 온 필드들을 사용
            output_lines.append(f"\n🏥 {i}. {r_detail.get('title', 'N/A')} (ID: {r_detail.get('board_id', 'N/A')})")
            output_lines.extend([
                f"   - 병원: {r_detail.get('hospital_name', 'N/A')}, 지역: {r_detail.get('region', 'N/A')}",
                f"   - 진료과: {r_detail.get('department', 'N/A')}, 고용형태: {r_detail.get('employment_type', 'N/A')}",
                f"   - 근무유형: {r_detail.get('work_type', 'N/A')}",
                f"   - 급여: {r_detail.get('pay_details', '정보 없음') if r_detail.get('pay_details') else '정보 없음'}",
                f"   - 간략 설명: {r_detail.get('description_short', 'N/A')}"
            ])
        
        formatted_output_str = "\n".join(output_lines)
        token_count = calculate_tokens(formatted_output_str)
        logger.info(f"📊 create_formatted_recommendations 응답 크기 - 토큰 수: {token_count}, 추천 수: {len(detailed_recommendations)}")
        
        return f"추천 결과가 성공적으로 생성되었습니다. (저장 파일: {filename})\n\n{formatted_output_str}"

    except Exception as e:
        logger.error(f"create_formatted_recommendations 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"추천 생성 및 포맷팅 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def summarize_board_by_id(board_id: str) -> str:
    """
    주어진 채용공고 ID에 해당하는 공고의 주요 정보를 요약하여 반환합니다.
    Args:
        board_id (str): 요약할 채용공고의 고유 ID.
    Returns:
        str: 공고의 주요 정보 요약 문자열 또는 오류 메시지.
    """
    logger.info(f"도구 실행: summarize_board_by_id, ID: {board_id}")
    try:
        board_info_json_str = await get_board_by_id(board_id)
        board_info = json.loads(board_info_json_str)

        if not board_info.get("success") or "data" not in board_info:
            return f"공고 ID {board_id} 정보를 가져오는데 실패했습니다: {board_info.get('error', '데이터 없음')}"

        data = board_info["data"]
        summary_parts = [
            f"'{data.get('title', '제목 없음')}' 공고(ID: {data.get('board_id')}) 요약:",
            f"- 병원명: {data.get('hospital_name', '정보 없음')}",
            f"- 지역: {data.get('region', '정보 없음')}",
            f"- 진료과: {data.get('department', '정보 없음')}",
            f"- 고용형태: {data.get('employment_type', '정보 없음')}",
            f"- 실제 근무유형: {data.get('work_type', '정보 없음')}",
        ]
        if data.get('pay_details'): # 급여 정보가 있을 때만 추가
            summary_parts.append(f"- 급여: {data.get('pay_details')}")
        summary_parts.append(f"- 주요 내용: {data.get('description_short', '정보 없음')}")
        
        summary = "\n".join(summary_parts)
        logger.info(f"summarize_board_by_id 성공: ID {board_id}")
        return summary
    except json.JSONDecodeError:
        logger.error(f"공고 ID {board_id} 정보 조회 결과 JSON 파싱 실패 (summarize_board_by_id): {board_info_json_str}")
        return f"공고 ID {board_id} 정보 처리 중 오류가 발생했습니다 (파싱 실패)."
    except Exception as e:
        logger.error(f"summarize_board_by_id 오류: {e}", exc_info=True)
        return f"공고 요약 중 오류 발생: {str(e)}"

@app.tool()
async def recommend_popular_jobs(user_id: Optional[str] = None, size: int = 5) -> str:
    """
    인기 있는 채용공고를 추천합니다. 사용자 ID가 제공되면 해당 사용자의 전문과를 고려하여 추천합니다.
    인기 순서는 가상의 'metadata.view_count' 필드를 기준으로 합니다. (실제 필드명 및 타입 확인 필요)
    Args:
        user_id (str, optional): 사용자 ID. 제공되면 사용자의 전문과를 필터 조건으로 사용합니다.
        size (int, optional): 추천할 공고 수 (기본값: 5).
    Returns:
        str: 인기 공고 추천 목록 문자열 또는 오류 메시지.
    """
    logger.info(f"도구 실행: recommend_popular_jobs, 사용자 ID: {user_id}, 개수: {size}")
    try:
        filters = {}
        user_department_for_log = "전체"
        if user_id:
            user_info_json_str = await get_user_by_id(user_id)
            user_info = json.loads(user_info_json_str)
            if user_info.get("success") and user_info["data"].get("department"):
                user_department = user_info["data"]["department"]
                if user_department: # 전문과 정보가 실제로 있을 때만 필터 추가
                    filters["department"] = user_department
                    user_department_for_log = user_department
                    logger.info(f"사용자 {user_id}의 전문과 '{user_department}' 기준으로 인기 공고 검색")
                else:
                    logger.info(f"사용자 {user_id}의 전문과 정보가 비어있습니다. 전체 인기 공고를 검색합니다.")
            else:
                logger.warning(f"사용자 ID {user_id}의 프로필 또는 전문과 정보를 가져오지 못했습니다. 전체 인기 공고를 검색합니다.")
        
        # 'metadata.view_count' 필드가 숫자 타입이고 존재한다고 가정.
        sort_options = [{"metadata.view_count": {"order": "desc", "missing": "_last", "unmapped_type": "long"}}]

        query = os_service.build_search_query(
            index_hint="recruit",
            filters=filters, # department 필터 적용
            size=min(size, 20),
            sort_options=sort_options
        )
        
        result = await os_service.search("recruit", query)

        if not result["hits"]["hits"]:
            return f"{user_department_for_log} 분야에서 추천할 인기 공고를 찾을 수 없습니다."

        output_lines = [f"'{user_department_for_log}' 분야의 인기 채용공고 {len(result['hits']['hits'])}개를 추천합니다:"]
        for i, hit in enumerate(result['hits']['hits'], 1):
            board_detail = _format_board_details_to_dict(hit)
            output_lines.append(
                f"\n🌟 {i}. {board_detail.get('title', 'N/A')} (ID: {board_detail.get('board_id', 'N/A')})"
            )
            output_lines.extend([
                f"   - 병원: {board_detail.get('hospital_name', 'N/A')}, 지역: {board_detail.get('region', 'N/A')}",
                f"   - 진료과: {board_detail.get('department', 'N/A')}",
                # view_count가 실제로 숫자일 때만 의미있게 표시
                f"   - 인기도(조회수): {board_detail.get('view_count', '정보 없음') if isinstance(board_detail.get('view_count'), int) else '집계 중'}"
            ])
        
        logger.info(f"recommend_popular_jobs 성공: {len(result['hits']['hits'])}개 공고 추천 (분야: {user_department_for_log})")
        return "\n".join(output_lines)

    except opensearch_exceptions.RequestError as e:
        # OpenSearch 에러 메시지에 따라 더 구체적인 사용자 안내 가능
        if "Field [metadata.view_count] is not a numeric type" in str(e) or \
           "No mapping found for [metadata.view_count]" in str(e) or \
           "can't load numeric doc values" in str(e):
             logger.error(f"인기 공고 추천 오류: 'metadata.view_count' 필드로 정렬할 수 없습니다. 필드가 없거나 숫자 타입이 아닐 수 있습니다. OpenSearch 매핑을 확인하세요. 에러: {e}", exc_info=False)
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
