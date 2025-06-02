#!/usr/bin/env python3

import os
import ssl
import json
import copy
import re
import logging
import time # time 모듈 추가 (JSON 파일명 생성용)
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # server.py 내에서는 __name__ 사용

app = FastMCP("의료진 채용 검색 서버")

class OpenSearchService:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key: raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        self._connect()
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=30, openai_api_key=openai_api_key)
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
            if field == "region" and value: filter_clauses.append({"wildcard": {"metadata.REGION_NAME": f"*{value}*"}})
            elif field == "department" and value: filter_clauses.append({"wildcard": {"metadata.SPECIALTIES": f"*{value}*"}})
            # employment_type 필터는 이전 요청에 의해 주석 처리된 상태로 유지
            # elif field == "employment_type" and value: filter_clauses.append({"wildcard": {"metadata.REGULAR_STATUS": f"*{value}*"}})
            elif field == "hospital_name" and value: filter_clauses.append({"wildcard": {"metadata.ORGANIZATION_NAME": f"*{value}*"}})
        return filter_clauses
    
    def build_search_query(self, index_hint: str, filters: Dict[str, Any] = None, semantic_query: str = None, query_vector: List[float] = None, exclude_filters: List[Dict[str, Any]] = None, size: int = 10) -> Dict[str, Any]:
        query = {"size": min(size, 50), "_source": {"excludes": ["vector_field"]}}
        bool_query = {"must": [], "filter": [], "should": [], "must_not": []}
        
        if filters: bool_query["filter"].extend(self.build_filter_conditions(filters))
        if exclude_filters: bool_query["must_not"].extend(exclude_filters)
        if query_vector: bool_query["must"].append({"knn": {"vector_field": {"vector": query_vector, "k": 10}}})
        if semantic_query and not query_vector: bool_query["must"].append({"multi_match": {"query": semantic_query, "fields": ["text", "metadata.*"]}})
        
        query["query"] = {"bool": bool_query} if any(bool_query.values()) else {"match_all": {}}
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
        except Exception as e:
            logger.error(f"OpenSearch 검색 오류: {e}", exc_info=True)
            return {"hits": {"hits": [], "total": {"value": 0}}}

os_service = OpenSearchService()

@app.tool()
async def get_board_by_id(board_id: str) -> str:
    logger.info(f"도구 실행: get_board_by_id, ID: {board_id}")
    try:
        search_query = {"query": {"term": {"metadata.BOARD_IDX": board_id}}, "size": 1, "_source": {"excludes": ["vector_field"]}}
        result = await os_service.search("recruit", search_query)
        if not result["hits"]["hits"]: return json.dumps({"success": False, "error": f"공고 ID {board_id} 없음"}, ensure_ascii=False)
        
        hit = result["hits"]["hits"][0]; source = hit["_source"]; metadata = source.get("metadata", {}); text_content = source.get("text", "")
        title = text_content.split("||")[0].replace("TITLE => ", "").strip() if "TITLE =>" in text_content else ""
        
        formatted_result = {"success": True, "data": {
            "board_id": metadata.get("BOARD_IDX", board_id), "title": title, "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
            "department": metadata.get("SPECIALTIES", ""), "region": metadata.get("REGION_NAME", ""), 
            "employment_type": metadata.get("REGULAR_STATUS", ""), # REGULAR_STATUS 또는 WORK_TYPE. 현재 build_filter_conditions에서는 employment_type 필터 주석처리됨.
            "work_type_actual": metadata.get("WORK_TYPE", ""), # 실제 공고의 WORK_TYPE
            "pay_details": metadata.get("PAY_DETAILS", ""), "work_hours": metadata.get("WORK_HOUR_DETAILS", ""), 
            "address": metadata.get("ADDRESS", ""), "benefits": metadata.get("MEAL_HOUSE_STATUS", ""), 
            "description": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }}
        logger.info(f"get_board_by_id 성공: ID {board_id}")
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"get_board_by_id 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"공고 조회 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def get_user_by_id(user_id: str) -> str:
    logger.info(f"도구 실행: get_user_by_id, ID: {user_id}")
    try:
        search_query = {"query": {"term": {"metadata.U_ID": user_id}}, "size": 1, "_source": {"excludes": ["vector_field"]}}
        result = await os_service.search("user", search_query)
        if not result["hits"]["hits"]: return json.dumps({"success": False, "error": f"사용자 ID {user_id} 없음"}, ensure_ascii=False)
        
        hit = result["hits"]["hits"][0]; metadata = hit["_source"]["metadata"]; profile_text = hit["_source"].get("text", "")
        formatted_result = {"success": True, "data": {
            "user_id": metadata.get("U_ID", user_id), "department": metadata.get("SPECIALTY", ""),
            "preferred_region": metadata.get("HOPE_LOCATION", ""), "experience_level": metadata.get("EXPERIENCE", ""),
            "work_type": metadata.get("WORK_TYPE",""), # 사용자 프로필의 WORK_TYPE
            "profile_text": profile_text[:300] + "..." if len(profile_text) > 300 else profile_text
        }}
        logger.info(f"get_user_by_id 성공: ID {user_id}")
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"get_user_by_id 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"사용자 조회 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def create_and_search_recruits(
    region: str = None, department: str = None, employment_type: str = None, 
    hospital_name: str = None, semantic_keywords: str = None, 
    exclude_board_id: str = None, size: int = 10
) -> str:
    params = locals()
    log_params = {k:v for k,v in params.items() if v is not None and k != 'semantic_keywords'}
    if semantic_keywords: log_params['semantic_keywords_present'] = True
    logger.info(f"도구 실행: create_and_search_recruits, Params: {log_params}")
    try:
        filters = {k:v for k,v in {"region":region, "department":department, 
                                   "employment_type":employment_type, 
                                   "hospital_name":hospital_name}.items() if v}
        exclude_filters = [{"term": {"metadata.BOARD_IDX": exclude_board_id}}] if exclude_board_id else []
        query_vector = await os_service.create_embedding(semantic_keywords) if semantic_keywords else None
        
        query = os_service.build_search_query("recruit", filters, semantic_keywords, query_vector, exclude_filters, size)
        result = await os_service.search("recruit", query)
        
        formatted_results = []; board_ids = []
        t1=result["hits"]["hits"]
        logger.info(f"💾 Board IDs 저장됨: {t1}")
        for hit in result["hits"]["hits"]:
            source = hit["_source"]; metadata = source.get("metadata", {}); text_content = source.get("text", "")
            title = text_content.split("||")[0].replace("TITLE => ", "").strip() if "TITLE =>" in text_content else ""
            board_id_val = metadata.get("BOARD_IDX", hit["_id"]); board_ids.append(board_id_val)
            
            formatted_results.append({
                "board_id": board_id_val, "title": title, 
                "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
                "department": metadata.get("SPECIALTIES", ""), "region": metadata.get("REGION_NAME", ""), 
                "employment_type": metadata.get("REGULAR_STATUS", ""), # build_filter_conditions 주석과 일치하도록. 만약 WORK_TYPE으로 필터링한다면 이것도 WORK_TYPE으로.
                "work_type_actual": metadata.get("WORK_TYPE",""), # 실제 공고의 WORK_TYPE 정보도 제공
                "pay_details": metadata.get("PAY_DETAILS", ""), 
                "description": text_content[:100] + "..." if len(text_content) > 100 else text_content
            })
            logger.info(f"t2: {formatted_results}")
        
        result_data = {
            "success": True, "total_hits": result["hits"]["total"]["value"], 
            "results": formatted_results, "board_ids": board_ids
        }
        logger.info(f"t3: {result_data}")
        
        try:
            board_ids_data = {
                "search_timestamp": time.time(),
                "search_params": {
                    "region": region, "department": department, "employment_type": employment_type,
                    "hospital_name": hospital_name, "semantic_keywords_present": bool(semantic_keywords),
                    "exclude_board_id": exclude_board_id, "size": size
                },
                "total_results_returned": len(board_ids),
                "board_ids": board_ids
            }
            logger.info(f"t4: {board_ids_data}")
            filename = f"search_results_board_ids_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(board_ids_data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Board IDs 저장됨: {filename}")
        except Exception as e:
            logger.error(f"❌ JSON 파일 저장 오류: {e}")
        
        logger.info(f"create_and_search_recruits 성공: 반환된 결과 {len(formatted_results)}개 (총 검색 {result['hits']['total']['value']}개)")
        return json.dumps(result_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"create_and_search_recruits 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"공고 검색 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def create_and_search_users(department: str = None, preferred_region: str = None, semantic_keywords: str = None, size: int = 10) -> str:
    params = locals(); log_params = {k:v for k,v in params.items() if v is not None and k != 'semantic_keywords'}
    if semantic_keywords: log_params['semantic_keywords_present'] = True
    logger.info(f"도구 실행: create_and_search_users, Params: {log_params}")
    try:
        user_query_bool_filter = []
        if department: user_query_bool_filter.append({"wildcard": {"metadata.SPECIALTY": f"*{department}*"}})
        if preferred_region: user_query_bool_filter.append({"wildcard": {"metadata.HOPE_LOCATION": f"*{preferred_region}*"}})
        user_query_bool_must = []
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)
            if query_vector: user_query_bool_must.append({"knn": {"vector_field": {"vector": query_vector, "k": 10}}})

        user_query_body = {"size": min(size, 50), "_source": {"excludes": ["vector_field"]}}
        user_query_bool_conditions = {}
        if user_query_bool_filter: user_query_bool_conditions["filter"] = user_query_bool_filter
        if user_query_bool_must: user_query_bool_conditions["must"] = user_query_bool_must
        user_query_body["query"] = {"bool": user_query_bool_conditions} if user_query_bool_conditions else {"match_all": {}}
        
        result = await os_service.search("user", user_query_body)
        formatted_results = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]; metadata = source.get("metadata", {}); profile_text = source.get("text", "")
            formatted_results.append({
                "user_id": metadata.get("U_ID", hit["_id"]), "department": metadata.get("SPECIALTY", ""),
                "preferred_region": metadata.get("HOPE_LOCATION", ""), "experience_level": metadata.get("EXPERIENCE", ""),
                "work_type": metadata.get("WORK_TYPE",""),
                "profile_text": profile_text[:200] + "..." if len(profile_text) > 200 else profile_text
            })
        result_data = {"success": True, "total_hits": result["hits"]["total"]["value"], "results": formatted_results}
        logger.info(f"create_and_search_users 성공: {len(formatted_results)}개 결과")
        return json.dumps(result_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"create_and_search_users 오류: {e}", exc_info=True)
        return json.dumps({"success": False, "error": f"사용자 검색 오류: {str(e)}"}, ensure_ascii=False)

@app.tool()
async def format_search_results(results_json, format_type: str = "summary") -> str:
    logger.info(f"도구 실행: format_search_results, Type: {format_type}, Input size: {len(str(results_json))}")
    try:
        data = json.loads(results_json) if isinstance(results_json, str) else results_json if isinstance(results_json, dict) else None
        if not data or not data.get("success", False) or "results" not in data: return "포맷팅할 결과가 없거나 유효하지 않습니다."
        results = data["results"]; total_hits = data.get("total_hits", len(results)); output_lines = []
        
        if format_type == "brief":
            output_lines.append(f"총 {total_hits}개의 결과 요약 (최대 5개 표시):")
            for i, r in enumerate(results[:5], 1):
                if "hospital_name" in r: output_lines.append(f"{i}. {r.get('title', 'N/A')} - {r.get('hospital_name', 'N/A')} ({r.get('region', 'N/A')}|{r.get('employment_type', 'N/A')})")
                else: output_lines.append(f"{i}. 사용자 {r.get('user_id', 'N/A')} ({r.get('preferred_region', 'N/A')}|{r.get('department', 'N/A')})")
        elif format_type == "summary":
            output_lines.append(f"총 {total_hits}개의 검색 결과 요약 (최대 10개 표시):")
            for i, r in enumerate(results[:10], 1):
                is_recruit = "hospital_name" in r
                emp_type_key = "employment_type" if is_recruit else "work_type" 
                if is_recruit:
                    output_lines.append(f"\n🏥 {i}. {r.get('title', 'N/A')} (ID: {r.get('board_id', 'N/A')})")
                    output_lines.extend([f"   - 병원: {r.get('hospital_name', 'N/A')}, 지역: {r.get('region', 'N/A')}", 
                                         f"   - 진료과: {r.get('department', 'N/A')}, 고용: {r.get(emp_type_key, 'N/A')}", # 공고의 employment_type
                                         f"   - 실제 근무유형(공고): {r.get('work_type_actual', 'N/A')}", # 공고의 work_type_actual 추가
                                         f"   - 급여: {r.get('pay_details', 'N/A')}"])
                else: # 사용자 결과
                    output_lines.append(f"\n👤 {i}. 사용자 {r.get('user_id', 'N/A')}")
                    output_lines.extend([f"   - 전문: {r.get('department', 'N/A')}, 선호지역: {r.get('preferred_region', 'N/A')}", 
                                         f"   - 경력: {r.get('experience_level', 'N/A')}, 근무유형(사용자): {r.get(emp_type_key, 'N/A')}"]) # 사용자의 work_type
        else: # detailed
            output_lines.append(f"상세 검색 결과 (총 {total_hits}개, 최대 5개 표시):")
            for i, r in enumerate(results[:5], 1):
                is_recruit = "hospital_name" in r
                emp_type_key = "employment_type" if is_recruit else "work_type"
                if is_recruit:
                    output_lines.append(f"\n🏥 {i}. {r.get('title', 'N/A')} (ID: {r.get('board_id', 'N/A')})")
                    output_lines.extend([
                        f"   - 병원명: {r.get('hospital_name', 'N/A')}, 진료과: {r.get('department', 'N/A')}",
                        f"   - 지역: {r.get('region', 'N/A')}, 고용형태: {r.get(emp_type_key, 'N/A')}",
                        f"   - 실제 근무유형(공고): {r.get('work_type_actual', 'N/A')}",
                        f"   - 급여: {r.get('pay_details', 'N/A')}, 근무시간: {r.get('work_hours', '정보 없음')}", 
                        f"   - 주소: {r.get('address', '정보 없음')}", 
                        f"   - 설명: {r.get('description', 'N/A')}"
                    ])
                else: # 사용자 결과
                    output_lines.append(f"\n👤 {i}. 사용자 {r.get('user_id', 'N/A')}")
                    output_lines.extend([
                        f"   - 전문과목: {r.get('department', 'N/A')}, 선호지역: {r.get('preferred_region', 'N/A')}",
                        f"   - 경력: {r.get('experience_level', 'N/A')}, 근무유형(사용자): {r.get(emp_type_key, 'N/A')}",
                        f"   - 프로필: {r.get('profile_text', 'N/A')}"
                    ])
        output = "\n".join(output_lines)
        logger.info(f"format_search_results 성공, Output size: {len(output)}")
        return output
    except Exception as e:
        logger.error(f"format_search_results 오류: {e}", exc_info=True)
        return f"포맷팅 오류: {str(e)}"

if __name__ == "__main__":
    app.run(transport="stdio")
