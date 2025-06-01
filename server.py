#!/usr/bin/env python3

import os
import ssl
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastMCP("의료진 채용 검색 서버")

tool_call_counter = {}

def log_tool_call(tool_name: str, **kwargs):
    if tool_name not in tool_call_counter:
        tool_call_counter[tool_name] = 0
    tool_call_counter[tool_name] += 1
    
    logger.info(f"🔧 도구 호출 [{tool_name}] #{tool_call_counter[tool_name]}")
    
    if kwargs:
        simplified_kwargs = {}
        for key, value in kwargs.items():
            if key == 'vector_field' or 'vector' in key.lower():
                simplified_kwargs[key] = f"<벡터 데이터 {len(value) if hasattr(value, '__len__') else '?'}차원>"
            elif isinstance(value, str) and len(value) > 200:
                simplified_kwargs[key] = f"{value[:200]}... (총 {len(value)}자)"
            elif isinstance(value, (list, dict)) and len(str(value)) > 300:
                simplified_kwargs[key] = f"<{type(value).__name__} 객체 크기: {len(value)}>"
            else:
                simplified_kwargs[key] = value
        
        logger.info(f"📥 INPUT: {simplified_kwargs}")

def log_tool_result(tool_name: str, result: str, truncate_length: int = 500):
    if isinstance(result, str):
        if len(result) > truncate_length:
            logger.info(f"📤 OUTPUT [{tool_name}]: {result[:truncate_length]}... (총 {len(result)}자)")
        else:
            logger.info(f"📤 OUTPUT [{tool_name}]: {result}")
    else:
        logger.info(f"📤 OUTPUT [{tool_name}]: {type(result).__name__} 객체")

class URLParser:
    @staticmethod
    def parse_board_id_from_url(url: str) -> Optional[str]:
        try:
            pattern = r'/recruit/(\d+)'
            match = re.search(pattern, url)
            return match.group(1) if match else None
        except Exception as e:
            logger.error(f"URL 파싱 오류: {e}")
            return None

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
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
            ssl_context=ssl_context
        )

    async def create_embedding(self, text: str) -> List[float]:
        try:
            result = await self.embeddings_model.aembed_query(text)
            return result
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return []

    def build_search_query(self, filter_conditions: Dict[str, Any], semantic_query: str = None, 
                          query_vector: List[float] = None, vector_field_name: str = "vector_field", 
                          default_knn_k: int = 10) -> Dict[str, Any]:
        
        bool_query_parts = {"must": [], "filter": [], "should": [], "must_not": []}

        for field, value in filter_conditions.items():
            if field == "region":
                bool_query_parts["filter"].append({"term": {"metadata.REGION_NAME": value}})
            elif field == "department":
                bool_query_parts["filter"].append({"match": {"metadata.SPECIALTIES": value}})
            elif field == "salary":
                if isinstance(value, dict) and "gte" in value:
                    min_salary_num = value["gte"] // 10000
                    bool_query_parts["should"].extend([
                        {"match": {"metadata.PAY_DETAILS": f"{min_salary_num}만원"}},
                        {"match": {"metadata.PAY_DETAILS": f"{min_salary_num}0만원"}},
                        {"wildcard": {"metadata.PAY_DETAILS": f"*{min_salary_num}*"}},
                    ])
                    bool_query_parts["minimum_should_match"] = 1
            elif field == "employment_type":
                bool_query_parts["filter"].append({"match": {"metadata.REGULAR_STATUS": value}})
            elif field == "hospital_name":
                bool_query_parts["filter"].append({"match": {"metadata.ORGANIZATION_NAME": value}})

        if query_vector and vector_field_name:
            knn_clause = {"knn": {vector_field_name: {"vector": query_vector, "k": default_knn_k}}}
            bool_query_parts["must"].append(knn_clause)

        return {"query": {"bool": bool_query_parts}, "size": 20}

    async def search(self, index_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.search(index=index_name, body=query)
            return response
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return {"hits": {"hits": []}}

os_service = OpenSearchService()

@app.tool()
async def parse_url_for_board_id(url: str) -> str:
    log_tool_call("parse_url_for_board_id", url=url)
    
    try:
        board_id = URLParser.parse_board_id_from_url(url)
        result = json.dumps({
            "success": bool(board_id),
            "board_id": board_id,
            "message": f"URL에서 board_id '{board_id}'를 추출했습니다." if board_id else "URL에서 board_id를 찾을 수 없습니다."
        }, ensure_ascii=False)
        log_tool_result("parse_url_for_board_id", result)
        return result
    except Exception as e:
        error_msg = f"URL 파싱 오류: {str(e)}"
        logger.error(error_msg)
        result = json.dumps({"success": False, "board_id": None, "message": error_msg}, ensure_ascii=False)
        log_tool_result("parse_url_for_board_id", result)
        return result

@app.tool()
async def create_recruit_search_query(region: str = None, department: str = None, min_salary: int = None,
                                    max_salary: int = None, experience_years: int = None, 
                                    employment_type: str = None, semantic_keywords: str = None) -> str:
    log_tool_call("create_recruit_search_query", region=region, department=department, 
                  min_salary=min_salary, semantic_keywords=semantic_keywords)
    
    try:
        filter_conditions = {}
        
        if region: filter_conditions["region"] = region
        if department: filter_conditions["department"] = department
        if min_salary or max_salary:
            salary_range = {}
            if min_salary: salary_range["gte"] = min_salary * 10000
            if max_salary: salary_range["lte"] = max_salary * 10000
            filter_conditions["salary"] = salary_range
        if experience_years is not None: filter_conditions["required_experience"] = {"lte": experience_years}
        if employment_type: filter_conditions["employment_type"] = employment_type

        query_vector = None
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)

        query = os_service.build_search_query(filter_conditions, semantic_keywords, query_vector)
        result = json.dumps(query, ensure_ascii=False, indent=2)
        log_tool_result("create_recruit_search_query", result, 800)
        return result
    except Exception as e:
        error_msg = f"쿼리 생성 오류: {str(e)}"
        logger.error(error_msg)
        log_tool_result("create_recruit_search_query", error_msg)
        return error_msg

@app.tool()
async def search_recruits(query_json) -> str:
    log_tool_call("search_recruits", query_type=type(query_json).__name__)
    
    try:
        if isinstance(query_json, str):
            query = json.loads(query_json)
        elif isinstance(query_json, dict):
            query = query_json
        else:
            error_result = f"지원하지 않는 쿼리 타입: {type(query_json)}"
            log_tool_result("search_recruits", error_result)
            return error_result
        
        if "query" not in query:
            query = {"query": query, "size": 20}
        
        result = await os_service.search(os_service.recruit_index, query)
        formatted_results = []
        
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            
            # 급여 추출
            salary = 0
            pay_details = metadata.get("PAY_DETAILS", "")
            if pay_details:
                salary_match = re.search(r'(\d+)만원', pay_details)
                if salary_match:
                    salary = int(salary_match.group(1)) * 10000
            
            formatted_result = {
                "board_id": metadata.get("BOARD_IDX", hit["_id"]),
                "score": hit["_score"],
                "title": source.get("text", "").split("||")[0].replace("TITLE => ", "").strip() if "TITLE =>" in source.get("text", "") else "",
                "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
                "department": metadata.get("SPECIALTIES", ""),
                "region": metadata.get("REGION_NAME", ""),
                "salary": salary,
                "employment_type": metadata.get("REGULAR_STATUS", ""),
                "description": source.get("text", "")[:200] + "..." if len(source.get("text", "")) > 200 else source.get("text", ""),
                "pay_details": pay_details,
                "work_hours": metadata.get("WORK_HOUR_DETAILS", ""),
                "address": metadata.get("ADDRESS", "")
            }
            formatted_results.append(formatted_result)
        
        result_data = {"total_hits": result["hits"]["total"]["value"], "results": formatted_results}
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        log_tool_result("search_recruits", result_json, 1000)
        return result_json
    except Exception as e:
        error_msg = f"공고 검색 오류: {str(e)}"
        logger.error(error_msg)
        log_tool_result("search_recruits", error_msg)
        return error_msg

@app.tool()
async def get_recruit_by_id(board_id: str) -> str:
    log_tool_call("get_recruit_by_id", board_id=board_id)
    
    try:
        search_query = {"query": {"term": {"metadata.BOARD_IDX": board_id}}, "size": 1}
        result = await os_service.search(os_service.recruit_index, search_query)
        
        if result["hits"]["hits"]:
            hit = result["hits"]["hits"][0]
            source = hit["_source"]
            metadata = source.get("metadata", {})
            
            # 급여 추출
            salary = 0
            pay_details = metadata.get("PAY_DETAILS", "")
            if pay_details:
                salary_match = re.search(r'(\d+)만원', pay_details)
                if salary_match:
                    salary = int(salary_match.group(1)) * 10000
            
            formatted_result = {
                "board_id": metadata.get("BOARD_IDX", board_id),
                "title": source.get("text", "").split("||")[0].replace("TITLE => ", "").strip() if "TITLE =>" in source.get("text", "") else "",
                "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
                "department": metadata.get("SPECIALTIES", ""),
                "region": metadata.get("REGION_NAME", ""),
                "salary": salary,
                "employment_type": metadata.get("REGULAR_STATUS", ""),
                "description": source.get("text", ""),
                "benefits": metadata.get("MEAL_HOUSE_STATUS", ""),
                "pay_details": pay_details,
                "work_hours": metadata.get("WORK_HOUR_DETAILS", ""),
                "address": metadata.get("ADDRESS", "")
            }
            
            result_json = json.dumps(formatted_result, ensure_ascii=False, indent=2)
            log_tool_result("get_recruit_by_id", result_json, 800)
            return result_json
        else:
            error_msg = f"공고 ID {board_id}를 찾을 수 없습니다."
            result_json = json.dumps({"error": error_msg}, ensure_ascii=False)
            log_tool_result("get_recruit_by_id", result_json)
            return result_json
    except Exception as e:
        error_msg = f"공고 조회 오류: {str(e)}"
        logger.error(error_msg)
        result_json = json.dumps({"error": error_msg}, ensure_ascii=False)
        log_tool_result("get_recruit_by_id", result_json)
        return result_json

@app.tool()
async def get_user_by_id(user_id: str) -> str:
    log_tool_call("get_user_by_id", user_id=user_id)
    
    try:
        search_query = {"query": {"term": {"_id": user_id}}, "size": 1}
        result = await os_service.search(os_service.resume_index, search_query)
        
        if result["hits"]["hits"]:
            hit = result["hits"]["hits"][0]
            source = hit["_source"]
            
            formatted_result = {
                "user_id": hit["_id"],
                "department": source.get("department", ""),
                "experience_years": source.get("experience_years", 0),
                "preferred_region": source.get("preferred_region", ""),
                "applied_jobs": source.get("applied_jobs", []),
                "skills": source.get("skills", [])
            }
            
            result_json = json.dumps(formatted_result, ensure_ascii=False, indent=2)
            log_tool_result("get_user_by_id", result_json, 600)
            return result_json
        else:
            error_msg = f"사용자 ID {user_id}를 찾을 수 없습니다."
            result_json = json.dumps({"error": error_msg}, ensure_ascii=False)
            log_tool_result("get_user_by_id", result_json)
            return result_json
    except Exception as e:
        error_msg = f"사용자 조회 오류: {str(e)}"
        logger.error(error_msg)
        result_json = json.dumps({"error": error_msg}, ensure_ascii=False)
        log_tool_result("get_user_by_id", result_json)
        return result_json

@app.tool()
async def create_user_search_query(user_id: str = None, department: str = None, 
                                 experience_years: int = None, preferred_region: str = None, 
                                 semantic_keywords: str = None) -> str:
    log_tool_call("create_user_search_query", user_id=user_id, department=department)
    
    try:
        filter_conditions = {}
        
        if user_id: filter_conditions["user_id"] = user_id
        if department: filter_conditions["department"] = department
        if experience_years is not None: filter_conditions["experience_years"] = {"gte": experience_years}
        if preferred_region: filter_conditions["preferred_region"] = preferred_region

        query_vector = None
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)

        query = os_service.build_search_query(filter_conditions, semantic_keywords, query_vector)
        result = json.dumps(query, ensure_ascii=False, indent=2)
        log_tool_result("create_user_search_query", result, 800)
        return result
    except Exception as e:
        error_msg = f"사용자 쿼리 생성 오류: {str(e)}"
        logger.error(error_msg)
        log_tool_result("create_user_search_query", error_msg)
        return error_msg

@app.tool()
async def search_users(query_json) -> str:
    log_tool_call("search_users", query_type=type(query_json).__name__)
    
    try:
        if isinstance(query_json, str):
            query = json.loads(query_json)
        elif isinstance(query_json, dict):
            query = query_json
        else:
            error_result = f"지원하지 않는 쿼리 타입: {type(query_json)}"
            log_tool_result("search_users", error_result)
            return error_result
            
        result = await os_service.search(os_service.resume_index, query)
        formatted_results = []
        
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            formatted_result = {
                "user_id": hit["_id"],
                "score": hit["_score"],
                "department": source.get("department", ""),
                "experience_years": source.get("experience_years", 0),
                "preferred_region": source.get("preferred_region", ""),
                "applied_jobs": source.get("applied_jobs", []),
                "skills": source.get("skills", [])
            }
            formatted_results.append(formatted_result)
        
        result_data = {"total_hits": result["hits"]["total"]["value"], "results": formatted_results}
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        log_tool_result("search_users", result_json, 800)
        return result_json
    except Exception as e:
        error_msg = f"사용자 검색 오류: {str(e)}"
        logger.error(error_msg)
        log_tool_result("search_users", error_msg)
        return error_msg

@app.tool()
async def format_results(results_json, format_type: str = "summary") -> str:
    log_tool_call("format_results", format_type=format_type)
    
    try:
        if isinstance(results_json, str):
            data = json.loads(results_json)
        elif isinstance(results_json, dict):
            data = results_json
        else:
            error_result = f"지원하지 않는 결과 타입: {type(results_json)}"
            log_tool_result("format_results", error_result)
            return error_result
        
        if "results" not in data:
            error_result = "포맷팅할 결과가 없습니다."
            log_tool_result("format_results", error_result)
            return error_result
        
        results = data["results"]
        total_hits = data.get("total_hits", len(results))
        
        if format_type == "brief":
            output = f"총 {total_hits}개의 결과가 있습니다.\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:
                    output += f"{i}. {result.get('title', '제목 없음')} - {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   📍 {result.get('region', '지역 없음')} | 💰 {result.get('salary', 0):,}원\n\n"
        
        elif format_type == "summary":
            output = f"📋 총 {total_hits}개의 검색 결과\n\n"
            for i, result in enumerate(results[:10], 1):
                if "hospital_name" in result:
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n\n"
        
        else:  # detailed
            output = f"📋 상세 검색 결과 (총 {total_hits}개)\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원명: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n"
                    output += f"   설명: {result.get('description', '설명 없음')}\n"
                    output += f"   공고ID: {result.get('board_id', 'ID 없음')}\n\n"
        
        log_tool_result("format_results", output, 1200)
        return output
    except Exception as e:
        error_msg = f"포맷팅 오류: {str(e)}"
        logger.error(error_msg)
        log_tool_result("format_results", error_msg)
        return error_msg

if __name__ == "__main__":
    def print_tool_stats():
        if tool_call_counter:
            logger.info("=== 도구 호출 통계 ===")
            for tool_name, count in tool_call_counter.items():
                logger.info(f"{tool_name}: {count}회")
    
    try:
        app.run(transport="stdio")
    finally:
        print_tool_stats()
