#!/usr/bin/env python3

import os
import ssl
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch

load_dotenv()

# 로그 레벨 환경변수가 소문자일 경우 대문자로 변경
if os.getenv('LOG_LEVEL'):
    os.environ['LOG_LEVEL'] = os.getenv('LOG_LEVEL').upper()

app = FastMCP("의료진 채용 검색 서버")

class OpenSearchService:
    def __init__(self, hosts=None, http_auth=None, use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=30):
        load_dotenv()
        
        # OpenAI API 키 설정
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.client = None
        self._connect(hosts, http_auth, use_ssl, verify_certs, ssl_show_warn, timeout)
        
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            request_timeout=30,
            openai_api_key=openai_api_key
        )
        
        self.recruit_index = "recruit_text-embedding-3-small_1536_100000_300_20250529_150924"
        self.resume_index = "resume_text-embedding-3-large_3072_100000_300_20250221_175445"

    def _connect(self, hosts, http_auth, use_ssl, verify_certs, ssl_show_warn, timeout):
        opensearch_host = os.getenv("OPENSEARCH_HOST", 'opensearch.medigate.net')
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", 9200))
        opensearch_user = os.getenv("OPENSEARCH_USER", 'medigate')
        opensearch_password = os.getenv("OPENSEARCH_PASSWORD", 'Soakaeofh12!@')

        actual_hosts = hosts or [{'host': opensearch_host, 'port': opensearch_port}]
        actual_auth = http_auth or (opensearch_user, opensearch_password)

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        self.client = OpenSearch(
            hosts=actual_hosts,
            http_auth=actual_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=ssl_show_warn,
            timeout=timeout,
            ssl_context=ssl_context
        )

    async def create_embedding(self, text: str) -> List[float]:
        try:
            return await self.embeddings_model.aembed_query(text)
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return []

    def build_search_query(self, filter_conditions: Dict[str, Any], semantic_query: str = None, 
                          query_vector: List[float] = None, vector_field_name: str = "vector_field", 
                          default_knn_k: int = 10) -> Dict[str, Any]:
        bool_query_parts = {
            "must": [],
            "filter": [],
            "should": [],
            "must_not": []
        }

        # 실제 필드 구조에 맞게 수정
        for field, value in filter_conditions.items():
            # metadata 안의 필드들로 변경
            if field == "region":
                bool_query_parts["filter"].append({"term": {"metadata.REGION_NAME": value}})
            elif field == "department":
                # SPECIALTIES 필드에서 해당 진료과 검색
                bool_query_parts["filter"].append({"match": {"metadata.SPECIALTIES": value}})
            elif field == "salary":
                if isinstance(value, dict):
                    # 급여는 텍스트로 저장되어 있어서 범위 검색이 어려움
                    # 일단 키워드 매칭으로 처리
                    if "gte" in value:
                        bool_query_parts["must"].append({
                            "script": {
                                "script": {
                                    "source": """
                                    String payDetails = params._source.metadata.PAY_DETAILS;
                                    if (payDetails == null) return false;
                                    
                                    // 숫자 추출 로직
                                    Pattern pattern = Pattern.compile("(\\d+)만원");
                                    Matcher matcher = pattern.matcher(payDetails);
                                    if (matcher.find()) {
                                        int salary = Integer.parseInt(matcher.group(1));
                                        return salary >= params.min_salary;
                                    }
                                    return false;
                                    """,
                                    "params": {
                                        "min_salary": value["gte"] // 10000  # 만원 단위로 변환
                                    }
                                }
                            }
                        })
            elif field == "employment_type":
                bool_query_parts["filter"].append({"match": {"metadata.REGULAR_STATUS": value}})
            elif field == "hospital_name":
                bool_query_parts["filter"].append({"match": {"metadata.ORGANIZATION_NAME": value}})

        if query_vector and vector_field_name:
            knn_clause = {
                "knn": {
                    vector_field_name: {
                        "vector": query_vector,
                        "k": default_knn_k
                    }
                }
            }
            bool_query_parts["must"].append(knn_clause)

        query = {
            "query": {
                "bool": bool_query_parts
            },
            "size": 20
        }

        return query

    async def search(self, index_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.search(index=index_name, body=query)
            return response
        except Exception as e:
            print(f"검색 오류: {e}")
            return {"hits": {"hits": []}}

os_service = OpenSearchService()

@app.tool()
async def create_recruit_search_query(
    region: str = None,
    department: str = None,
    min_salary: int = None,
    max_salary: int = None,
    experience_years: int = None,
    employment_type: str = None,
    semantic_keywords: str = None
) -> str:
    """
    공고 검색을 위한 쿼리를 생성합니다.
    
    Args:
        region: 지역 (예: 서울, 부산)
        department: 진료과목 (예: 마취통증의학과, 내과)
        min_salary: 최소 연봉 (단위: 만원)
        max_salary: 최대 연봉 (단위: 만원)
        experience_years: 경력 연수
        employment_type: 고용형태 (정규직, 계약직 등)
        semantic_keywords: 의미론적 검색 키워드 (예: 초음파, 내시경)
    
    Returns:
        검색 쿼리 JSON 문자열
    """
    try:
        filter_conditions = {}
        
        if region:
            filter_conditions["region"] = region
        if department:
            filter_conditions["department"] = department
        if min_salary or max_salary:
            salary_range = {}
            if min_salary:
                salary_range["gte"] = min_salary * 10000
            if max_salary:
                salary_range["lte"] = max_salary * 10000
            filter_conditions["salary"] = salary_range
        if experience_years is not None:
            filter_conditions["required_experience"] = {"lte": experience_years}
        if employment_type:
            filter_conditions["employment_type"] = employment_type

        query_vector = None
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)

        query = os_service.build_search_query(
            filter_conditions=filter_conditions,
            semantic_query=semantic_keywords,
            query_vector=query_vector
        )
        
        return json.dumps(query, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"쿼리 생성 오류: {str(e)}"

@app.tool()
async def create_user_search_query(
    user_id: str = None,
    department: str = None,
    experience_years: int = None,
    preferred_region: str = None,
    semantic_keywords: str = None
) -> str:
    """
    사용자 기반 검색을 위한 쿼리를 생성합니다.
    
    Args:
        user_id: 사용자 ID
        department: 전문과목
        experience_years: 경력 연수
        preferred_region: 선호 지역
        semantic_keywords: 의미론적 검색 키워드
    
    Returns:
        검색 쿼리 JSON 문자열
    """
    try:
        filter_conditions = {}
        
        if user_id:
            filter_conditions["user_id"] = user_id
        if department:
            filter_conditions["department"] = department
        if experience_years is not None:
            filter_conditions["experience_years"] = {"gte": experience_years}
        if preferred_region:
            filter_conditions["preferred_region"] = preferred_region

        query_vector = None
        if semantic_keywords:
            query_vector = await os_service.create_embedding(semantic_keywords)

        query = os_service.build_search_query(
            filter_conditions=filter_conditions,
            semantic_query=semantic_keywords,
            query_vector=query_vector
        )
        
        return json.dumps(query, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"사용자 쿼리 생성 오류: {str(e)}"

@app.tool()
async def search_recruits(query_json: str) -> str:
    """
    공고 데이터베이스에서 검색을 수행합니다.
    
    Args:
        query_json: 검색 쿼리 JSON 문자열
    
    Returns:
        검색 결과 JSON 문자열
    """
    try:
        query = json.loads(query_json)
        result = await os_service.search(os_service.recruit_index, query)
        
        formatted_results = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            
            # 급여에서 숫자 추출
            salary = 0
            pay_details = metadata.get("PAY_DETAILS", "")
            if pay_details:
                import re
                salary_match = re.search(r'(\d+)만원', pay_details)
                if salary_match:
                    salary = int(salary_match.group(1)) * 10000  # 원 단위로 변환
            
            formatted_result = {
                "board_id": metadata.get("BOARD_IDX", hit["_id"]),
                "score": hit["_score"],
                "title": source.get("text", "").split("||")[0].replace("TITLE => ", "").strip() if "TITLE =>" in source.get("text", "") else "",
                "hospital_name": metadata.get("ORGANIZATION_NAME", ""),
                "department": metadata.get("SPECIALTIES", ""),
                "region": metadata.get("REGION_NAME", ""),
                "salary": salary,
                "employment_type": metadata.get("REGULAR_STATUS", ""),
                "required_experience": 0,  # 이 필드는 없는 것 같음
                "description": source.get("text", "")[:200] + "..." if len(source.get("text", "")) > 200 else source.get("text", ""),
                "pay_details": pay_details,
                "work_hours": metadata.get("WORK_HOUR_DETAILS", ""),
                "address": metadata.get("ADDRESS", "")
            }
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "total_hits": result["hits"]["total"]["value"],
            "results": formatted_results
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"공고 검색 오류: {str(e)}"

@app.tool()
async def search_users(query_json: str) -> str:
    """
    사용자 데이터베이스에서 검색을 수행합니다.
    
    Args:
        query_json: 검색 쿼리 JSON 문자열
    
    Returns:
        검색 결과 JSON 문자열
    """
    try:
        query = json.loads(query_json)
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
        
        return json.dumps({
            "total_hits": result["hits"]["total"]["value"],
            "results": formatted_results
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"사용자 검색 오류: {str(e)}"

@app.tool()
async def get_recruit_by_id(board_id: str) -> str:
    """
    특정 공고 ID로 공고 정보를 조회합니다.
    
    Args:
        board_id: 공고 ID
    
    Returns:
        공고 정보 JSON 문자열
    """
    try:
        result = os_service.client.get(index=os_service.recruit_index, id=board_id)
        source = result["_source"]
        metadata = source.get("metadata", {})
        
        # 급여에서 숫자 추출
        salary = 0
        pay_details = metadata.get("PAY_DETAILS", "")
        if pay_details:
            import re
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
            "required_experience": 0,
            "description": source.get("text", ""),
            "benefits": metadata.get("MEAL_HOUSE_STATUS", ""),
            "requirements": metadata.get("LABOR_LAW_STATUS", ""),
            "pay_details": pay_details,
            "work_hours": metadata.get("WORK_HOUR_DETAILS", ""),
            "address": metadata.get("ADDRESS", "")
        }
        
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"공고 조회 오류: {str(e)}"

@app.tool()
async def format_results(results_json: str, format_type: str = "summary") -> str:
    """
    검색 결과를 지정된 형식으로 포맷팅합니다.
    
    Args:
        results_json: 검색 결과 JSON 문자열
        format_type: 포맷 타입 (summary, detailed, brief)
    
    Returns:
        포맷된 결과 문자열
    """
    try:
        data = json.loads(results_json)
        
        if "results" not in data:
            return "포맷팅할 결과가 없습니다."
        
        results = data["results"]
        total_hits = data.get("total_hits", len(results))
        
        if format_type == "brief":
            output = f"총 {total_hits}개의 결과가 있습니다.\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"{i}. {result.get('title', '제목 없음')} - {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   📍 {result.get('region', '지역 없음')} | �� {result.get('salary', 0):,}만원\n\n"
        
        elif format_type == "summary":
            output = f"📋 총 {total_hits}개의 검색 결과\n\n"
            for i, result in enumerate(results[:10], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}만원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n\n"
        
        elif format_type == "detailed":
            output = f"📋 상세 검색 결과 (총 {total_hits}개)\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원명: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}만원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n"
                    output += f"   요구경력: {result.get('required_experience', 0)}년\n"
                    output += f"   설명: {result.get('description', '설명 없음')}\n"
                    output += f"   공고ID: {result.get('board_id', 'ID 없음')}\n\n"
        
        return output
    
    except Exception as e:
        return f"포맷팅 오류: {str(e)}"

if __name__ == "__main__":
    app.run(transport="stdio")
