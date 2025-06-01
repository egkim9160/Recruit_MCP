#!/usr/bin/env python3

import os
import ssl
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # DEBUG에서 INFO로 변경하여 로그 간소화
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 로그 레벨 환경변수가 소문자일 경우 대문자로 변경
if os.getenv('LOG_LEVEL'):
    os.environ['LOG_LEVEL'] = os.getenv('LOG_LEVEL').upper()

app = FastMCP("의료진 채용 검색 서버")

# 도구 호출 추적을 위한 카운터
tool_call_counter = {}

def log_tool_call(tool_name: str, **kwargs):
    """도구 호출 로깅 (간소화된 버전)"""
    if tool_name not in tool_call_counter:
        tool_call_counter[tool_name] = 0
    tool_call_counter[tool_name] += 1
    
    logger.info(f"🔧 도구 호출 [{tool_name}] #{tool_call_counter[tool_name]}")
    
    # 벡터 필드나 긴 데이터는 간소화해서 로그
    simplified_kwargs = {}
    for key, value in kwargs.items():
        if key == 'query_length' and value > 500:
            simplified_kwargs[key] = f"{value}자 (긴 쿼리)"
        elif isinstance(value, str) and len(value) > 100:
            simplified_kwargs[key] = f"{value[:100]}... (총 {len(value)}자)"
        else:
            simplified_kwargs[key] = value
    
    logger.info(f"파라미터: {simplified_kwargs}")

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
        
        logger.info(f"OpenSearch 서비스 초기화 완료")
        logger.info(f"공고 인덱스: {self.recruit_index}")
        logger.info(f"이력서 인덱스: {self.resume_index}")

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
        
        logger.info(f"OpenSearch 연결 설정: {actual_hosts}")

    async def create_embedding(self, text: str) -> List[float]:
        try:
            logger.info(f"임베딩 생성 요청: {text[:50]}...")
            result = await self.embeddings_model.aembed_query(text)
            logger.info(f"임베딩 생성 완료: 벡터 차원 {len(result)}")
            return result
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return []

    def build_search_query(self, filter_conditions: Dict[str, Any], semantic_query: str = None, 
                          query_vector: List[float] = None, vector_field_name: str = "vector_field", 
                          default_knn_k: int = 10) -> Dict[str, Any]:
        logger.info(f"검색 쿼리 생성 - 필터: {len(filter_conditions)}개, 벡터 사용: {bool(query_vector)}")
        
        bool_query_parts = {
            "must": [],
            "filter": [],
            "should": [],
            "must_not": []
        }

        # 실제 필드 구조에 맞게 수정
        for field, value in filter_conditions.items():
            logger.info(f"필터 추가: {field} = {value}")
            # metadata 안의 필드들로 변경
            if field == "region":
                bool_query_parts["filter"].append({"term": {"metadata.REGION_NAME": value}})
            elif field == "department":
                # SPECIALTIES 필드에서 해당 진료과 검색
                bool_query_parts["filter"].append({"match": {"metadata.SPECIALTIES": value}})
            elif field == "salary":
                if isinstance(value, dict):
                    if "gte" in value:
                        # 더 유연한 급여 검색 (부분 매칭)
                        min_salary_num = value["gte"] // 10000  # 만원 단위
                        bool_query_parts["should"].extend([
                            {"match": {"metadata.PAY_DETAILS": f"{min_salary_num}만원"}},
                            {"match": {"metadata.PAY_DETAILS": f"{min_salary_num}0만원"}},  # 3000만원
                            {"match": {"metadata.PAY_DETAILS": f"{min_salary_num}00만원"}}, # 3000만원
                            {"wildcard": {"metadata.PAY_DETAILS": f"*{min_salary_num}*"}}, # 와일드카드 검색
                        ])
                        # should 조건 중 최소 1개는 매칭되어야 함
                        if "minimum_should_match" not in bool_query_parts:
                            bool_query_parts["minimum_should_match"] = 1
            elif field == "employment_type":
                bool_query_parts["filter"].append({"match": {"metadata.REGULAR_STATUS": value}})
            elif field == "hospital_name":
                bool_query_parts["filter"].append({"match": {"metadata.ORGANIZATION_NAME": value}})

        if query_vector and vector_field_name:
            logger.info(f"KNN 쿼리 추가: k={default_knn_k}")
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
        
        logger.info(f"생성된 쿼리 크기: {len(json.dumps(query))} 문자")
        return query

    async def search(self, index_name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"검색 실행: 인덱스 {index_name}")
            response = self.client.search(index=index_name, body=query)
            total_hits = response["hits"]["total"]["value"]
            logger.info(f"검색 완료: {total_hits}개 결과")
            return response
        except Exception as e:
            logger.error(f"검색 오류: {e}")
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
    log_tool_call("create_recruit_search_query", 
                  region=region, department=department, min_salary=min_salary,
                  max_salary=max_salary, experience_years=experience_years,
                  employment_type=employment_type, semantic_keywords=semantic_keywords)
    
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
            logger.info(f"의미 검색 키워드로 임베딩 생성: {semantic_keywords}")
            query_vector = await os_service.create_embedding(semantic_keywords)

        query = os_service.build_search_query(
            filter_conditions=filter_conditions,
            semantic_query=semantic_keywords,
            query_vector=query_vector
        )
        
        result = json.dumps(query, ensure_ascii=False, indent=2)
        logger.info(f"쿼리 생성 완료: {len(result)} 문자")
        return result
    
    except Exception as e:
        error_msg = f"쿼리 생성 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

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
    log_tool_call("create_user_search_query",
                  user_id=user_id, department=department, experience_years=experience_years,
                  preferred_region=preferred_region, semantic_keywords=semantic_keywords)
    
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
        
        result = json.dumps(query, ensure_ascii=False, indent=2)
        logger.info(f"사용자 쿼리 생성 완료: {len(result)} 문자")
        return result
    
    except Exception as e:
        error_msg = f"사용자 쿼리 생성 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@app.tool()
async def search_recruits(query_json) -> str:
    """
    공고 데이터베이스에서 검색을 수행합니다.
    
    Args:
        query_json: 검색 쿼리 (JSON 문자열 또는 딕셔너리)
    
    Returns:
        검색 결과 JSON 문자열
    """
    log_tool_call("search_recruits", query_type=type(query_json).__name__)
    
    try:
        # query_json의 타입에 따라 처리
        if isinstance(query_json, str):
            logger.info("문자열 쿼리 받음")
            try:
                query = json.loads(query_json)
                logger.info("JSON 파싱 성공")
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                return f"JSON 파싱 오류: {str(e)}"
        elif isinstance(query_json, dict):
            logger.info("딕셔너리 쿼리 받음")
            query = query_json
        else:
            logger.error(f"지원하지 않는 쿼리 타입: {type(query_json)}")
            return f"지원하지 않는 쿼리 타입: {type(query_json)}"
        
        # 쿼리가 완전한 형태인지 확인
        if "query" not in query:
            logger.warning("쿼리에 'query' 키가 없음. 래핑 시도...")
            query = {"query": query, "size": 20}
        
        result = await os_service.search(os_service.recruit_index, query)
        
        formatted_results = []
        hits = result["hits"]["hits"]
        logger.info(f"원본 검색 결과: {len(hits)}개")
        
        for i, hit in enumerate(hits):
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
            
            # 벡터 필드는 로그에서 제외 (너무 길어서)
            if "vector_field" in source:
                logger.debug(f"결과 {i+1}: 벡터 필드 포함 (1536차원)")
            
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
        
        result_data = {
            "total_hits": result["hits"]["total"]["value"],
            "results": formatted_results
        }
        
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        logger.info(f"공고 검색 완료: {len(formatted_results)}개 포맷된 결과")
        return result_json
    
    except Exception as e:
        error_msg = f"공고 검색 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@app.tool()
async def search_users(query_json) -> str:
    """
    사용자 데이터베이스에서 검색을 수행합니다.
    
    Args:
        query_json: 검색 쿼리 (JSON 문자열 또는 딕셔너리)
    
    Returns:
        검색 결과 JSON 문자열
    """
    log_tool_call("search_users", query_type=type(query_json).__name__)
    
    try:
        # query_json의 타입에 따라 처리
        if isinstance(query_json, str):
            query = json.loads(query_json)
        elif isinstance(query_json, dict):
            query = query_json
        else:
            return f"지원하지 않는 쿼리 타입: {type(query_json)}"
            
        result = await os_service.search(os_service.resume_index, query)
        
        formatted_results = []
        hits = result["hits"]["hits"]
        logger.info(f"사용자 검색 결과: {len(hits)}개")
        
        for i, hit in enumerate(hits):
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
        
        result_data = {
            "total_hits": result["hits"]["total"]["value"],
            "results": formatted_results
        }
        
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        logger.info(f"사용자 검색 완료: {len(formatted_results)}개 결과")
        return result_json
    
    except Exception as e:
        error_msg = f"사용자 검색 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@app.tool()
async def get_recruit_by_id(board_id: str) -> str:
    """
    특정 공고 ID로 공고 정보를 조회합니다.
    
    Args:
        board_id: 공고 ID
    
    Returns:
        공고 정보 JSON 문자열
    """
    log_tool_call("get_recruit_by_id", board_id=board_id)
    
    try:
        logger.info(f"공고 ID로 조회 시작: {board_id}")
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
        
        result_json = json.dumps(formatted_result, ensure_ascii=False, indent=2)
        logger.info(f"공고 조회 완료: {board_id}")
        return result_json
    
    except Exception as e:
        error_msg = f"공고 조회 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@app.tool()
async def format_results(results_json, format_type: str = "summary") -> str:
    """
    검색 결과를 지정된 형식으로 포맷팅합니다.
    
    Args:
        results_json: 검색 결과 (JSON 문자열 또는 딕셔너리)
        format_type: 포맷 타입 (summary, detailed, brief)
    
    Returns:
        포맷된 결과 문자열
    """
    log_tool_call("format_results", format_type=format_type, 
                  results_type=type(results_json).__name__)
    
    try:
        logger.info(f"결과 포맷팅 시작: {format_type} 형식")
        
        # results_json의 타입에 따라 처리
        if isinstance(results_json, str):
            data = json.loads(results_json)
        elif isinstance(results_json, dict):
            data = results_json
        else:
            return f"지원하지 않는 결과 타입: {type(results_json)}"
        
        if "results" not in data:
            logger.warning("포맷팅할 결과가 없습니다.")
            return "포맷팅할 결과가 없습니다."
        
        results = data["results"]
        total_hits = data.get("total_hits", len(results))
        logger.info(f"포맷팅 대상: {len(results)}개 결과")
        
        if format_type == "brief":
            output = f"총 {total_hits}개의 결과가 있습니다.\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"{i}. {result.get('title', '제목 없음')} - {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   📍 {result.get('region', '지역 없음')} | 💰 {result.get('salary', 0):,}원\n\n"
        
        elif format_type == "summary":
            output = f"📋 총 {total_hits}개의 검색 결과\n\n"
            for i, result in enumerate(results[:10], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n\n"
        
        elif format_type == "detailed":
            output = f"📋 상세 검색 결과 (총 {total_hits}개)\n\n"
            for i, result in enumerate(results[:5], 1):
                if "hospital_name" in result:  # 공고 결과
                    output += f"🏥 {i}. {result.get('title', '제목 없음')}\n"
                    output += f"   병원명: {result.get('hospital_name', '병원명 없음')}\n"
                    output += f"   진료과: {result.get('department', '진료과 없음')}\n"
                    output += f"   지역: {result.get('region', '지역 없음')}\n"
                    output += f"   연봉: {result.get('salary', 0):,}원\n"
                    output += f"   고용형태: {result.get('employment_type', '고용형태 없음')}\n"
                    output += f"   요구경력: {result.get('required_experience', 0)}년\n"
                    output += f"   설명: {result.get('description', '설명 없음')}\n"
                    output += f"   공고ID: {result.get('board_id', 'ID 없음')}\n\n"
        
        logger.info(f"포맷팅 완료: {len(output)} 문자")
        return output
    
    except Exception as e:
        error_msg = f"포맷팅 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

if __name__ == "__main__":
    # 도구 호출 통계 출력
    def print_tool_stats():
        if tool_call_counter:
            logger.info("=== 도구 호출 통계 ===")
            for tool_name, count in tool_call_counter.items():
                logger.info(f"{tool_name}: {count}회")
        else:
            logger.info("도구 호출이 없었습니다.")
    
    try:
        app.run(transport="stdio")
    finally:
        print_tool_stats()
