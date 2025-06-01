# 의료진 채용 검색 MCP 시스템

의료진 채용 공고를 지능적으로 검색하고 추천하는 Model Context Protocol (MCP) 기반 AI 에이전트 시스템입니다.

## 📋 개요

이 시스템은 의료진들이 자신에게 맞는 채용 공고를 쉽게 찾을 수 있도록 도와주는 AI 어시스턴트입니다. 사용자의 자연어 질문을 이해하고 OpenSearch를 통해 관련 공고를 검색하여 맞춤형 결과를 제공합니다.

## 🏗️ 시스템 구조

```
├── server.py          # MCP 서버 (도구 정의)
├── agent.py           # MCP 에이전트 (LLM + 도구 조합)
├── .env              # 환경 변수 설정
└── README.md         # 사용 가이드
```

## 🔧 필요 조건

### Python 패키지
```bash
pip install -r requirements.txt
```

필요한 패키지:
- `mcp`
- `langchain-mcp-adapters`
- `langchain-openai`
- `langgraph`
- `opensearch-py`
- `python-dotenv`

### 환경 변수 설정

`.env` 파일에 다음 내용을 설정하세요:

```bash
# OpenAI API
OPENAI_API_KEY="Your key"

# OpenSearch 설정
OPENSEARCH_HOST=opensearch.medigate.net
OPENSEARCH_PORT=9200
OPENSEARCH_USER=medigate
OPENSEARCH_PASSWORD=Soakaeofh12!@
```

## 🚀 사용 방법

### 1. 직접 실행 (테스트용)

```bash
python agent.py
```

### 2. API 형태로 사용

```python
from agent import process_medical_recruit_request

# 예시 1: 조건별 검색
result = await process_medical_recruit_request(
    board_id=None,
    user_id=None, 
    prompt="서울 지역 마취통증의학과 연봉 3000만원 이상 공고 찾아줘"
)

# 예시 2: 현재 공고와 유사한 공고 찾기
result = await process_medical_recruit_request(
    board_id="current_job_123",
    user_id=None,
    prompt="지금 보고 있는 공고와 비슷한 조건의 공고 찾아줘"
)

# 예시 3: 맞춤 추천
result = await process_medical_recruit_request(
    board_id=None,
    user_id="user_456",
    prompt="나의 조건에 적합한 공고를 찾아줘"
)
```

## 🛠️ 제공 기능 (도구)

### 1. create_recruit_search_query
공고 검색을 위한 쿼리 생성
- 지역, 진료과, 연봉, 경력, 고용형태 등으로 필터링
- 의미론적 검색 키워드 지원

### 2. create_user_search_query  
사용자 기반 검색 쿼리 생성
- 사용자 ID, 전문과목, 경력, 선호지역 등으로 필터링
- 유사한 사용자 찾기에 활용

### 3. search_recruits
공고 데이터베이스에서 실제 검색 수행
- OpenSearch 기반 하이브리드 검색 (키워드 + 벡터)
- 검색 결과 점수와 함께 반환

### 4. search_users
사용자 데이터베이스에서 검색 수행
- 유사한 배경의 사용자 찾기
- 지원 이력 분석에 활용

### 5. get_recruit_by_id
특정 공고 ID로 상세 정보 조회
- 현재 보고 있는 공고 정보 가져오기
- 유사 공고 찾기의 기준점으로 활용

### 6. format_results
검색 결과를 사용자 친화적으로 포맷팅
- brief, summary, detailed 형태 지원
- 가독성 좋은 출력 형식 제공

## 📊 주요 시나리오

### A. 조건별 공고 검색
```
"서울 지역에 연봉 3000만원 이상 마취통증의학과 공고 찾아줘"

처리 과정:
1. 조건 추출 (지역=서울, 연봉≥3000만원, 진료과=마취통증의학과)
2. 검색 쿼리 생성
3. 공고 DB 검색
4. 결과 포맷팅
```

### B. 유사 공고 찾기
```
"지금 보고 있는 공고와 비슷한 조건의 공고 찾아줘"

처리 과정:  
1. 현재 공고(board_id) 정보 조회
2. 유사 조건 추출하여 검색 쿼리 생성
3. 비슷한 공고 검색
4. 결과 포맷팅
```

### C. 인기 공고 찾기
```
"나와 비슷한 조건의 사람들이 많이 지원한 공고 찾아줘"

처리 과정:
1. 유사한 사용자 검색
2. 지원 이력(applied_jobs) 수집
3. 해당 공고들 조회
4. 인기도순 정렬 및 포맷팅
```

### D. 맞춤 추천
```
"나의 조건에 적합한 공고를 찾아줘"

처리 과정:
1. 사용자 정보 조회 (전문과, 경력 등)
2. 기본 조건 + 의미론 검색으로 쿼리 생성
3. 맞춤 공고 검색
4. 추천 이유와 함께 포맷팅
```

## 🧪 테스트 시나리오

### 기본 검색 테스트
```python
test_cases = [
    "서울 내과 공고 찾아줘",
    "연봉 5000만원 이상 정형외과 공고 있어?", 
    "부산 지역 응급의학과 계약직 공고 보여줘",
    "경력 3년 이하 신경외과 공고 추천해줘"
]
```

### 의미론 검색 테스트
```python
semantic_test_cases = [
    "초음파 경험이 있는 의사가 지원할 수 있는 공고 찾아줘",
    "대장내시경 시술 가능한 곳에서 일하고 싶어",
    "로봇수술 경험을 쌓을 수 있는 병원 공고 있어?",
    "야간근무가 적은 공고 추천해줘"
]
```

### 컨텍스트 기반 테스트
```python
context_test_cases = [
    {
        "board_id": "job_123",
        "prompt": "이 공고보다 연봉이 높은 비슷한 공고 찾아줘"
    },
    {
        "user_id": "user_456", 
        "prompt": "내 경력에 맞는 공고 중에서 서울 지역만 보여줘"
    }
]
```

## 🔍 OpenSearch 인덱스 구조

### 공고 인덱스: `recruit_text-embedding-3-small_1536_100000_300_20250529_150924`
```json
{
  "title": "공고 제목",
  "hospital_name": "병원명",
  "department": "진료과",
  "region": "지역", 
  "salary": 50000000,
  "employment_type": "정규직",
  "required_experience": 3,
  "description": "공고 설명",
  "benefits": "복리후생",
  "requirements": "지원 조건",
  "embedding_vector": [0.1, 0.2, ...]
}
```

### 사용자 인덱스: `resume_text-embedding-3-large_3072_100000_300_20250221_175445`
```json
{
  "user_id": "사용자ID",
  "department": "전문과목",
  "experience_years": 5,
  "preferred_region": "선호지역",
  "applied_jobs": ["job_1", "job_2"],
  "skills": ["초음파", "내시경"],
  "embedding_vector": [0.1, 0.2, ...]
}
```

## ⚠️ 주의사항

1. **OpenAI API 키**: 실제 사용 시 보안을 위해 환경변수나 안전한 저장소 사용
2. **OpenSearch 연결**: 네트워크 설정과 인증 정보 확인 필요
3. **인덱스 이름**: 실제 환경의 인덱스 이름으로 변경 필요
4. **에러 처리**: 네트워크 오류, 검색 실패 등에 대한 적절한 처리 필요

## 🔧 확장 가능한 기능

1. **추가 필터**: 병원 규모, 병상 수, 특수 장비 보유 여부 등
2. **추천 알고리즘**: 협업 필터링, 컨텐츠 기반 필터링 고도화
3. **알림 기능**: 맞춤 공고 자동 알림
4. **지원 관리**: 지원 현황 추적 및 관리
5. **분석 기능**: 채용 시장 트렌드 분석

## 📞 문의

시스템 관련 문의사항이나 개선 제안은 개발팀으로 연락해주세요.
