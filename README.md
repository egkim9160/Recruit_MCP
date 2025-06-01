# 의료진 채용 검색 MCP 시스템

의료진 채용 공고를 지능적으로 검색하고 추천하는 Model Context Protocol (MCP) 기반 AI 에이전트 시스템입니다.

## 📋 개요

이 시스템은 의료진들이 자신에게 맞는 채용 공고를 쉽게 찾을 수 있도록 도와주는 AI 어시스턴트입니다. 사용자의 자연어 질문을 이해하고 OpenSearch를 통해 관련 공고를 검색하여 맞춤형 결과를 제공합니다.

### 🆕 v2.0 주요 특징
- **명령행 인터페이스**: 간단한 명령어로 즉시 실행
- **URL 자동 파싱**: 채용 공고 URL에서 board_id 자동 추출
- **맞춤형 추천**: 사용자 프로필 기반 개인화 검색
- **컨텍스트 인식**: 현재 보고 있는 공고와 사용자 정보를 고려한 지능적 검색

## 🏗️ 시스템 구조

```
├── server.py          # MCP 서버 (도구 정의)
├── agent.py           # MCP 에이전트 (LLM + 도구 조합)
├── .env              # 환경 변수 설정
├── requirements.txt   # Python 패키지 목록
└── README.md         # 사용 가이드
```

## 🔧 설치 및 설정

### 1. Python 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일에 다음 내용을 설정하세요:

```bash
# OpenAI API (필수)
OPENAI_API_KEY="your_openai_api_key"

# OpenSearch 설정 (필수)
OPENSEARCH_HOST=opensearch.medigate.net
OPENSEARCH_PORT=9200
OPENSEARCH_USER=medigate
OPENSEARCH_PASSWORD=Soakaeofh12!@

# LangSmith (선택적 - 디버깅용)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY="your_langsmith_key"
LANGSMITH_PROJECT="tracing_RAG"
```

## 🚀 사용 방법

### 기본 명령어 형식
```bash
python agent.py [옵션] --prompt "검색 질문"
```

### 명령행 옵션
- `--prompt` (필수): 사용자 질문/요청
- `--url` (선택): 채용 공고 URL (board_id 자동 추출)
- `--uid` (선택): 사용자 ID (맞춤 추천용)

## 📝 사용 예시

### 1. 기본 검색
```bash
python agent.py --prompt "서울 지역 피부과 공고 추천해줘"
```

**결과:**
```
🏥 1. 서울시 강남구 피부과 전문의 모집
   병원: 강남피부과의원
   진료과: 피부과
   지역: 서울
   연봉: 50,000,000원
   고용형태: 정규직
```

### 2. 특정 공고 기반 검색
```bash
python agent.py --url "https://s-new.medigate.net/recruit/1172036" --prompt "이 공고와 비슷한 조건의 공고 찾아줘"
```

**처리 과정:**
1. URL에서 board_id(1172036) 자동 추출
2. 해당 공고 정보 조회
3. 비슷한 조건(지역, 진료과, 연봉대)으로 검색
4. 유사 공고 목록 제공

### 3. 맞춤형 추천
```bash
python agent.py --uid "medizzang" --prompt "내 조건에 맞는 공고 추천해줘"
```

**처리 과정:**
1. 사용자 프로필 조회
2. 전문과목, 경력, 선호지역 분석
3. 개인화된 검색 수행
4. 맞춤 공고 추천

### 4. 복합 조건 검색
```bash
python agent.py --url "https://s-new.medigate.net/recruit/1172036" --uid "medizzang" --prompt "이 공고보다 연봉이 높은 공고 중에서 내가 갈 수 있는 곳 추천해줘"
```

### 5. 고급 검색 예시

#### 의미론적 검색
```bash
python agent.py --prompt "초음파 경험을 쌓을 수 있는 내과 공고 찾아줘"
```

#### 조건별 검색
```bash
python agent.py --prompt "연봉 5000만원 이상 정형외과 정규직 공고 있어?"
```

#### 지역 세분화 검색
```bash
python agent.py --prompt "부산 지역 응급의학과 야간 근무 적은 곳 보여줘"
```

## 🛠️ 제공 기능 (도구)

### 1. URL 파싱
- **parse_url_for_board_id**: URL에서 board_id 자동 추출
- 지원 형식: `https://s-new.medigate.net/recruit/숫자`

### 2. 공고 관련 도구
- **create_recruit_search_query**: 공고 검색 쿼리 생성
- **search_recruits**: 공고 데이터베이스 검색
- **get_recruit_by_id**: 특정 공고 상세 조회

### 3. 사용자 관련 도구
- **create_user_search_query**: 사용자 기반 검색 쿼리 생성
- **search_users**: 사용자 데이터베이스 검색
- **get_user_by_id**: 특정 사용자 정보 조회

### 4. 결과 처리 도구
- **format_results**: 검색 결과 포맷팅 (brief, summary, detailed)

## 🎯 주요 검색 시나리오

### A. 조건별 공고 검색
```
입력: "서울 지역에 연봉 3000만원 이상 마취통증의학과 공고 찾아줘"

처리 과정:
1. 조건 추출 (지역=서울, 연봉≥3000만원, 진료과=마취통증의학과)
2. 검색 쿼리 생성
3. 공고 DB 검색
4. 결과 포맷팅
```

### B. 유사 공고 찾기
```
입력: --url "https://s-new.medigate.net/recruit/1172036" --prompt "비슷한 공고 찾아줘"

처리 과정:  
1. URL에서 board_id 추출
2. 현재 공고 정보 조회
3. 유사 조건으로 검색 쿼리 생성
4. 비슷한 공고 검색 및 추천
```

### C. 맞춤형 추천
```
입력: --uid "medizzang" --prompt "나에게 적합한 공고 찾아줘"

처리 과정:
1. 사용자 프로필 조회
2. 전문과목, 경력, 선호사항 분석
3. 맞춤형 검색 쿼리 생성
4. 개인화된 공고 추천
```

### D. 인기 공고 분석
```
입력: "나와 비슷한 사람들이 많이 지원한 공고 찾아줘"

처리 과정:
1. 유사한 배경의 사용자 검색
2. 지원 이력 데이터 수집
3. 인기 공고 식별
4. 추천 이유와 함께 제공
```

## 🧪 테스트 시나리오

### 기본 검색 테스트
```bash
python agent.py --prompt "서울 내과 공고 찾아줘"
python agent.py --prompt "연봉 5000만원 이상 정형외과 공고 있어?"
python agent.py --prompt "부산 지역 응급의학과 계약직 공고 보여줘"
python agent.py --prompt "경력 3년 이하 신경외과 공고 추천해줘"
```

### 의미론 검색 테스트
```bash
python agent.py --prompt "초음파 경험이 있는 의사가 지원할 수 있는 공고 찾아줘"
python agent.py --prompt "대장내시경 시술 가능한 곳에서 일하고 싶어"
python agent.py --prompt "로봇수술 경험을 쌓을 수 있는 병원 공고 있어?"
python agent.py --prompt "야간근무가 적은 공고 추천해줘"
```

### 컨텍스트 기반 테스트
```bash
python agent.py --url "https://s-new.medigate.net/recruit/1172036" --prompt "이 공고보다 연봉이 높은 비슷한 공고 찾아줘"
python agent.py --uid "medizzang" --prompt "내 경력에 맞는 공고 중에서 서울 지역만 보여줘"
```

## 📊 데이터베이스 구조

### 공고 인덱스: `recruit_text-embedding-3-small_1536_100000_300_20250529_150924`
```json
{
  "metadata": {
    "BOARD_IDX": "1172036",
    "ORGANIZATION_NAME": "병원명",
    "SPECIALTIES": "진료과",
    "REGION_NAME": "지역",
    "PAY_DETAILS": "급여 정보",
    "REGULAR_STATUS": "고용형태",
    "WORK_HOUR_DETAILS": "근무시간",
    "ADDRESS": "주소"
  },
  "text": "공고 전체 텍스트",
  "vector_field": [0.1, 0.2, ...]
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
  "vector_field": [0.1, 0.2, ...]
}
```

## 🔄 프로그래밍 인터페이스

명령행 외에도 Python 코드에서 직접 사용할 수 있습니다:

```python
from agent import process_medical_recruit_request

# 비동기 함수 호출
result = await process_medical_recruit_request(
    url="https://s-new.medigate.net/recruit/1172036",
    uid="medizzang",
    prompt="서울지역 피부과 공고 추천해줘"
)

# 결과 확인
if result["success"]:
    print(result["data"])
else:
    print(f"오류: {result['message']}")
```

## ⚠️ 주의사항 및 제한사항

### 1. 필수 요구사항
- **OpenAI API 키**: GPT-4 사용을 위해 필수
- **OpenSearch 연결**: 검색 데이터베이스 접근 필요
- **Python 3.8+**: 비동기 처리 및 타입 힌트 지원

### 2. URL 형식 제한
- 지원 형식: `https://s-new.medigate.net/recruit/[숫자]`
- board_id는 숫자만 지원
- 잘못된 URL 형식은 일반 검색으로 대체 처리

### 3. 네트워크 의존성
- OpenSearch 서버 연결 필요
- OpenAI API 호출 필요
- 네트워크 오류 시 적절한 오류 메시지 제공

### 4. 성능 고려사항
- 벡터 검색으로 인한 지연 시간 (보통 2-5초)
- 대용량 검색 시 메모리 사용량 증가
- API 호출 제한 (OpenAI rate limit)

## 🔧 고급 설정

### 로깅 레벨 조정
```bash
# 환경 변수로 로깅 레벨 설정
export LOG_LEVEL=DEBUG
python agent.py --prompt "테스트 검색"
```

### 검색 결과 개수 조정
`server.py`에서 검색 결과 개수를 조정할 수 있습니다:
```python
query = {
    "query": {...},
    "size": 50  # 기본값: 20
}
```

### 임베딩 모델 변경
OpenAI 임베딩 모델을 변경할 수 있습니다:
```python
self.embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 더 높은 성능
    request_timeout=30
)
```

## 🆕 확장 가능한 기능

### 1. 추가 필터링 옵션
- 병원 규모별 검색
- 병상 수 기준 필터링
- 특수 장비 보유 여부

### 2. 고급 추천 알고리즘
- 협업 필터링 기반 추천
- 머신러닝 기반 선호도 예측
- 실시간 트렌드 분석

### 3. 알림 및 모니터링
- 맞춤 공고 자동 알림
- 지원 현황 추적
- 채용 시장 트렌드 분석

### 4. 다국어 지원
- 영어 공고 검색
- 다국어 임베딩 모델 적용

## 🐛 문제 해결

### 일반적인 오류 및 해결 방법

#### 1. OpenAI API 키 오류
```
ValueError: OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.
```
**해결 방법**: `.env` 파일에 올바른 API 키 설정

#### 2. OpenSearch 연결 오류
```
opensearchpy.exceptions.ConnectionError
```
**해결 방법**: 
- 네트워크 연결 확인
- OpenSearch 서버 상태 확인
- 인증 정보 재확인

#### 3. 검색 결과 없음
```
포맷팅할 결과가 없습니다.
```
**해결 방법**: 
- 검색 조건 완화
- 다른 키워드로 재검색
- 지역이나 진료과 조건 변경

#### 4. URL 파싱 실패
```
URL에서 board_id를 찾을 수 없습니다.
```
**해결 방법**: 
- URL 형식 확인 (`/recruit/숫자` 형태)
- 올바른 메디게이트 공고 URL 사용

## �� 지원 및 문의

### 개발팀 연락처
- **이슈 리포팅**: GitHub Issues
- **기능 제안**: 개발팀 이메일
- **긴급 문의**: 시스템 관리자

### 로그 및 디버깅
문제 발생 시 다음 정보를 함께 제공해주세요:
1. 사용한 정확한 명령어
2. 오류 메시지 전문
3. 환경 정보 (Python 버전, OS)
4. 로그 파일 (`--verbose` 옵션 사용 시)

## 📈 업데이트 이력

### v2.0 (2025.06.01)
- 명령행 인터페이스 추가
- URL 자동 파싱 기능
- 사용자 프로필 기반 맞춤 추천
- 대화형 모드 제거

### v1.0 (2025.05.29)
- 기본 MCP 시스템 구축
- OpenSearch 연동
- 벡터 검색 기능
