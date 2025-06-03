```markdown
# 의료진 채용 검색 MCP 시스템 (v2)

의료진 채용 공고를 지능적으로 검색하고 **맞춤형 추천**을 제공하며, **특정 공고 요약** 및 **인기 공고 추천** 기능까지 갖춘 Model Context Protocol (MCP) 기반 AI 에이전트 시스템입니다.

## 📋 개요

이 시스템은 의료진들이 자신에게 맞는 채용 공고를 쉽고 빠르게 찾을 수 있도록 돕는 AI 어시스턴트입니다. 사용자의 자연어 질문을 이해하고, OpenSearch를 통해 관련 공고를 검색하며, LLM의 분석을 통해 개인화된 맞춤 추천, 공고 요약, 인기 공고 정보 등을 제공합니다.

## 🚀 주요 특징

### ✨ **향상된 지능형 추천 및 정보 제공**
- **LLM 심층 분석**: AI가 검색 결과를 직접 분석하여 사용자의 요구에 가장 적합한 공고 선별 (최대 5개).
- **다양한 정보 제공**: 일반 검색, 맞춤 추천, 유사 공고, 특정 공고 요약, 인기 공고 추천 등 다각적인 지원.
- **사용자 친화적 결과**: 최종 결과는 자연스러운 문장으로 가공되어 제공.
- **파일 저장**: 맞춤 추천 시 선별된 공고 ID 목록은 서버에 JSON 파일로 자동 저장.

### �� **강력한 검색 기능**
- **의미적 검색**: "초음파 경험 많은 곳", "야간 당직 없는 내과" 등 자연어 키워드 기반 검색.
- **다중 필터**: 지역, 진료과, 병원명 등 복합 조건으로 정교한 검색 가능.
- **실시간 분석 지원**: 서버 로그를 통해 토큰 수, 응답 크기 등 모니터링 가능.

## 🏗️ 시스템 구조

```
├── server.py          # MCP 서버 (도구 정의, OpenSearch 연동, 핵심 로직)
├── agent.py           # MCP 에이전트 (LLM + 도구 조합, 사용자 요청 처리)
├── .env              # 환경 변수 설정 파일
└── README.md         # 시스템 사용 가이드
```

## 🔧 필요 조건

### Python 패키지
```bash
pip install langchain-openai langchain-mcp-adapters langgraph opensearch-py tiktoken python-dotenv httpx openai
```
(정확한 버전은 `requirements.txt` 파일 생성 후 관리하는 것을 권장합니다.)

필요한 주요 패키지:
- `langchain-openai` (OpenAI 모델 연동)
- `langchain-mcp-adapters` (MCP 클라이언트)
- `langgraph` (ReAct 에이전트 구성)
- `opensearch-py` (OpenSearch 연동)
- `tiktoken` (토큰 계산용)
- `python-dotenv` (환경 변수 관리)
- `httpx`, `openai` (OpenAI API 직접 호출 시 필요할 수 있음, LangChain이 내부적으로 사용)

### 환경 변수 설정

`.env` 파일에 다음 내용을 설정하세요:

```bash
# OpenAI API
OPENAI_API_KEY="your_openai_api_key_here"

# OpenSearch 설정 (예시, 실제 환경에 맞게 수정)
OPENSEARCH_HOST=opensearch.medigate.net
OPENSEARCH_PORT=9200
OPENSEARCH_USER=medigate
OPENSEARCH_PASSWORD=your_opensearch_password_here
```

## 🛠️ 제공 기능 (도구) - `server.py`

### 1. 개별 조회 도구
- **`get_board_by_id(board_id: str)`**: 특정 공고 ID로 상세 정보 조회.
- **`get_user_by_id(user_id: str)`**: 특정 사용자 ID로 프로필 정보 조회.

### 2. 검색 도구
- **`create_and_search_recruits(...)`**: 다양한 조건으로 채용공고 검색.
  - 지역, 진료과, 병원명, 의미적 키워드, 제외할 공고 ID, 결과 수 지정 가능.
  - 결과로 공고 요약 정보 목록과 `board_ids` 반환.
- **`create_and_search_users(...)`**: 의료진 프로필 검색.

### 3. 추천, 요약, 인기 공고 도구 ⭐ **NEW & ENHANCED!**
- **`create_formatted_recommendations(search_criteria: str, selected_board_ids: List[str])`**:
  - LLM이 `create_and_search_recruits` 결과에서 **직접 선별한 최대 5개 공고 ID**와 원본 검색 조건을 받아, 상세 정보가 포함된 사용자 친화적 추천 메시지 생성.
  - 선별된 공고 ID 목록은 서버에 JSON 파일로 저장.
- **`summarize_board_by_id(board_id: str)`**:
  - 주어진 공고 ID에 해당하는 공고의 주요 정보를 간결하게 요약하여 반환.
- **`recommend_popular_jobs(user_id: Optional[str] = None, size: int = 5)`**:
  - OpenSearch의 가상 `view_count` 필드를 기준으로 인기 있는 채용공고 추천.
  - `user_id` 제공 시, 해당 사용자의 전문과를 필터 조건으로 우선 고려.

## 🎯 주요 사용 시나리오 (`agent.py`의 시스템 프롬프트 기반)

### A. 🔍 일반 검색
```
"서울 강남구 피부과 공고 2개만 찾아줘"

처리 과정 (예상):
1. LLM이 `create_and_search_recruits(region="서울 강남구", department="피부과", size=2)` 호출 결정.
2. `server.py`에서 OpenSearch 검색 후 결과 (공고 요약 목록, board_ids) 반환.
3. LLM이 결과를 바탕으로 사용자에게 자연스러운 문장으로 답변 생성.
   (예: "서울 강남구 피부과 공고 2개를 찾았습니다: ...")
```

### B. ⭐ 맞춤 추천 (핵심 기능)
```
"부산 내과 전문의 중 연봉 좋은 곳 3개 추천해줘" (사용자 ID: test_user)

처리 과정 (예상):
1. LLM이 `get_user_by_id(user_id="test_user")` 호출하여 사용자 정보 (예: 전문과 '내과') 확인.
2. LLM이 `create_and_search_recruits(region="부산", department="내과", semantic_keywords="연봉 좋은 곳", size=20)` 호출하여 후보군 확보.
3. **LLM이 직접** 반환된 20개 공고의 요약 정보(특히 `pay_details`)를 분석하여 **최적의 공고 3개 선별**.
4. LLM이 선별된 `board_ids`와 `search_criteria`="부산 내과 연봉 좋은 곳 3개 추천"을 `create_formatted_recommendations` 도구에 전달.
5. `server.py`에서 해당 공고들의 상세 정보를 포함한 추천 메시지 생성 후 반환.
6. LLM이 최종 추천 메시지를 사용자에게 전달.
```

### C. 📄 특정 공고 요약
```
"지금 보고 있는 공고(ID: 12345) 내용 좀 간단히 알려줘" (URL 컨텍스트에서 board_id="12345" 감지)

처리 과정 (예상):
1. LLM이 컨텍스트의 `board_id`를 인지.
2. LLM이 `summarize_board_by_id(board_id="12345")` 호출 결정.
3. `server.py`에서 해당 공고 요약 정보 생성 후 반환.
4. LLM이 요약 정보를 사용자에게 전달.
```

### D. 🔥 인기 공고 추천
```
"내 전문분야에서 요즘 인기 있는 공고 3개만 추천해줄래?" (사용자 ID: test_user_ent)

처리 과정 (예상):
1. LLM이 `recommend_popular_jobs(user_id="test_user_ent", size=3)` 호출 결정.
2. `server.py`에서 `get_user_by_id`로 "test_user_ent"의 전문과(예: 이비인후과) 확인.
3. 해당 전문과를 필터로 OpenSearch에서 `view_count` 높은 순으로 공고 검색 후 포맷팅하여 반환.
4. LLM이 인기 공고 목록을 사용자에게 전달.
```

## 🚀 사용 방법

### 1. CLI (명령줄 인터페이스) 사용

터미널에서 `agent.py`를 직접 실행하여 에이전트와 상호작용할 수 있습니다.

```bash
# 기본 사용법
python agent.py --prompt "사용자 질문"

# 옵션: URL 컨텍스트 제공 (공고 상세 페이지 보고 있을 때)
python agent.py --url "/recruit/12345" --prompt "이 공고 요약해줘"

# 옵션: 사용자 ID 제공 (개인화된 응답 유도)
python agent.py --uid "testuser001" --prompt "나에게 맞는 공고 추천해줘"

# 모든 옵션 사용
python agent.py --url "/recruit/12345" --uid "testuser001" --prompt "이 공고와 비슷한 다른 내과 공고 찾아줘"
```
**주의:** `--url`의 `12345` 부분이나 `--uid`의 `testuser001`은 실제 테스트 가능한 ID로 변경해야 합니다.

### 2. API 형태로 사용 (애플리케이션 연동)

`agent.py`의 `process_medical_recruit_request` 함수를 사용하여 다른 Python 애플리케이션에서 에이전트 기능을 호출할 수 있습니다.

```python
import asyncio
from agent import process_medical_recruit_request

async def run_agent_api_examples():
    # 예시 1: 맞춤 추천 (사용자 ID 활용)
    result1 = await process_medical_recruit_request(
        uid="ctoman", # 실제 사용자 ID
        prompt="내 전문과(내과)에 맞는 서울 지역 공고 3개 추천해줘, 최근 등록된 순으로."
    )
    print("API 결과 1 (맞춤 추천):", result1)

    # 예시 2: 특정 공고 기반 유사 공고 검색 (URL 컨텍스트 활용)
    result2 = await process_medical_recruit_request(
        url="https://example.com/recruit/1172559", # 실제 공고 URL 또는 board_id를 포함한 경로
        prompt="이 공고와 비슷한 조건이면서 경기도 지역 공고를 찾아줘."
    )
    print("API 결과 2 (유사 공고):", result2)

    # 예시 3: 인기 공고 조회
    result3 = await process_medical_recruit_request(
        prompt="요즘 가장 인기 있는 영상의학과 공고 3개 보여줘."
    )
    print("API 결과 3 (인기 공고):", result3)

if __name__ == "__main__":
    asyncio.run(run_agent_api_examples())
```

## 📊 LLM 기반 추천 및 정보 제공 원리

1.  **사용자 요청 및 컨텍스트 분석**: `agent.py`는 사용자의 프롬프트, URL, 사용자 ID를 종합적으로 분석하여 시스템 프롬프트에 컨텍스트와 처리 힌트를 추가합니다.
2.  **LLM의 도구 선택 및 호출**: 시스템 프롬프트와 사용자 질문을 받은 LLM (gpt-4o-mini)은 ReAct 로직에 따라 필요한 도구를 순차적으로 선택하고 호출합니다.
3.  **서버의 도구 실행**: `server.py`는 요청받은 도구(예: `get_user_by_id`, `create_and_search_recruits`)를 실행합니다. 이 과정에서 OpenSearch와 연동하여 데이터를 조회하거나 가공합니다.
4.  **LLM의 추가 분석 및 판단 (필요시)**:
    *   **맞춤 추천 시**: `create_and_search_recruits`로 얻은 다수의 후보 공고들을 LLM이 직접 검토하여 사용자의 세부 요구(예: "연봉 높은 순", "워라밸 좋은 곳")에 맞춰 최대 5개를 선별합니다.
    *   선별된 ID는 `create_formatted_recommendations` 도구로 전달되어 최종 추천 메시지가 생성됩니다.
5.  **최종 답변 생성**: LLM은 모든 도구 실행 결과를 종합하고, 시스템 프롬프트의 지침에 따라 사용자 친화적인 최종 답변을 생성하여 반환합니다.

## 🔍 OpenSearch 인덱스 구조 (예시)

(이전 README와 동일, 필요시 `view_count` 필드 명시)

### 공고 인덱스: `recruit_text-embedding-3-small_...`
```json
{
  "metadata": {
    "BOARD_IDX": "공고ID",
    "ORGANIZATION_NAME": "병원명",
    "SPECIALTIES": "진료과",
    "REGION_NAME": "지역",
    // ... 기타 필드 ...
    "view_count": 120 // 인기 공고 추천에 사용될 수 있는 필드 (숫자 타입)
  },
  "text": "공고 상세 내용",
  "vector_field": [0.1, 0.2, ...]
}
```
(사용자 인덱스 구조는 이전과 동일)

## ⚠️ 주의사항

1.  **API 키 및 인증 정보**: `.env` 파일에 올바른 OpenAI API 키와 OpenSearch 접속 정보를 설정해야 합니다. 실제 운영 환경에서는 환경 변수 관리에 더욱 주의해야 합니다.
2.  **OpenSearch 인덱스**: `server.py`에 정의된 인덱스 이름 (`recruit_index`, `resume_index`)이 실제 OpenSearch 환경의 인덱스 이름과 일치해야 합니다. `view_count`와 같은 특정 필드는 인덱스에 존재하고 올바른 타입으로 매핑되어 있어야 합니다.
3.  **Python 환경**: 필요한 모든 패키지가 설치된 가상 환경에서 실행하는 것을 권장합니다.
4.  **토큰 사용량**: LLM 호출, 특히 복잡한 분석이나 여러 단계의 도구 사용은 토큰 사용량을 증가시킬 수 있습니다. 비용 관리에 유의하세요.
5.  **에러 로그**: 실행 중 문제 발생 시 `agent.py` 및 `server.py`의 로그(표준 출력 또는 파일)를 확인하여 원인을 파악하세요.

## 🔮 향후 확장 계획

(이전 README와 유사하나, 현재 기능셋에 맞춰 일부 조정 가능)

### 🎯 고도화 기능
1.  **사용자 피드백 반영**: 추천 결과에 대한 사용자 피드백을 수집하여 추천 모델 개선.
2.  **상세 조건 분석 강화**: "워라밸", "성장 가능성" 등 추상적 조건에 대한 LLM의 이해도 및 분석 능력 향상.
3.  **알림 기능**: 새로운 맞춤 공고 발생 시 사용자에게 알림.

### 🔧 기술적 개선
1.  **LangGraph 최적화**: 콜백 로깅 및 그래프 실행 효율성 개선.
2.  **비동기 처리 강화**: MCP 서버 및 에이전트의 비동기 처리 로직 최적화.
3.  **테스트 자동화**: 주요 시나리오에 대한 통합 테스트 및 단위 테스트 구축.

## 🏆 시스템 장점

(이전 README와 유사)

### 💡 **MCP 아키텍처의 이점**
- **모듈화 및 유연성**: 도구 기반으로 기능을 분리하여 개발 및 유지보수가 용이.
- **확장성**: 새로운 검색 조건, 추천 로직, 정보 제공 방식을 도구로 쉽게 추가 가능.

### 🎯 **LLM 기반 ReAct 에이전트의 강점**
- **동적 추론**: 정해진 규칙뿐 아니라, LLM의 추론 능력을 통해 복잡하고 다양한 사용자 요청에 유연하게 대응.
- **자연스러운 상호작용**: 사용자의 자연어를 이해하고, 사람과 대화하듯 결과 제공.
- **설명 가능한 결정**: (로그 분석을 통해) 어떤 근거로 특정 도구를 사용하고 결정을 내렸는지 추적 가능.

### ⚡ **정보 중심 설계**
- **컨텍스트 활용**: URL, 사용자 ID 등의 컨텍스트 정보를 활용하여 더 정확하고 개인화된 응답 제공.
- **체계적인 정보 처리**: 검색, 분석, 요약, 추천 등 다양한 정보 처리 단계를 거쳐 신뢰도 높은 결과 도출.

## 📞 문의 및 지원

(이전 README와 동일)

---

**🎉 의료진 채용의 새로운 패러다임을 경험해보세요!**

이 시스템은 단순한 키워드 검색을 넘어서, **AI가 사용자의 의도를 파악하고, 관련 정보를 수집 및 분석하여, 맞춤형 정보를 제공하는** 지능형 채용 어시스턴트입니다.
```

**주요 변경 사항:**

*   **개요 및 주요 특징 업데이트**: 새로운 도구(`summarize_board_by_id`, `recommend_popular_jobs`)와 변경된 `create_formatted_recommendations`의 역할을 반영하여 시스템의 기능을 더 명확히 설명했습니다.
*   **필요 조건**: 패키지 목록을 간소화하고, `requirements.txt` 사용을 권장했습니다.
*   **제공 기능 (도구)**: `server.py` 기준으로 7개 도구를 명확히 설명하고, 특히 새롭거나 향상된 도구에 ⭐ 표시를 했습니다.
*   **주요 사용 시나리오**: `agent.py`의 시스템 프롬프트에 기술된 시나리오와 도구 사용 흐름을 반영하여 업데이트했습니다.
*   **사용 방법**:
    *   CLI 실행 방법을 `argparse` 인자에 맞춰 상세히 안내했습니다.
    *   API 사용 예시를 현재 `process_medical_recruit_request` 함수의 인자(url, uid, prompt)에 맞게 수정했습니다.
*   **LLM 기반 추천 및 정보 제공 원리**: 전체적인 시스템의 동작 방식을 단계별로 설명했습니다.
*   **OpenSearch 인덱스 구조**: `view_count` 필드 예시를 추가했습니다.
*   **주의사항**: CLI 실행 및 API 키/인증 정보 관련 내용을 보강했습니다.
*   **시스템 장점**: 현재 아키텍처(MCP, ReAct 에이전트)의 장점을 강조했습니다.
