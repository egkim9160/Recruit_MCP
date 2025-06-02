#!/usr/bin/env python3

import os
import json
import asyncio
import logging 
import argparse
import re
import uuid
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatGenerationChunk
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


LANGGRAPH_REACT_CORE_NODES = ["agent", "action", "tools"] 
MAX_LOG_LENGTH_SHORT = 500
MAX_LOG_LENGTH_LONG = 2000


class AgentToolLogger(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tool_info: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.current_llm_thoughts: Dict[uuid.UUID, str] = {}
        self.active_node_names: Dict[uuid.UUID, str] = {} 

    def _truncate_log(self, text: Any, max_len: int = MAX_LOG_LENGTH_SHORT) -> str:
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False, indent=2) 
            except TypeError:
                text = str(text)
            except Exception:
                text = str(text)
        return text[:max_len] + '...' if len(text) > max_len else text

    def _get_relevant_node_name(self, serialized: Dict[str, Any]) -> Optional[str]:
        if not serialized or not isinstance(serialized, dict): return None
        name_from_key = serialized.get("name")
        if name_from_key and name_from_key in LANGGRAPH_REACT_CORE_NODES: return name_from_key
        s_id = serialized.get("id")
        if isinstance(s_id, list) and s_id:
            last_id_part = s_id[-1]
            if isinstance(last_id_part, str) and last_id_part in LANGGRAPH_REACT_CORE_NODES: return last_id_part
        return None

    def _log_messages_preview(self, messages: List[BaseMessage], direction: str, node_name: str, run_id: uuid.UUID):
        if not messages: return
        last_message = messages[-1]
        log_parts = [f"    {direction} node [{node_name}] (run_id: {run_id}):"]
        content_preview = ""
        if hasattr(last_message, "content") and last_message.content is not None:
            content_preview = self._truncate_log(last_message.content, MAX_LOG_LENGTH_LONG)

        if isinstance(last_message, HumanMessage): log_parts.append(f"Human: {content_preview}")
        elif isinstance(last_message, AIMessage):
            log_parts.append(f"AI: {content_preview if content_preview else '(No content)'}")
            if last_message.tool_calls:
                tool_calls_summary = [
                    f"Tool Call: {tc.get('name', 'N/A')}({self._truncate_log(json.dumps(tc.get('args', {}), ensure_ascii=False), MAX_LOG_LENGTH_SHORT)}) ID: {tc.get('id')}"
                    for tc in last_message.tool_calls
                ]
                log_parts.append(f"  ↳ {'; '.join(tool_calls_summary)}")
        elif isinstance(last_message, ToolMessage): log_parts.append(f"Tool Result (for {last_message.tool_call_id}): {content_preview}")
        else: log_parts.append(f"{type(last_message).__name__}: {self._truncate_log(str(last_message), MAX_LOG_LENGTH_LONG)}")
        logger.info("\n".join(log_parts))

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        node_name = self._get_relevant_node_name(serialized)
        if not node_name: return
        self.active_node_names[run_id] = node_name
        logger.info(f"➡️ GRAPH NODE START: [{node_name}] (run_id: {run_id})")
        if inputs and "messages" in inputs and isinstance(inputs["messages"], list): self._log_messages_preview(inputs["messages"], "Input to", node_name, run_id)
        elif inputs: logger.info(f"    Input to node [{node_name}] (run_id: {run_id}):\n{self._truncate_log(inputs, MAX_LOG_LENGTH_LONG)}")

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        node_name = self.active_node_names.pop(run_id, None)
        if not node_name: return
        logger.info(f"⬅️ GRAPH NODE END: [{node_name}] (run_id: {run_id})")
        if outputs and "messages" in outputs and isinstance(outputs["messages"], list): self._log_messages_preview(outputs["messages"], "Output from", node_name, run_id)
        elif outputs: logger.info(f"    Output from node [{node_name}] (run_id: {run_id}):\n{self._truncate_log(outputs, MAX_LOG_LENGTH_LONG)}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: uuid.UUID, **kwargs) -> None:
        logger.info(f"🤖 LLM CALL START (run_id: {run_id})")
        self.current_llm_thoughts[run_id] = ""
        if prompts: logger.info(f"    LLM Prompts (run_id: {run_id}):\n{self._truncate_log(prompts[0] if prompts else '', MAX_LOG_LENGTH_LONG * 2)}")

    def on_llm_new_token(self, token: str, *, chunk: Optional[BaseMessage | ChatGenerationChunk] = None, run_id: uuid.UUID, **kwargs: Any,) -> None:
        if chunk and hasattr(chunk, 'content'): self.current_llm_thoughts[run_id] += chunk.content

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs) -> None:
        logger.info(f"✅ LLM CALL END (run_id: {run_id})")
        message_content_logged = False
        if response.generations and response.generations[0] and hasattr(response.generations[0][0], 'message'):
            message = response.generations[0][0].message
            llm_content = str(message.content).strip() if message.content else ""
            tool_calls = message.tool_calls
            if tool_calls:
                if llm_content: 
                    logger.info(f"🤔 LLM THOUGHT (for Action) (run_id: {run_id}):\n{self._truncate_log(llm_content, MAX_LOG_LENGTH_LONG)}")
                    message_content_logged = True
                actions_log_parts = [f"  Tool: {tc.get('name')}, Args: {self._truncate_log(json.dumps(tc.get('args', {}), ensure_ascii=False), MAX_LOG_LENGTH_SHORT)}, ID: {tc.get('id')}" for tc in tool_calls]
                logger.info(f"🛠️ LLM ACTION (Tool Calls) (run_id: {run_id}):\n" + "\n".join(actions_log_parts))
            # else: (Final Answer는 on_agent_finish에서 처리)
            #    if llm_content:
            #        logger.info(f"💬 LLM POTENTIAL FINAL ANSWER (from message.content) (run_id: {run_id}):\n{self._truncate_log(llm_content, MAX_LOG_LENGTH_LONG)}")
            #        message_content_logged = True
        
        accumulated_thought_text = self.current_llm_thoughts.pop(run_id, "").strip()
        if accumulated_thought_text and not message_content_logged: 
            # message.content로 이미 로깅되지 않았고, accumulated_thought_text에 내용이 있는 경우
            # 이것은 보통 LLM의 raw output (Thought + Action 문자열 또는 Thought + Final Answer 문자열)
            logger.info(f"�� LLM Raw Output (from streaming) (run_id: {run_id}):\n{self._truncate_log(accumulated_thought_text, MAX_LOG_LENGTH_LONG)}")


    def on_agent_action(self, action: AgentAction, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        thought = action.log.split("Action:")[0].replace("Thought:", "").strip()
        if thought: logger.info(f"🤔 AGENT THOUGHT (parsed from action.log) (run_id: {run_id}, parent: {parent_run_id}):\n{self._truncate_log(thought, MAX_LOG_LENGTH_LONG)}")
        action_input_str = json.dumps(action.tool_input, ensure_ascii=False)
        logger.info(f"🛠️ AGENT ACTION DECISION (run_id: {run_id}, parent: {parent_run_id}): Tool: `{action.tool}`, Input: `{self._truncate_log(action_input_str, MAX_LOG_LENGTH_SHORT)}`")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self.tool_info[run_id] = {"name": tool_name}
        logger.info(f"🔧 TOOL START [{tool_name}] (run_id: {run_id}, parent: {parent_run_id})")
        logger.info(f"    Input: {self._truncate_log(input_str, MAX_LOG_LENGTH_LONG)}")

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        tool_name = self.tool_info.pop(run_id, {"name": "unknown_tool"})["name"]
        logger.info(f"✅ TOOL END [{tool_name}] (run_id: {run_id}, parent: {parent_run_id})")
        output_str = output.content if hasattr(output, 'content') and isinstance(output, ToolMessage) else str(output)
        logger.info(f"    Output: {self._truncate_log(output_str, MAX_LOG_LENGTH_LONG)}")

    def on_tool_error(self, error: BaseException, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        tool_name = self.tool_info.pop(run_id, {"name": "unknown_tool"})["name"]
        logger.error(f"❌ TOOL ERROR [{tool_name}] (run_id: {run_id}, parent: {parent_run_id}): {error}", exc_info=True)

    def on_agent_finish(self, finish: AgentFinish, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        thought = finish.log.split("Final Answer:")[0].replace("Thought:", "").strip()
        final_answer = finish.return_values.get('output', "N/A (No 'output' in return_values)")
        if thought: logger.info(f"🤔 AGENT FINAL THOUGHT (parsed from finish.log) (run_id: {run_id}, parent: {parent_run_id}):\n{self._truncate_log(thought, MAX_LOG_LENGTH_LONG)}")
        logger.info(f"🏁 AGENT FINISH (Callback) (run_id: {run_id}, parent: {parent_run_id})\nFinal Answer:\n{self._truncate_log(str(final_answer), MAX_LOG_LENGTH_LONG)}")

# --- Langchain Agent and Application Logic ---
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler # 이미 위에서 import 했지만, 명확성을 위해 남겨둘 수 있음
from langchain_core.outputs import LLMResult, ChatGenerationChunk # 위와 동일
from langchain_core.agents import AgentAction, AgentFinish # 위와 동일
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage # 위와 동일


class URLParser:
    @staticmethod
    def parse_recruit_url(url: str) -> Optional[str]:
        try:
            match = re.search(r'/recruit/(\d+)', url)
            if match: return match.group(1)
            return None
        except Exception as e:
            logger.error(f"URL 파싱 오류: {e}")
            return None
    
    @staticmethod
    def is_recruit_detail_request(url: str) -> bool:
        return '/recruit/' in url and bool(re.search(r'/recruit/\d+', url))

class MedicalRecruitAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key: raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")        
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if "LANGSMITH_TRACING" in os.environ: del os.environ["LANGSMITH_TRACING"]
        if "LANGCHAIN_API_KEY" in os.environ: del os.environ["LANGCHAIN_API_KEY"]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=self.openai_api_key, streaming=True)
        self.client = None
        self.agent_executor = None
        
    async def initialize(self):
        try:
            server_path = os.path.abspath("server.py") # server.py는 agent.py와 같은 디렉토리에 있다고 가정
            self.client = MultiServerMCPClient({"medical_recruit": {"command": "python", "args": [server_path], "transport": "stdio"}})
            tools = await self.client.get_tools()
            if not tools: raise RuntimeError("MCP 클라이언트에서 도구를 로드하지 못했습니다.")
            tool_names = [tool.name for tool in tools]
            logger.info(f"로드된 도구 목록: {tool_names}")
            print(f"{len(tools)}개의 도구가 로드되었습니다: {', '.join(tool_names)}")
            self.agent_executor = create_react_agent(self.llm, tools)
            print("에이전트 실행기가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"초기화 오류: {e}", exc_info=True)
            print(f"초기화 오류: {e}")
            raise

    async def process_request(self, url: Optional[str], uid: Optional[str], prompt: str) -> str:
        if not self.agent_executor: raise ValueError("에이전트가 초기화되지 않았습니다.")
        try:
            board_id = None; is_detail_request = False
            if url and URLParser.is_recruit_detail_request(url):
                is_detail_request = True
                board_id = URLParser.parse_recruit_url(url)
                if board_id: logger.info(f"채용 공고 상세 요청 감지 - board_id: {board_id}")
            
            context_info = []
            if board_id: context_info.append(f"현재 보고 있는 공고 ID: {board_id}")
            if uid: context_info.append(f"사용자 ID: {uid}")
            if is_detail_request: context_info.append("요청 유형: 채용 공고 상세 조회 관련")
            
            system_prompt = """
당신은 의료진 채용 검색 전문 AI 어시스턴트입니다. 사용자의 요청을 정확히 분석하고 적절한 도구를 순서대로 사용하여 최상의 결과를 제공합니다.

**사용 가능한 도구 (5개):**
1. `get_board_by_id(board_id)`
2. `get_user_by_id(user_id)`
3. `create_and_search_recruits(region, department, employment_type, hospital_name, semantic_keywords, exclude_board_id, size)`
4. `create_and_search_users(department, preferred_region, semantic_keywords, size)`
5. `format_search_results(results_json, format_type)`

**작업 흐름 가이드:**
1. **요청 분석**
2. **기본 정보 수집**
3. **검색 실행**
4. **결과 포맷팅**

**처리 전략:**
1. 현재 공고 관련 질문 (board_id 존재 시): `get_board_by_id` -> `create_and_search_recruits` (exclude_board_id 사용) -> `format_search_results`.
2. 사용자 맞춤 추천 (user_id 존재 시): `get_user_by_id` -> `create_and_search_recruits` -> `format_search_results`.
3. 일반 공고 검색: `create_and_search_recruits` -> `format_search_results`.
4. 사용자 검색: `create_and_search_users` -> `format_search_results`.

**중요 규칙:**
- 논리적 도구 호출 순서.
- `format_search_results`로 최종 결과 포맷팅.
- 결과 부재/오류 시 명확한 안내.
- 연봉 검색 미지원.
- 필터는 wildcard 부분 일치 지원.

**응답 형식:**
- 간결하고 유용한 정보 위주.
- 검색 결과는 기본 'summary' 포맷, 요청 시 'detailed' 가능.
"""
            if context_info: system_prompt += f"\n\n**현재 컨텍스트:**\n" + "\n".join(context_info)
            if board_id and any(keyword in prompt.lower() for keyword in ["비슷한", "유사한", "같은", "더 나은", "비교"]): system_prompt += f"\n\n**처리 힌트:** 현재 공고(ID: {board_id}) 관련 비교/추천 요청. 해당 공고 조회 후 `exclude_board_id={board_id}` 설정하여 검색."
            if uid and any(keyword in prompt.lower() for keyword in ["내", "나", "맞춤", "추천", "개인"]): system_prompt += f"\n\n**처리 힌트:** 사용자(ID: {uid}) 맞춤형 추천 요청. 사용자 정보 조회 후 개인화 검색."
            
            input_data = { "messages": [ {"role": "system", "content": system_prompt}, {"role": "user", "content": prompt} ] }
            logger.info(f"사용자 요청 처리 시작: {prompt}")
            print(f"사용자 요청 처리 시작: {prompt}") 
            callback_handler = AgentToolLogger()
            print("에이전트 실행 중...") 
            response_data = await self.agent_executor.ainvoke(input_data, config={ "callbacks": [callback_handler], "recursion_limit": 15 })
            logger.info("에이전트 실행 완료")

            final_answer_content = "응답을 생성하거나 추출할 수 없었습니다."
            if response_data and "messages" in response_data and isinstance(response_data["messages"], list):
                for message in reversed(response_data["messages"]):
                    if isinstance(message, AIMessage) and not message.tool_calls and message.content:
                        final_answer_content = str(message.content); break
                else: 
                    if response_data["messages"]:
                        last_msg = response_data["messages"][-1]
                        if hasattr(last_msg, 'content') and last_msg.content: final_answer_content = str(last_msg.content); logger.warning(f"표준 최종 AI 응답 못찾아 마지막 메시지 content 사용.")
                    else: logger.warning(f"messages 리스트 비었거나 최종 AI 응답 못찾음.")
            elif response_data and "output" in response_data: final_answer_content = str(response_data["output"])
            logger.info(f"최종 응답 문자열 생성 완료 (길이: {len(final_answer_content)}).")
            return final_answer_content
        except Exception as e:
            error_msg = f"요청 처리 중 심각한 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"{error_msg}") 
            return error_msg

    async def cleanup(self):
        if self.client:
            try:
                if hasattr(self.client, 'close') and asyncio.iscoroutinefunction(self.client.close): await self.client.close()
                elif hasattr(self.client, 'close'): self.client.close()
                logger.info("MCP 클라이언트 정리 완료")
            except Exception as e: logger.error(f"클라이언트 정리 오류: {e}", exc_info=True); print(f"클라이언트 정리 오류: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='의료진 채용 검색 AI 에이전트')
    parser.add_argument('--url', type=str, help='요청 URL')
    parser.add_argument('--uid', type=str, help='사용자 ID')
    parser.add_argument('--prompt', type=str, required=True, help='사용자 질문')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    main_logger = logging.getLogger(f"{__name__}.main")
    agent = MedicalRecruitAgent()
    exit_code = 1
    try:
        print("의료진 채용 검색 에이전트 초기화 중...") 
        await agent.initialize()
        print(f"\n{'='*60}\n요청 처리 시작\n프롬프트: {args.prompt}")
        if args.url: print(f"URL: {args.url}")
        if args.uid: print(f"사용자 ID: {args.uid}")
        print('-'*60)
        result = await agent.process_request(args.url, args.uid, args.prompt)
        print(f"\n🤖 최종 응답 (from main function):\n{result}") 
        exit_code = 0
    except Exception as e:
        main_logger.error(f"메인 실행 오류: {e}", exc_info=True) 
        print(f"메인 실행 중 오류 발생: {e}") 
    finally:
        await agent.cleanup()
    return exit_code

async def process_medical_recruit_request(url: Optional[str] = None, uid: Optional[str] = None, prompt: str = "") -> Dict[str, Any]:
    api_logger = logging.getLogger(f"{__name__}.API")
    agent = MedicalRecruitAgent()
    try:
        await agent.initialize()
        result = await agent.process_request(url, uid, prompt)
        is_error = "오류" in result or "error" in result.lower() or "없었습니다" in result
        return {"success": not is_error, "message": "요청 처리 중 문제 발생." if is_error else "요청 성공.", "data": result}
    except Exception as e:
        api_logger.error(f"API 요청 처리 오류: {e}", exc_info=True)
        return {"success": False, "message": f"API 오류: {str(e)}", "data": None}
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
