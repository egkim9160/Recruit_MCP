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

LANGGRAPH_REACT_CORE_NODES = ["agent", "action", "__end__"] 
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
            except TypeError: text = str(text)
            except Exception: text = str(text)
        return text[:max_len] + '...' if len(text) > max_len else text

    def _get_relevant_node_name(self, serialized: Dict[str, Any]) -> Optional[str]:
        if not serialized or not isinstance(serialized, dict): return None
        name_from_key = serialized.get("name")
        s_id = serialized.get("id") 
        if name_from_key and name_from_key in LANGGRAPH_REACT_CORE_NODES: return name_from_key
        if isinstance(s_id, list) and s_id:
            last_id_part = s_id[-1]
            if isinstance(last_id_part, str) and last_id_part in LANGGRAPH_REACT_CORE_NODES: return last_id_part
        if name_from_key and isinstance(name_from_key, str):
            if "agent" == name_from_key.lower(): return "agent" 
            if "action" == name_from_key.lower(): return "action"
            return name_from_key 
        if isinstance(s_id, list) and s_id:
            last_id_part = s_id[-1]
            if isinstance(last_id_part, str):
                if "agent" == last_id_part.lower(): return "agent"
                if "action" == last_id_part.lower() or "tools" == last_id_part.lower() : return "action" 
                if "__end__" == last_id_part.lower(): return "__end__"
                return last_id_part 
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
        elif isinstance(last_message, ToolMessage): 
            log_parts.append(f"Tool Result (for {last_message.tool_call_id}, Name: {getattr(last_message, 'name', 'N/A')}): {content_preview}")
        else: log_parts.append(f"{type(last_message).__name__}: {self._truncate_log(str(last_message), MAX_LOG_LENGTH_LONG)}")
        logger.info("\n".join(log_parts))

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        node_name = self._get_relevant_node_name(serialized)
        if node_name and node_name in LANGGRAPH_REACT_CORE_NODES:
            self.active_node_names[run_id] = node_name 
            logger.info(f"➡️ GRAPH NODE START: [{node_name}] (run_id: {run_id})")
            if inputs and "messages" in inputs and isinstance(inputs["messages"], list):
                self._log_messages_preview(inputs["messages"], "Input to", node_name, run_id)
            elif inputs:
                logger.info(f"    Input to node [{node_name}] (run_id: {run_id}):\n{self._truncate_log(inputs, MAX_LOG_LENGTH_LONG)}")

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        if run_id in self.active_node_names:
            node_name = self.active_node_names.pop(run_id)
            logger.info(f"⬅️ GRAPH NODE END: [{node_name}] (run_id: {run_id})")
            if outputs and "messages" in outputs and isinstance(outputs["messages"], list):
                self._log_messages_preview(outputs["messages"], "Output from", node_name, run_id)
            elif outputs:
                logger.info(f"    Output from node [{node_name}] (run_id: {run_id}):\n{self._truncate_log(outputs, MAX_LOG_LENGTH_LONG)}")

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
        
        accumulated_thought_text = self.current_llm_thoughts.pop(run_id, "").strip()
        if accumulated_thought_text and not message_content_logged:
            logger.info(f"🧠 LLM Raw Output (from streaming) (run_id: {run_id}):\n{self._truncate_log(accumulated_thought_text, MAX_LOG_LENGTH_LONG)}")

    def on_agent_action(self, action: AgentAction, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        thought_match = re.search(r"Thought:(.*?)(Action:|$)", action.log, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "N/A (Thought not parsed from action.log)"
        if thought != "N/A (Thought not parsed from action.log)":
             logger.info(f"🤔 AGENT THOUGHT (parsed from action.log) (run_id: {run_id}, parent: {parent_run_id}):\n{self._truncate_log(thought, MAX_LOG_LENGTH_LONG)}")

        action_input_str = ""
        if isinstance(action.tool_input, dict): action_input_str = json.dumps(action.tool_input, ensure_ascii=False)
        elif isinstance(action.tool_input, str): action_input_str = action.tool_input
        else: action_input_str = str(action.tool_input)
            
        logger.info(f"🛠️ AGENT ACTION DECISION (run_id: {run_id}, parent: {parent_run_id}): Tool: `{action.tool}`, Input: `{self._truncate_log(action_input_str, MAX_LOG_LENGTH_SHORT)}`")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self.tool_info[run_id] = {"name": tool_name}
        logger.info(f"🔧 TOOL START [{tool_name}] (run_id: {run_id}, parent: {parent_run_id})")
        logger.info(f"    Input: {self._truncate_log(input_str, MAX_LOG_LENGTH_LONG)}")

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, name: Optional[str] = None, **kwargs) -> None:
        # LangGraph의 ToolNode는 name을 인자로 전달해줌
        tool_name_from_args = name
        tool_name_from_info = self.tool_info.pop(run_id, {"name": "unknown_tool"})["name"]
        
        actual_tool_name = tool_name_from_args if tool_name_from_args else tool_name_from_info

        logger.info(f"✅ TOOL END [{actual_tool_name}] (run_id: {run_id}, parent: {parent_run_id})")
        output_str = output
        if hasattr(output, 'content') and isinstance(output.content, str): output_str = output.content
        elif not isinstance(output, str): output_str = str(output)
            
        logger.info(f"    Output: {self._truncate_log(output_str, MAX_LOG_LENGTH_LONG)}")

    def on_tool_error(self, error: BaseException, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, name: Optional[str] = None, **kwargs) -> None:
        tool_name_from_args = name
        tool_name_from_info = self.tool_info.pop(run_id, {"name": "unknown_tool"})["name"]
        actual_tool_name = tool_name_from_args if tool_name_from_args else tool_name_from_info
        logger.error(f"❌ TOOL ERROR [{actual_tool_name}] (run_id: {run_id}, parent: {parent_run_id}): {error}", exc_info=True)

    def on_agent_finish(self, finish: AgentFinish, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs) -> None:
        thought_match = re.search(r"Thought:(.*?)(Final Answer:|$)", finish.log, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "N/A (Thought not parsed from finish.log)"
        if thought != "N/A (Thought not parsed from finish.log)":
            logger.info(f"🤔 AGENT FINAL THOUGHT (parsed from finish.log) (run_id: {run_id}, parent: {parent_run_id}):\n{self._truncate_log(thought, MAX_LOG_LENGTH_LONG)}")
        
        final_answer = finish.return_values.get('output', "N/A (No 'output' in return_values)")
        logger.info(f"🏁 AGENT FINISH (Callback) (run_id: {run_id}, parent: {parent_run_id})\nFinal Answer:\n{self._truncate_log(str(final_answer), MAX_LOG_LENGTH_LONG)}")

class URLParser:
    @staticmethod
    def parse_recruit_url(url: str) -> Optional[str]:
        try:
            match = re.search(r'/recruit/(\d+)', url)
            return match.group(1) if match else None
        except Exception as e:
            logger.error(f"URL 파싱 오류 ({url}): {e}")
            return None

    @staticmethod
    def is_recruit_detail_request(url: str) -> bool:
        return '/recruit/' in url and bool(re.search(r'/recruit/\d+', url))

class MedicalRecruitAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if "LANGSMITH_TRACING" in os.environ: del os.environ["LANGSMITH_TRACING"]
        if "LANGCHAIN_API_KEY" in os.environ: del os.environ["LANGCHAIN_API_KEY"]

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=self.openai_api_key, streaming=True)
        self.client = None
        self.agent_executor = None

    async def initialize(self):
        try:
            server_script_name = "server.py"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_path = os.path.join(current_dir, server_script_name)

            if not os.path.exists(server_path):
                raise FileNotFoundError(f"서버 스크립트 '{server_path}'를 찾을 수 없습니다.")
            logger.info(f"사용할 서버 스크립트 경로: {server_path}")

            self.client = MultiServerMCPClient({"medical_recruit": {"command": "python3", "args": [server_path], "transport": "stdio"}})
            tools = await self.client.get_tools()
            if not tools: raise RuntimeError("MCP 클라이언트에서 도구를 로드하지 못했습니다. 서버 로그를 확인하세요.")
            
            tool_names = [tool.name for tool in tools]
            logger.info(f"로드된 도구 목록: {tool_names}")
            print(f"✅ {len(tools)}개의 도구가 로드되었습니다: {', '.join(tool_names)}")
            
            self.agent_executor = create_react_agent(self.llm, tools)
            print("✅ ReAct 에이전트 실행기가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"초기화 오류: {e}", exc_info=True)
            print(f"❌ 초기화 오류: {e}")
            raise

    async def process_request(self, url: Optional[str], uid: Optional[str], prompt: str) -> str:
        if not self.agent_executor:
            raise ValueError("에이전트가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        try:
            board_id_from_url: Optional[str] = None
            is_detail_page_context = False
            if url:
                if URLParser.is_recruit_detail_request(url):
                    is_detail_page_context = True
                    board_id_from_url = URLParser.parse_recruit_url(url)
                    if board_id_from_url: logger.info(f"URL에서 채용 공고 ID 감지: {board_id_from_url}")
                else: logger.info(f"제공된 URL '{url}'은 채용 공고 상세 페이지 형식이 아닙니다.")

            system_prompt_parts = [
                "당신은 의료진 채용 검색 전문 AI 어시스턴트입니다. 사용자의 요청을 정확히 분석하고, 제공된 도구를 활용하여 최적의 답변을 생성합니다. 항상 친절하고 명확하게 답변하세요.",
                "\n**사용 가능한 도구 목록:**",
                "1. `get_board_by_id(board_id: str)`: 특정 공고 ID로 상세 정보 조회 (JSON 반환).",
                "2. `get_user_by_id(user_id: str)`: 특정 사용자 ID로 프로필 조회 (JSON 반환).",
                "3. `create_and_search_recruits(region: str = None, department: str = None, hospital_name: str = None, semantic_keywords: str = None, exclude_board_id: str = None, size: int = 10)`: 채용공고 검색. 결과로 공고 요약 목록과 `board_ids`를 JSON으로 반환.",
                "4. `create_and_search_users(department: str = None, preferred_region: str = None, semantic_keywords: str = None, size: int = 10)`: 의료진 프로필 검색 (JSON 반환).",
                "5. `create_formatted_recommendations(search_criteria: str, selected_board_ids: List[str])`: LLM이 선별한 최대 5개 공고 ID 목록과 검색 조건으로 사용자 친화적 추천 텍스트와 그 뒤에 JSON 메타데이터 문자열을 포함한 **하나의 긴 문자열**을 생성합니다. 이 도구가 반환하는 **전체 문자열 자체가 최종 사용자 응답**입니다. 이 도구의 결과를 받으면, **절대로 내용을 변경하거나 다른 말을 덧붙이지 말고, 해당 전체 문자열을 그대로 반환하세요.** (반환 형식 예시: '추천 텍스트...\n\n<<METADATA_JSON_START>>\n{\"datatype\": \"recommendation_metadata\", ...}\n<<METADATA_JSON_END>>')",
                "6. `summarize_board_by_id(board_id: str)`: 주어진 `board_id` 공고의 주요 정보 요약 (server.py에서 현재 주석 처리됨).",
                "7. `recommend_popular_jobs(user_id: Optional[str] = None, size: int = 5)`: 인기 공고 추천 (사용자 친화적 문자열 반환). `user_id` 제공 시 해당 사용자 전문과 고려.",
                
                "\n**주요 작업 흐름 및 전략:**",
                "1.  **요청 의도 파악:** 사용자의 질문에서 핵심 요구사항(검색, 추천, 요약, 인기 공고 등)을 정확히 파악합니다.",
                "2.  **정보 수집 (필요시):**",
                "    *   특정 공고 관련(요약, 유사 공고): `board_id`가 있다면 `get_board_by_id` 또는 `summarize_board_by_id` 활용.",
                "    *   사용자 맞춤형: `user_id`가 있다면 `get_user_by_id`로 사용자 정보(특히 전문과) 확인.",
                "3.  **핵심 작업 수행:**",
                "    *   **일반 검색:** `create_and_search_recruits` 사용. 그 결과를 바탕으로 답변하거나, 추가 분석 후 `create_formatted_recommendations` 호출.",
                "    *   **맞춤 추천:**",
                "        1. `create_and_search_recruits`로 후보군 확보 (size는 넉넉하게, 예: 20~30).",
                "        2. **당신(LLM)이 직접** 검색 결과(`results`의 요약 정보)를 분석하여 사용자 요구에 가장 적합한 공고 **최대 5개 선별**.",
                "        3. 선별된 `board_ids`와 원본 검색어(`search_criteria`)를 `create_formatted_recommendations`에 전달. **이 도구가 반환하는 전체 문자열(텍스트 + JSON 메타데이터)이 최종 답변입니다.**",
                "    *   **공고 요약:** `summarize_board_by_id(board_id)` 사용.",
                "    *   **유사 공고:** `get_board_by_id`로 원본 공고 정보 파악 -> `create_and_search_recruits` (원본 공고 조건 활용, `exclude_board_id` 사용).",
                "    *   **인기 공고:** `recommend_popular_jobs` 사용. `user_id`가 있으면 전달하여 전문과 맞춤. 특정 과목 명시 시, `user_id` 없으면 전체 인기 공고 중 LLM이 필터링하거나 사용자에게 안내.",
                "4.  **결과 전달:** `create_formatted_recommendations`를 제외한 다른 도구의 결과(JSON 또는 문자열)는 최종 사용자에게 전달하기 전에 항상 자연스러운 문장으로 가공합니다. `create_formatted_recommendations`의 결과(텍스트 + JSON 메타데이터 문자열)는 **그대로 사용합니다.**",
                
                "\n**중요 규칙:**",
                "-   모든 도구 호출은 논리적인 순서를 따라야 합니다.",
                "-   `create_formatted_recommendations`는 LLM이 **반드시 사전에 공고를 선별한 후** 사용해야 하며, **그 결과인 전체 문자열(텍스트 + JSON 메타데이터)이 최종 사용자 답변이 됩니다.**",
                "-   검색 결과가 없거나 오류 발생 시, 사용자에게 명확하고 친절하게 안내합니다.",
                "-   `recommend_popular_jobs`는 `metadata.view_count` 필드에 의존하므로, 데이터가 없으면 기대대로 작동하지 않을 수 있습니다."
            ]
            active_context_info = []
            if board_id_from_url: active_context_info.append(f"현재 보고 있는 공고 ID (URL에서 추출): {board_id_from_url}")
            if uid: active_context_info.append(f"사용자 ID: {uid}")
            if is_detail_page_context: active_context_info.append("요청 유형: 현재 보고 있는 채용 공고와 관련된 질문일 가능성이 높습니다.")

            if active_context_info:
                system_prompt_parts.append("\n\n**현재 컨텍스트:**")
                system_prompt_parts.extend([f"- {info}" for info in active_context_info])

            if board_id_from_url and any(keyword in prompt.lower() for keyword in ["비슷한", "유사한", "같은 종류", "다른 추천"]):
                system_prompt_parts.append(f"\n**처리 힌트:** 현재 공고(ID: {board_id_from_url})와 유사 공고 요청. `get_board_by_id` -> `create_and_search_recruits` (`exclude_board_id='{board_id_from_url}')")
            elif uid and any(keyword in prompt.lower() for keyword in ["나에게", "내 전문과", "맞춤", "개인화된 추천", "저한테"]):
                system_prompt_parts.append(f"\n**처리 힌트:** 사용자(ID: {uid}) 맞춤형 추천 요청. `get_user_by_id` -> `create_and_search_recruits` 또는 `recommend_popular_jobs` 반영.")
            elif board_id_from_url and any(keyword in prompt.lower() for keyword in ["요약", "간단히", "내용 알려줘"]):
                 system_prompt_parts.append(f"\n**처리 힌트:** 현재 공고(ID: {board_id_from_url}) 요약 요청. `summarize_board_by_id(board_id='{board_id_from_url}')` 사용.")

            final_system_prompt = "\n".join(system_prompt_parts)
            input_data = {"messages": [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": prompt}]}

            logger.info(f"사용자 요청 처리 시작: {prompt} (URL: {url}, UID: {uid})")
            if len(final_system_prompt) < 2000: logger.debug(f"전송될 시스템 프롬프트:\n{final_system_prompt}")
            else: logger.debug(f"전송될 시스템 프롬프트 길이: {len(final_system_prompt)}")

            print(f"에이전트 호출 시작 (프롬프트: {prompt[:100]}...)")
            callback_handler = AgentToolLogger()
            
            response_data = await self.agent_executor.ainvoke(input_data, config={"callbacks": [callback_handler], "recursion_limit": 15})
            logger.info("에이전트 실행 완료 (ainvoke 반환)")

            # 기본적으로 오류 JSON을 반환하도록 설정 (create_formatted_recommendations가 오류 JSON 반환 시 대비)
            final_answer_content = json.dumps({"success": False, "error": "답변을 생성하는 데 문제가 발생했습니다."}, ensure_ascii=False) 
            
            if response_data and "messages" in response_data and isinstance(response_data["messages"], list):
                tool_message_processed = False
                for msg in reversed(response_data["messages"]):
                    # ToolMessage의 name 속성을 사용하여 정확히 create_formatted_recommendations 도구인지 확인
                    if isinstance(msg, ToolMessage) and getattr(msg, 'name', None) == "create_formatted_recommendations":
                        logger.info(f"create_formatted_recommendations 도구의 직접 출력을 최종 답변으로 사용합니다.")
                        final_answer_content = msg.content # "텍스트...\n\n<<METADATA_JSON_START>>\n{JSON}\n<<METADATA_JSON_END>>" 또는 오류 JSON
                        tool_message_processed = True
                        break 
                
                if not tool_message_processed: # create_formatted_recommendations가 사용되지 않은 경우
                    logger.info("create_formatted_recommendations 도구 출력을 찾지 못했습니다. 일반적인 최종 답변 추출 로직을 사용합니다.")
                    # 이 경우, 최종 답변은 일반 텍스트일 가능성이 높음
                    extracted_text_answer = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다." # 기본 텍스트 오류
                    for message in reversed(response_data["messages"]):
                        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
                            extracted_text_answer = str(message.content)
                            break
                    else: 
                        if response_data["messages"]:
                            last_msg_in_list = response_data["messages"][-1]
                            if hasattr(last_msg_in_list, 'content') and last_msg_in_list.content:
                                extracted_text_answer = str(last_msg_in_list.content)
                                logger.warning(f"표준적인 최종 AI 응답을 찾지 못해 messages 리스트의 마지막 메시지 content를 사용합니다: 타입 {type(last_msg_in_list)}")
                            else: logger.warning(f"messages 리스트는 있으나, 표준적인 최종 AI 응답 또는 마지막 메시지 content를 찾지 못했습니다.")
                        else: logger.warning(f"messages 리스트가 비어있어 최종 응답을 추출할 수 없습니다.")
                    final_answer_content = extracted_text_answer

            elif response_data and "output" in response_data: # LangGraph의 create_react_agent는 'output' 키에 최종 결과를 담기도 함
                final_answer_content = str(response_data["output"]) # 이 경우도 일반 텍스트일 가능성
                logger.info("messages 리스트는 없지만 'output' 필드에서 최종 응답을 추출했습니다.")
            
            logger.info(f"최종 응답 문자열 생성 완료 (길이: {len(final_answer_content)}).")
            print(f"에이전트 호출 종료. 응답 길이: {len(final_answer_content)}")
            return final_answer_content

        except Exception as e:
            error_msg = f"요청 처리 중 심각한 오류 발생: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"❌ {error_msg}")
            return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)

    async def cleanup(self):
        if self.client:
            try:
                if hasattr(self.client, '_clients'):
                    for client_name, client_info in self.client._clients.items():
                        actual_client = client_info.get('client')
                        if actual_client and hasattr(actual_client, '_transport') and hasattr(actual_client._transport, 'close'):
                            logger.info(f"'{client_name}' 클라이언트의 transport 정리 중...")
                            actual_client._transport.close()
                        elif actual_client and hasattr(actual_client, 'close'):
                             if asyncio.iscoroutinefunction(actual_client.close): await actual_client.close()
                             else: actual_client.close()
                elif hasattr(self.client, 'close'):
                     if asyncio.iscoroutinefunction(self.client.close): await self.client.close()
                     else: self.client.close()
                logger.info("MCP 클라이언트 정리 시도 완료.")
                print(" MCP 클라이언트가 정리되었습니다 (시도됨).")
            except Exception as e:
                logger.error(f"클라이언트 정리 중 오류: {e}", exc_info=True)
                print(f"⚠️ 클라이언트 정리 중 오류: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='의료진 채용 검색 AI 에이전트 CLI')
    parser.add_argument('--url', type=str, help='옵션: 현재 보고 있는 페이지의 URL')
    parser.add_argument('--uid', type=str, help='옵션: 요청하는 사용자의 ID')
    parser.add_argument('--prompt', type=str, required=True, help='필수: 사용자 질문 또는 요청 사항')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    main_logger = logging.getLogger(f"{__name__}.main_cli")
    agent = MedicalRecruitAgent()
    exit_code = 1
    try:
        print("🚀 의료진 채용 검색 에이전트 초기화 시작...")
        await agent.initialize()
        print(f"\n{'='*60}\n📋 요청 정보:\n  프롬프트: {args.prompt}" + (f"\n  URL: {args.url}" if args.url else "") + (f"\n  사용자 ID: {args.uid}" if args.uid else "") + f"\n{'-'*60}\n⏳ 처리 중...")
        result = await agent.process_request(args.url, args.uid, args.prompt)
        print(f"\n{'='*60}\n🤖 최종 응답 (CLI):\n{'-'*60}\n{result}\n{'-'*60}")
        exit_code = 0
    except FileNotFoundError as e: main_logger.error(f"필수 파일 없음: {e}", exc_info=False); print(f"❌ 필수 파일을 찾을 수 없습니다: {e}")
    except ValueError as e: main_logger.error(f"설정 오류: {e}", exc_info=False); print(f"❌ 설정 오류: {e}")
    except RuntimeError as e: main_logger.error(f"실행 환경 오류: {e}", exc_info=False); print(f"❌ 실행 환경 오류: {e}")
    except Exception as e: main_logger.error(f"알 수 없는 메인 실행 오류: {e}", exc_info=True); print(f"❌ 알 수 없는 메인 실행 오류가 발생했습니다: {e}")
    finally:
        if agent: print("\n🌀 리소스 정리 중..."); await agent.cleanup(); print("🌀 리소스 정리 완료.")
    return exit_code

async def process_medical_recruit_request(url: Optional[str] = None, uid: Optional[str] = None, prompt: str = "") -> Dict[str, Any]:
    api_logger = logging.getLogger(f"{__name__}.API_Handler")
    agent = MedicalRecruitAgent()
    try:
        api_logger.info(f"API 요청 수신: prompt='{prompt[:50]}...', url='{url}', uid='{uid}'")
        await agent.initialize()
        result_content_str = await agent.process_request(url, uid, prompt) 
        
        try:
            # create_formatted_recommendations 도구가 오류 JSON을 반환했는지 확인
            parsed_json_if_tool_error = json.loads(result_content_str)
            if isinstance(parsed_json_if_tool_error, dict) and parsed_json_if_tool_error.get("success") is False:
                api_logger.warning(f"API 요청 처리 중 도구에서 오류 발생 (JSON 오류 응답): {parsed_json_if_tool_error.get('error')}")
                return {"success": False, "message": f"요청 처리 중 오류: {parsed_json_if_tool_error.get('error')}", "data": parsed_json_if_tool_error}
        except json.JSONDecodeError:
            # 오류 JSON이 아니면, 성공적인 "텍스트 + JSON 메타데이터" 문자열이거나, 다른 도구의 일반 텍스트/JSON 응답.
            # 이 경우는 성공으로 간주하고 data에 그대로 전달.
            pass # 다음 로직으로 진행 (성공 처리)
        except Exception as e: # 예상치 못한 파싱 오류
             api_logger.error(f"API 응답 파싱 중 예상치 못한 오류: {e}. 원본 응답: {result_content_str[:200]}")
             # 이 경우도 일단 성공으로 간주하고 원본 문자열 전달, 프론트에서 최종 판단
             pass


        # 에이전트가 반환한 문자열을 성공 응답으로 간주하고 data에 담아 전달
        # 프론트엔드는 이 data 문자열을 파싱하여 사용.
        api_logger.info(f"API 요청 성공적으로 처리됨. 응답 (또는 일부): {result_content_str[:300]}...")
        return {"success": True, "message": "요청이 성공적으로 처리되었습니다.", "data": result_content_str}
            
    except Exception as e: # API 핸들러 자체의 심각한 오류
        api_logger.error(f"API 요청 처리 중 심각한 오류 발생: {e}", exc_info=True)
        return {"success": False, "message": f"API 처리 중 심각한 오류가 발생했습니다: {str(e)}", "data": None}
    finally:
        if agent: await agent.cleanup()

if __name__ == "__main__":
    import sys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하거나 직접 설정해주세요.")
        sys.exit(1)
    sys.exit(asyncio.run(main()))
