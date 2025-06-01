#!/usr/bin/env python3

import os
import json
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

class MedicalRecruitAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")        
        # LangSmith 추적 비활성화 (403 오류 방지)
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if "LANGSMITH_TRACING" in os.environ:
            del os.environ["LANGSMITH_TRACING"]
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=self.openai_api_key
        )
        
        self.client = None
        self.agent = None
        
    async def initialize(self):
        """MCP 클라이언트와 에이전트를 초기화합니다."""
        try:
            server_path = os.path.abspath("server.py")
            
            self.client = MultiServerMCPClient({
                "medical_recruit": {
                    "command": "python",
                    "args": [server_path],
                    "transport": "stdio",
                }
            })
            
            tools = await self.client.get_tools()
            print(f"✅ {len(tools)}개의 도구가 로드되었습니다.")
            
            self.agent = create_react_agent(self.llm, tools)
            print("✅ 에이전트가 초기화되었습니다.")
            
        except Exception as e:
            print(f"❌ 초기화 오류: {e}")
            raise

    async def process_request(self, board_id: Optional[str], user_id: Optional[str], prompt: str) -> str:
        """
        사용자 요청을 처리합니다.
        
        Args:
            board_id: 현재 보고 있는 공고 ID (선택적)
            user_id: 사용자 ID (선택적)
            prompt: 사용자 질문/요청
            
        Returns:
            처리 결과 문자열
        """
        if not self.agent:
            raise ValueError("에이전트가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        try:
            # 컨텍스트 정보를 포함한 시스템 메시지 구성
            context_info = []
            if board_id:
                context_info.append(f"현재 보고 있는 공고 ID: {board_id}")
            if user_id:
                context_info.append(f"사용자 ID: {user_id}")
            
            # 시스템 프롬프트 구성
            system_prompt = """
당신은 의료진 채용 검색 전문 AI 어시스턴트입니다. 다음 도구들을 활용하여 사용자의 요청을 처리하세요:

1. create_recruit_search_query: 공고 검색 쿼리 생성
2. create_user_search_query: 사용자 기반 검색 쿼리 생성  
3. search_recruits: 공고 데이터베이스 검색
4. search_users: 사용자 데이터베이스 검색
5. get_recruit_by_id: 특정 공고 조회
6. format_results: 결과 포맷팅

주요 시나리오 처리 방법:
- 조건별 공고 검색: 지역, 진료과, 연봉 등으로 필터링 후 검색
- 유사 공고 찾기: 현재 공고 조회 → 유사 조건으로 검색
- 맞춤 공고 추천: 사용자 정보 기반 검색
- 인기 공고 찾기: 유사 사용자들의 지원 이력 분석

항상 사용자 친화적이고 구체적인 답변을 제공하세요.
"""
            
            if context_info:
                system_prompt += f"\n\n현재 컨텍스트:\n" + "\n".join(context_info)
            
            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # 에이전트 실행 (재귀 한도 설정)
            response = await self.agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": 5}
            )
            
            # 응답에서 마지막 메시지 추출
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            
            return "응답을 생성할 수 없습니다."
            
        except Exception as e:
            error_msg = f"요청 처리 중 오류가 발생했습니다: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def cleanup(self):
        """리소스 정리"""
        if self.client:
            try:
                # 클라이언트 정리 (구체적인 메서드는 라이브러리에 따라 다를 수 있음)
                await self.client.close() if hasattr(self.client, 'close') else None
                print("✅ 클라이언트가 정리되었습니다.")
            except Exception as e:
                print(f"⚠️ 클라이언트 정리 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'client') and self.client:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
        except:
            pass

async def main():
    """테스트 및 데모 함수"""
    agent = MedicalRecruitAgent()
    
    try:
        print("🚀 의료진 채용 검색 에이전트를 초기화하는 중...")
        await agent.initialize()
        
        # 테스트 시나리오들
        test_scenarios = [
            {
                "name": "지역별 마취통증의학과 공고 검색",
                "board_id": None,
                "user_id": None,
                "prompt": "서울 지역에 연봉 3000만원 이상 마취통증의학과 공고 찾아줘"
            },
            {
                "name": "현재 공고와 유사한 공고 찾기",
                "board_id": "test_board_123",
                "user_id": None,
                "prompt": "지금 보고 있는 공고와 비슷한 조건의 공고 찾아줘"
            },
            {
                "name": "사용자 맞춤 공고 추천",
                "board_id": None,
                "user_id": "test_user_456",
                "prompt": "나의 조건에 적합한 공고를 찾아줘"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n" + "="*60)
            print(f"📋 테스트: {scenario['name']}")
            print(f"📝 요청: {scenario['prompt']}")
            if scenario['board_id']:
                print(f"📌 공고 ID: {scenario['board_id']}")
            if scenario['user_id']:
                print(f"👤 사용자 ID: {scenario['user_id']}")
            print("-"*60)
            
            result = await agent.process_request(
                board_id=scenario['board_id'],
                user_id=scenario['user_id'],
                prompt=scenario['prompt']
            )
            
            print(f"🤖 응답:\n{result}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    finally:
        await agent.cleanup()

# API 스타일 사용을 위한 함수
async def process_medical_recruit_request(board_id: Optional[str] = None, 
                                        user_id: Optional[str] = None, 
                                        prompt: str = "") -> Dict[str, Any]:
    """
    외부에서 호출할 수 있는 API 스타일 함수
    
    Args:
        board_id: 현재 보고 있는 공고 ID
        user_id: 사용자 ID  
        prompt: 사용자 질문/요청
        
    Returns:
        결과 딕셔너리 (success, message, data)
    """
    agent = MedicalRecruitAgent()
    
    try:
        await agent.initialize()
        result = await agent.process_request(board_id, user_id, prompt)
        
        return {
            "success": True,
            "message": "요청이 성공적으로 처리되었습니다.",
            "data": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "data": None
        }
    
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
