#!/usr/bin/env python3

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨로 설정하여 중요한 정보만 표시
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            logger.info(f"서버 경로: {server_path}")
            
            self.client = MultiServerMCPClient({
                "medical_recruit": {
                    "command": "python",
                    "args": [server_path],
                    "transport": "stdio",
                }
            })
            
            tools = await self.client.get_tools()
            logger.info(f"로드된 도구 목록: {[tool.name for tool in tools]}")
            print(f"✅ {len(tools)}개의 도구가 로드되었습니다.")
            
            self.agent = create_react_agent(self.llm, tools)
            print("✅ 에이전트가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"초기화 오류: {e}", exc_info=True)
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
            
            # 명확하고 효과적인 시스템 프롬프트
            system_prompt = """
당신은 의료진 채용 검색 전문 AI 어시스턴트입니다. 사용자의 요청에 따라 의료진 채용 공고를 검색하고 결과를 제공합니다.

**사용 가능한 도구:**
1. create_recruit_search_query: 공고 검색 쿼리 생성
2. search_recruits: 공고 데이터베이스 검색  
3. format_results: 결과 포맷팅
4. get_recruit_by_id: 특정 공고 조회
5. create_user_search_query: 사용자 기반 검색 쿼리 생성
6. search_users: 사용자 데이터베이스 검색

**처리 순서:**
1. 사용자 요청 분석
2. 적절한 검색 쿼리 생성 (create_recruit_search_query 또는 get_recruit_by_id)
3. 검색 실행 (search_recruits)
4. 결과 포맷팅 (format_results)
5. 사용자 친화적인 최종 답변 제공

**중요한 주의사항:**
- 각 도구는 한 번씩만 사용하세요
- search_recruits에는 create_recruit_search_query의 결과를 그대로 전달하세요
- 오류 발생 시 재시도하지 말고 사용자에게 상황을 설명하세요
- 검색 결과가 없어도 유용한 정보를 제공하세요

지역, 진료과, 연봉, 고용형태 등 다양한 조건으로 검색할 수 있습니다.
"""
            
            if context_info:
                system_prompt += f"\n\n**현재 컨텍스트:**\n" + "\n".join(context_info)
            
            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"사용자 요청 처리 시작: {prompt}")
            
            # 에이전트 실행 
            response = await self.agent.ainvoke(
                {"messages": messages},
                config={
                    "recursion_limit": 10,  # 충분한 재귀 한도 설정
                }
            )
            
            logger.info("에이전트 실행 완료")
            
            # 응답 처리
            if response and "messages" in response:
                last_message = response["messages"][-1]
                
                if hasattr(last_message, 'content') and last_message.content:
                    content = last_message.content
                    logger.info(f"응답 생성 완료: {len(str(content))} 문자")
                    return str(content)
                else:
                    logger.warning(f"마지막 메시지에 content 속성이 없음")
                    return "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다."
            else:
                logger.error(f"예상치 못한 응답 형식")
                return "응답을 생성할 수 없습니다."
            
        except Exception as e:
            error_msg = f"요청 처리 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    async def cleanup(self):
        """리소스 정리"""
        if self.client:
            try:
                await self.client.close() if hasattr(self.client, 'close') else None
                logger.info("클라이언트 정리 완료")
                print("✅ 클라이언트가 정리되었습니다.")
            except Exception as e:
                logger.error(f"클라이언트 정리 중 오류: {e}")
                print(f"⚠️ 클라이언트 정리 중 오류: {e}")

async def main():
    """메인 실행 함수 - 다양한 테스트 시나리오"""
    agent = MedicalRecruitAgent()
    
    try:
        print("🚀 의료진 채용 검색 에이전트 초기화 중...")
        await agent.initialize()
        
        # 다양한 테스트 시나리오들
        test_scenarios = [
            {
                "name": "지역별 피부과 공고 검색",
                "board_id": None,
                "user_id": None,
                "prompt": "서울 지역 피부과 공고 찾아줘"
            },
            {
                "name": "조건별 내과 공고 검색", 
                "board_id": None,
                "user_id": None,
                "prompt": "부산 지역 내과 정규직 공고 있어?"
            },
            {
                "name": "급여 조건 포함 검색",
                "board_id": None, 
                "user_id": None,
                "prompt": "연봉 5000만원 이상 정형외과 공고 추천해줘"
            }
        ]
        
        # 첫 번째 테스트 시나리오만 실행 (필요시 더 추가 가능)
        scenario = test_scenarios[0]
        
        print(f"\n" + "="*60)
        print(f"📋 테스트: {scenario['name']}")
        print(f"📝 요청: {scenario['prompt']}")
        if scenario.get('board_id'):
            print(f"📌 공고 ID: {scenario['board_id']}")
        if scenario.get('user_id'):
            print(f"👤 사용자 ID: {scenario['user_id']}")
        print("-"*60)
        
        result = await agent.process_request(
            board_id=scenario['board_id'],
            user_id=scenario['user_id'],
            prompt=scenario['prompt']
        )
        
        print(f"🤖 응답:\n{result}")
        
        # 추가 테스트를 원한다면 사용자 입력 받기
        print(f"\n" + "="*60)
        print("💬 추가 질문이 있으시면 입력하세요 (종료하려면 'quit' 입력):")
        
        while True:
            try:
                user_input = input("\n질문: ").strip()
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    break
                    
                if user_input:
                    print("🔍 검색 중...")
                    result = await agent.process_request(None, None, user_input)
                    print(f"🤖 응답:\n{result}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}", exc_info=True)
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
        logger.error(f"API 요청 처리 오류: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "data": None
        }
    
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
