#!/usr/bin/env python3

import os
import json
import asyncio
import logging
import argparse
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class URLParser:
    @staticmethod
    def parse_recruit_url(url: str) -> Optional[str]:
        try:
            pattern = r'/recruit/(\d+)'
            match = re.search(pattern, url)
            if match:
                board_id = match.group(1)
                logger.info(f"URL에서 board_id 추출: {board_id}")
                return board_id
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
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")        
        
        # LangSmith 추적 비활성화
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if "LANGSMITH_TRACING" in os.environ:
            del os.environ["LANGSMITH_TRACING"]
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=self.openai_api_key)
        self.client = None
        self.agent = None
        
    async def initialize(self):
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
            logger.info(f"로드된 도구 목록: {[tool.name for tool in tools]}")
            print(f"✅ {len(tools)}개의 도구가 로드되었습니다.")
            
            self.agent = create_react_agent(self.llm, tools)
            print("✅ 에이전트가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"초기화 오류: {e}", exc_info=True)
            print(f"❌ 초기화 오류: {e}")
            raise

    async def process_request(self, url: Optional[str], uid: Optional[str], prompt: str) -> str:
        if not self.agent:
            raise ValueError("에이전트가 초기화되지 않았습니다.")
        
        try:
            # URL 파싱
            board_id = None
            is_detail_request = False
            
            if url:
                is_detail_request = URLParser.is_recruit_detail_request(url)
                if is_detail_request:
                    board_id = URLParser.parse_recruit_url(url)
                    logger.info(f"채용 공고 상세 요청 감지 - board_id: {board_id}")
            
            # 컨텍스트 구성
            context_info = []
            if board_id:
                context_info.append(f"현재 보고 있는 공고 ID: {board_id}")
            if uid:
                context_info.append(f"사용자 ID: {uid}")
            if is_detail_request:
                context_info.append("요청 유형: 채용 공고 상세 조회 관련")
            
            # 시스템 프롬프트
            system_prompt = """
당신은 의료진 채용 검색 전문 AI 어시스턴트입니다. 사용자의 요청에 따라 의료진 채용 공고를 검색하고 결과를 제공합니다.

**사용 가능한 도구:**
1. create_recruit_search_query: 공고 검색 쿼리 생성
2. search_recruits: 공고 데이터베이스 검색  
3. format_results: 결과 포맷팅
4. get_recruit_by_id: 특정 공고 조회
5. get_user_by_id: 사용자 정보 조회
6. parse_url_for_board_id: URL에서 board_id 추출

**처리 순서:**
1. 사용자 요청 분석
2. 적절한 검색 쿼리 생성
3. 검색 실행
4. 결과 포맷팅
5. 사용자 친화적인 최종 답변 제공

**주의사항:**
- 각 도구는 한 번씩만 사용
- 현재 공고 ID가 있으면 get_recruit_by_id 먼저 사용
- 사용자 ID가 있으면 맞춤 추천 제공
"""
            
            if context_info:
                system_prompt += f"\n\n**현재 컨텍스트:**\n" + "\n".join(context_info)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"사용자 요청 처리: {prompt}")
            
            response = await self.agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": 10}
            )
            
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    content = str(last_message.content)
                    logger.info(f"응답 생성 완료: {len(content)} 문자")
                    return content
            
            return "응답을 생성할 수 없습니다."
            
        except Exception as e:
            error_msg = f"요청 처리 중 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    async def cleanup(self):
        if self.client:
            try:
                await self.client.close() if hasattr(self.client, 'close') else None
                logger.info("클라이언트 정리 완료")
            except Exception as e:
                logger.error(f"클라이언트 정리 오류: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='의료진 채용 검색 AI 에이전트')
    parser.add_argument('--url', type=str, help='요청 URL')
    parser.add_argument('--uid', type=str, help='사용자 ID')
    parser.add_argument('--prompt', type=str, required=True, help='사용자 질문')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    agent = MedicalRecruitAgent()
    
    try:
        print("🚀 의료진 채용 검색 에이전트 초기화 중...")
        await agent.initialize()
        
        print(f"\n" + "="*60)
        print(f"📋 요청 처리 시작")
        print(f"📝 프롬프트: {args.prompt}")
        if args.url:
            print(f"🔗 URL: {args.url}")
            if URLParser.is_recruit_detail_request(args.url):
                board_id = URLParser.parse_recruit_url(args.url)
                print(f"📌 감지된 공고 ID: {board_id}")
        if args.uid:
            print(f"👤 사용자 ID: {args.uid}")
        print("-"*60)
        
        result = await agent.process_request(args.url, args.uid, args.prompt)
        print(f"🤖 응답:\n{result}")
        
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}", exc_info=True)
        print(f"❌ 오류 발생: {e}")
        return 1
    finally:
        await agent.cleanup()
    
    return 0

# API 호환성 유지
async def process_medical_recruit_request(url: Optional[str] = None, 
                                        uid: Optional[str] = None, 
                                        prompt: str = "") -> Dict[str, Any]:
    agent = MedicalRecruitAgent()
    
    try:
        await agent.initialize()
        result = await agent.process_request(url, uid, prompt)
        
        return {
            "success": True,
            "message": "요청이 성공적으로 처리되었습니다.",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"API 요청 처리 오류: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"요청 처리 중 오류: {str(e)}",
            "data": None
        }
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
