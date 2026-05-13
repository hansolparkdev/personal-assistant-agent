# assistant/agents/search_agent.py
#
# ---------------------------------------------------------------------------
# 검색 서브에이전트
# ---------------------------------------------------------------------------
# 웹 검색 도구를 들고 사용자 질문에 최신 정보로 답한다.
# 단순 1-도구 에이전트지만, 검색 결과를 어떻게 요약/정리할지 LLM이 판단.
# ---------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools.search_tools import search_tools


# 검색 에이전트 전용 LLM
# (메인 에이전트와 동일 모델을 써도 되고, 더 작은/저렴한 모델로 바꿔도 됨)
_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

# 검색 전문 시스템 프롬프트
_SYSTEM_PROMPT = """너는 웹 검색 전문 비서야. 사용자의 질문에 최신 정보로 답해.

원칙:
- 모르는 개념이나 최신 정보 질문이면 무조건 web_search 사용
- 검색 결과를 그대로 나열하지 말고, 질문에 맞게 정리해서 답해
- 검색 결과 중 신뢰할 만한 것 (공식 문서, 위키피디아, 큰회사 블로그) 우선
- 너무 길지 않게 핵심만 짧게
- 한국어로 답해. 기술 용어는 영어 그대로.
- 답에 출처 URL 1~2개 포함해

"""

# 서브에이전트 인스턴스
search_agent = create_agent(
    model=_llm,
    tools=search_tools,
    system_prompt=_SYSTEM_PROMPT,
)