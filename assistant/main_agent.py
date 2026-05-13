# assistant/main_agent.py
#
# ---------------------------------------------------------------------------
# 메인 에이전트 (Supervisor / Orchestrator)
# ---------------------------------------------------------------------------
# 사용자 요청을 받아서 적절한 서브에이전트에게 위임하는 라우터 역할.
# 서브에이전트들을 @tool로 감싸서 자기 도구로 등록한다 (handoff via tool calling).
#
# 메인이 직접 작업을 안 함. 분류와 위임만.
# ---------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# 서브에이전트 import
from assistant.agents.note_agent import note_agent
from assistant.agents.search_agent import search_agent


# ---------------------------------------------------------------------------
# 1) 서브에이전트들을 메인의 "도구"로 wrapping
# ---------------------------------------------------------------------------
# 핵심 트릭: 서브에이전트 호출을 @tool 함수로 감싼다.
# 메인 에이전트 입장에선 그냥 "도구 하나" 처럼 보임.
# LLM이 docstring 보고 "이 요청은 노트 비서 일이네" 판단해서 자동 위임.

@tool
def call_note_agent(query: str) -> str:
    """학습 내용, 메모, 일기 등 **노트** 관련 작업을 노트 비서에게 위임합니다.
    
    이 도구를 사용해야 하는 경우:
    - 사용자가 뭔가를 "기록/저장/메모/노트"하고 싶다고 할 때
    - "어디 적었지", "전에 뭐라고 했지", "오늘 뭐 배웠지" 같은 검색/조회
    - 학습 내용 정리, 일기, 메모 관련 모든 요청
    
    이 도구를 사용하면 안 되는 경우:
    - 웹에서 정보를 찾아야 할 때 (그건 call_search_agent)

    Args:
        query: 노트 비서에게 전달할 자연어 요청. 사용자의 의도를 명확하게 전달.
               예: "LangGraph state에 대해 학습한 내용을 노트로 저장해줘"
    
    Returns:
        노트 비서의 응답.
    """
    result = note_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 25},
    )
    # 서브에이전트의 최종 답변만 반환
    return result["messages"][-1].content


@tool
def call_search_agent(query: str) -> str:
    """
    **웹 검색** 관련 작업을 검색 비서에게 위임합니다.
    
    이 도구를 사용해야 하는 경우:
    - 사용자가 모르는 개념/용어를 설명해야할 때
    - 최신 정보가 필요한 질문 (라이브러리 버전, 뉴스 등)
    - 노트에 저장되지 않은 정보를 찾아야할 때
    
    Args:
        query: 검색 비서에게 전달할 자연어 요청. 사용자의 의도를 명확하게 전달.
            예: "MCP가 뭐야?", "LangChain 새 기능 알려줘"
    
    Returns:
        검색 비서의 응답
    """
    result = search_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content

# ---------------------------------------------------------------------------
# 2) 메인 에이전트 생성
# ---------------------------------------------------------------------------
# tools에 서브에이전트 wrapper들을 등록.
# 앞으로 서브 추가하면 여기 리스트에 wrapper 함수만 추가하면 됨.

_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

_SYSTEM_PROMPT = """너는 박한솔의 범용 비서야. 다양한 전문 서브 비서들을 갖고 있어.

원칙:
- 사용자 요청을 분석해서 어느 서브 비서에게 위임할지 결정해.
- 너가 직접 답하기보다는, 가능하면 서브 비서를 활용해.
- 여러 서브가 필요한 작업이면 순차/병렬로 호출.
  예: "MCP 검색해서 노트로 저장" → search → note
- 서브 비서가 일을 끝내면 그 결과를 자연스럽게 사용자에게 전달.
- 단순 인사나 일상 대화는 직접 답해도 됨.
- 한국어로 답해. 짧고 직설적으로.

현재 사용 가능한 서브 비서:
- 노트 비서 (call_note_agent): 학습 내용/메모/일기 저장 및 검색
- 검색 비서 (call_search_agent): 웹 검색으로 최신 정보 조회
"""

main_agent = create_agent(
    model=_llm,
    tools=[call_note_agent, call_search_agent],  # ← 앞으로 서브 추가하면 여기 늘림
    system_prompt=_SYSTEM_PROMPT,
)