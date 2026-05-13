# assistant/supervisor_graph.py
#
# ---------------------------------------------------------------------------
# Supervisor 그래프 (StateGraph 직접 구성)
# ---------------------------------------------------------------------------
# Step 6의 create_agent 기반 main_agent를 StateGraph로 마이그레이션.
#
# 변경 사항:
# - 메인: create_agent → StateGraph 직접 구성
# - 서브: create_agent 그대로 (note_agent, search_agent)
# - 라우팅: tool_calls를 명시적 조건 엣지로 처리
#
# 학습 포인트:
# - StateGraph 멀티 에이전트 패턴
# - Supervisor 노드와 서브 노드의 관계
# - 핸드오프 도구를 라우팅 신호로 사용
# ---------------------------------------------------------------------------

import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState

from assistant.agents.note_agent import note_agent
from assistant.agents.search_agent import search_agent

# ---------------------------------------------------------------------------
# 1) 핸드오프 도구
# ---------------------------------------------------------------------------
# Supervisor가 "어느 서브로 갈지" 결정할 때 호출하는 도구.
# 실제로는 아무것도 안 함. tool_calls가 라우팅 신호 역할.

@tool
def transfer_to_note_agent(query: str) -> str:
    """**사용자가 직접 저장한** 개인 노트를 저장/검색합니다.

    이 도구를 사용하는 경우:
    - "내가 적은/저장한/기록한 X 보여줘"
    - "오늘 뭐 적었지", "전에 뭐라고 했지"
    - 새로 메모/학습 내용/일기 저장

    Args:
        query (str): 노트 비서에게 전달할 자연어 요청
    """
    # 실제 실행은 note_node에서. 여기선 라우팅 신호만 처리.
    return ""

@tool
def transfer_to_search_agent(query: str) -> str:
    """**웹**에서 최신 정보를 검색합니다. (사용자 개인 노트가 아님)

    이 도구를 사용하는 경우:
    - "X가 뭐야?", "X 알려줘" (일반 지식)
    - "최신 X", "최근 X"
    - 사용자가 모르는 외부 개념
    - 노트에 없는 경우

    Args:
        query (str): 검색 비서에게 전달할 자연어 요청.
    """
    return ""

handoff_tools = [transfer_to_note_agent, transfer_to_search_agent]

# ---------------------------------------------------------------------------
# 2) Supervisor LLM (도구 호출 가능)
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)
_llm_with_handoff = _llm.bind_tools(handoff_tools)

_SUPERVISOR_PROMPT = """너는 범용 비서를 총괄하는 supervisor야

너는 직접 답을 하지않아. 사용자 요청을 분석해서 적절한 서브 비서에게 위임해

사용 가능한 서브 비서:
- 노트 비서 (transfer_to_note_agent): 학습 내용/메모/일기 저장 및 검색
- 검색 비서 (transfer_to_search_agent): 웹 검색으로 최신 정보 조회

라우팅 규칙:
- "내가 저장한/적은 X" → 노트 비서
- "X가 뭐야?", "최신 X" → 검색 비서
- 복합 요청 ("검색해서 노트로 저장") → 순차 호출
  먼저 검색 비서 부르고, 결과 받은 후 노트 비서 호출.

원칙:
- 서브 결과가 나오면 그걸 사용자에게 자연스럽게 전달.
- 더 위임할 게 없으면 그냥 텍스트로 답변.
- 단순 인사는 직접 답해도 됨.
- 한국어로, 짧고 직설적으로. 
"""
# ---------------------------------------------------------------------------
# 3) 노드 정의
# ---------------------------------------------------------------------------

def supervisor_node(state: MessagesState) -> dict:
    """Supervisor 노드: LLM이 라우팅 결정 또는 최종 답변.

    Args:
        state (MessagesState): 지금까지 누적된 대화 히스토리

    Returns:
        dict: 다음 노드로 전달할 부분 상태
    """
    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *state["messages"]]

    response = _llm_with_handoff.invoke(messages)
    if response.tool_calls:
        print(f"supervisor 라우팅 결정: {response.tool_calls}")
    else:
        print(f"supervisor 최종 답변: {response.content}")

    return {"messages": [response]}

def note_node(state: MessagesState) -> dict:
    """노트 비서 노드: 학습 내용/메모/일기 저장 및 검색.

    Args:
        state (MessagesState): 지금까지 누적된 대화 히스토리

    Returns:
        dict: 다음 노드로 전달할 부분 상태
    """
    print("\nnote 에이전트 호출중 ...")

    last_message = state["messages"][-1]
    # tansfer_to_note_agent 도구 호출 정보 추출
    tool_call = next(
        tc for tc in last_message.tool_calls if tc["name"] == "transfer_to_note_agent"
    )
    query = tool_call["args"]["query"]
    tool_call_id = tool_call["id"]
    print("=" * 50)
    print(f"노트 비서에게 전달할 요청: {query}")
    print("=" * 50)

    #서브에이전트 호출
    result = note_agent.invoke({"messages": [HumanMessage(content=query)]})
    response_content = result["messages"][-1].content
    
    # ToolMessage로 반환 -> supervisor의 tool_calls에 응답하는 형태
    return {
        "messages": [
            ToolMessage(content=response_content, 
            tool_call_id=tool_call_id)
        ]
    }


def search_node(state: MessagesState) -> dict:
    """검색 비서 노드: 웹 검색으로 최신 정보 조회.

    Args:
        state (MessagesState): 지금까지 누적된 대화 히스토리

    Returns:
        dict: 다음 노드로 전달할 부분 상태
    """
    print("\nsearch 에이전트 호출중 ...")

    last_message = state["messages"][-1]
    # tansfer_to_search_agent 도구 호출 정보 추출
    tool_call = next(
        tc for tc in last_message.tool_calls if tc["name"] == "transfer_to_search_agent"
    )
    query = tool_call["args"]["query"]
    tool_call_id = tool_call["id"]
    print("=" * 50)
    print(f"검색 비서에게 전달할 요청: {query}")
    print("=" * 50)

    #서브에이전트 호출
    result = search_agent.invoke({"messages": [HumanMessage(content=query)]})
    response_content = result["messages"][-1].content

    # ToolMessage로 반환 -> supervisor의 tool_calls에 응답하는 형태
    return {
        "messages": [
            ToolMessage(content=response_content, 
            tool_call_id=tool_call_id)
        ]
    }

# ---------------------------------------------------------------------------
# 4) 라우터 (조건 엣지 함수)
# ---------------------------------------------------------------------------

def route_after_supervisor(state: MessagesState) -> Literal["note", "search", "end"]:
    """supervisor 노드 후 라우팅 결정.

    Args:
        state (MessagesState): 지금까지 누적된 대화 히스토리

    Returns:
        Literal["note", "search", "end"]: 다음 노드로 전달할 라우팅 결정
    """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    
    # tool_calls가 있으면 어떤 도구인지 확인
    tool_call = last_message.tool_calls[0]
    if tool_call["name"] == "transfer_to_note_agent":
        return "note"
    if tool_call["name"] == "transfer_to_search_agent":
        return "search"
    return "end"

# ---------------------------------------------------------------------------
# 5) 그래프 정의
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("note", note_node)
builder.add_node("search", search_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_after_supervisor, {
    "note": "note",
    "search": "search",
    "end": END,
})

builder.add_edge("note", "supervisor")
builder.add_edge("search", "supervisor")

supervisor_graph = builder.compile()

# ---------------------------------------------------------------------------
# 6) 시각화
# ---------------------------------------------------------------------------
print("\n📊 Supervisor 그래프 구조:")
print(supervisor_graph.get_graph().draw_ascii())