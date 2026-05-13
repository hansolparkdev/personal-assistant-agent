# step5_langgraph.py
#
# ---------------------------------------------------------------------------
# Step 5: LangGraph로 ReAct 루프 재구성
# ---------------------------------------------------------------------------
# Step 4(손코딩 for 루프) → Step 5(그래프로 선언)
#
# 핵심 변화:
# - 명시적 for 반복 → 그래프의 사이클(edge)로 표현
# - messages 누적 → add_messages reducer로 자동 병합
# - "도구 더 쓸지?" 판단 → 조건 엣지(conditional_edges)로 분리
#
# 그래프 구조(ASCII):
#
#       START
#         ↓
#       agent ←──────┐
#         ↓          │
#    [tool_calls?]   │
#       ├ Yes → tools
#       └ No  → END
#
# Step 4의 `for iteration` + `if not response.tool_calls` 가
# 위 구조(노드 + 조건 분기 + tools→agent 엣지)로 옮겨진 것.
# ---------------------------------------------------------------------------

# os: 환경변수에서 OPENAI_MODEL 등 읽기
import os
# TypedDict: State의 타입을 dict 형태로 정의
# Annotated: 필드에 메타데이터(여기선 add_messages reducer)를 붙일 때 사용
from typing import TypedDict, Annotated

# .env 로드
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# BaseMessage: HumanMessage/AIMessage/ToolMessage 의 공통 상위 타입
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# StateGraph: 상태 기반 유향 그래프 빌더
# START: 진입점(가상 노드 이름)
# END: 종료점(가상 노드 이름)
from langgraph.graph import StateGraph, START, END
# add_messages: messages 필드 업데이트 시 "교체"가 아니라 "리스트 병합(누적)"
from langgraph.graph.message import add_messages

load_dotenv()


# ---------------------------------------------------------------------------
# 1) 도구 정의 (Step 4와 동일 개념)
# ---------------------------------------------------------------------------
# @tool: 파이썬 함수를 LangChain Tool 규격으로 감싸 LLM에게 노출 가능하게 함
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 알려줍니다."""
    fake_data = {
        "서울": "맑음, 18°C",
        "부산": "흐림, 22°C",
        "제주": "비, 20°C",
        "도쿄": "맑음, 24°C",
    }
    return fake_data.get(city, f"{city}의 날씨 정보 없음")


@tool
def recommend_clothing(weather_description: str) -> str:
    """날씨 설명(예: '맑음, 18°C')을 받아 적절한 옷차림을 추천합니다."""
    if "비" in weather_description:
        return "우산과 방수 자켓을 챙기세요."
    if "20" in weather_description or "18" in weather_description:
        return "얇은 가디건이나 긴팔 셔츠가 적당합니다."
    if "24" in weather_description or "22" in weather_description:
        return "반팔 티셔츠가 좋습니다."
    return "오늘 날씨에 맞는 편한 복장을 추천합니다."


# LLM에게 바인딩할 도구 목록 (순서는 보통 중요하지 않음; 이름으로 구분)
tools = [get_weather, recommend_clothing]
# tool_calls의 "name"(문자열) → 실행 가능 Tool 객체 (invoke용)
tools_by_name = {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# 2) LLM (Step 4와 동일)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3) State 정의 (그래프가 공유하는 데이터 스키마)
# ---------------------------------------------------------------------------
# AgentState: 그래프 한 스텝마다 노드들이 읽고/갱신하는 공통 dict의 형태 정의.
#
# messages에 Annotated[..., add_messages]를 붙이는 이유:
# - Annotated 없이 messages: list[BaseMessage]만 두면 노드 반환값이 전체 상태를 덮어씀에 가까움
# - add_messages reducer 지정 시: 노드가 {"messages": [x]}만 반환해도 기존 messages에 병합(append처럼)
#   → Step 4의 매번 messages.append(...) 를 프레임워크가 대신 처리.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 4) 노드 함수 (상태 입력 → 부분 상태 출력)
# ---------------------------------------------------------------------------
# 노드 시그니처 관례: (state: StateType) -> dict | StateUpdate
# 반환 dict의 키만 State 필드 중 일부만 써도 됨 → 부분 업데이트로 해석됨.


def agent_node(state: AgentState) -> dict:
    """agent 노드: 전체 대화 messages를 LLM에 넣고, AIMessage 1개를 생성해 반환."""
    print("\n--- 🤖 agent 노드 실행 ---")
    # state["messages"]: 지금까지 누적된 대화(사람/어시스턴트/도구결과…)
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print(f"  📞 도구 호출 요청: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"  💬 최종 답변: {response.content[:50]}...")

    # add_messages 덕분에 여기서는 "새 AIMessage만" 넘기면 됨 (전체 리스트를 다시 만들 필요 없음)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """tools 노드: 직전 AIMessage의 tool_calls를 실행하고 ToolMessage들을 반환."""
    print("\n--- 🔧 tools 노드 실행 ---")
    # 가장 최근 메시지는 방금 agent가 낸 AIMessage (tool_calls 포함)라고 가정
    last_message = state["messages"][-1]
    # last_message.tool_calls: list[dict]
    # 예시: [{"name": "get_weather", "args": {"city": "서울"}, "id": "call_..."}]
    # name: 도구 이름
    # args: 도구 파라미터
    # id: 도구 호출 아이디
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name not in tools_by_name:
            error_msg = f"알 수 없는 도구: {tool_name}"
            print(f"  ⚠️  {error_msg}")
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
            continue

        result = tools_by_name[tool_name].invoke(tool_args)
        print(f"  ⚙️  {tool_name}({tool_args}) → {result}")
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# 5) 조건 엣지(라우터): agent 다음이 tools인지 END인지 문자열로 반환
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    """agent 직후 마지막 메시지에 tool_calls가 있으면 'tools', 없으면 'end'."""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# 6) 그래프 조립 (노드 등록 + 엣지 + compile)
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ---------------------------------------------------------------------------
# 7) 그래프 시각화 (구조 점검용)
# ---------------------------------------------------------------------------
print("\n📊 그래프 구조:")
print(graph.get_graph().draw_ascii())


# ---------------------------------------------------------------------------
# 8) 실행 진입점
# ---------------------------------------------------------------------------
def run_agent(user_input: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    # graph.invoke(초기 State, config):
    # - 초기 messages에 HumanMessage 1개만 넣고 시작
    # - recursion_limit: 그래프 스텝(노드 방문) 상한 — 무한 루프 방어 (Step 4의 max_iterations와 유사)
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )
    # 전체 메시지 흐름 확인 (추가)
    print("\n📜 전체 메시지 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        tc_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_info = f" [tool_calls: {[tc['name'] for tc in msg.tool_calls]}]"
        content_preview = (msg.content or "")[:50]
        print(f"  [{i}] {msg_type}{tc_info}: {content_preview}")

    final_message = result["messages"][-1]
    print(f"\n🎯 최종 답변: {final_message.content}")

    return final_message.content


if __name__ == "__main__":
    run_agent("서울 날씨 알려줘")
    run_agent("서울이랑 부산 날씨 알려줘")
    run_agent("서울 날씨 확인하고 거기 맞는 옷차림 추천해줘")
    run_agent("서울이랑 도쿄 날씨 비교해서 더 따뜻한 도시 옷차림 추천해줘")


# =============================================================================
# [마스터 가이드] 이 파일만으로 LangGraph 흐름 정리
# =============================================================================
#
# --- 1) 소스 읽는 순서 (추천) ---
#
#   (가) 파일 맨 위 ASCII 그래프 — 머릿속 도식과 코드가 같은지 확인
#   (나) imports — 무엇이 LangGraph 전용인지 구분 (StateGraph, START, END, add_messages)
#   (다) tools / llm_with_tools — Step 4와 동일 레이어
#   (라) AgentState — "상태 필드 하나인데 reducer로 누적" 이해가 핵심
#   (마) agent_node, tools_node — 각 노드가 state를 어떻게 읽고 무엇을 반환하는지
#   (바) should_continue — 조건 분기의 반환 문자열이 add_conditional_edges의 맵과 일치해야 함
#   (사) builder ... compile() — 그래프 빌드 순서 (노드 → 엣지 → compile)
#   (아) graph.invoke(...) — 초기 상태 + recursion_limit
#   (자) 아래 데이터 흐름 / 구현 흐름 요약 재독해
#
# --- 2) LangGraph 구현 흐름 (코드를 짤 때 순서) ---
#
#   ① State 타입 정의 (TypedDict + 필요 시 Annotated[..., reducer])
#   ② 노드 함수들 작성 (입: state 전체 또는 필요한 필드만, 출: 부분 dict)
#   ③ 라우팅 함수 작성 (조건 엣지용: 문자열 라벨 반환)
#   ④ StateGraph(AgentState) 생성
#   ⑤ add_node로 노드 이름과 함수 등록
#   ⑥ add_edge(START, 첫노드), add_conditional_edges(...), 사이클용 add_edge(...)
#   ⑦ compile() → Runnable 그래프
#   ⑧ invoke/stream 으로 실행
#
# --- 3) 한 번의 invoke 안에서의 데이터 흐름 (이 예제 기준) ---
#
#   초기:
#     state.messages = [HumanMessage("유저 입력")]
#
#   반복 (agent → (조건) → tools → agent → ... → END):
#
#     [agent 노드]
#       입력: state["messages"] 전체
#       동작: llm_with_tools.invoke(...)
#       출력: {"messages": [AIMessage]}  … content 및/또는 tool_calls 포함
#       병합: add_messages → 기존 messages 끝에 AIMessage 추가
#
#     [should_continue]
#       입력: 갱신된 state (마지막 메시지 = 방금 AIMessage)
#       출력: "tools" | "end"
#         tool_calls 비어있음 → "end" → END, invoke 종료
#         tool_calls 있음 → "tools"
#
#     [tools 노드] (분기된 경우만)
#       입력: 마지막 AIMessage.tool_calls 순회
#       동작: tools_by_name[name].invoke(args) → 문자열 결과
#       출력: {"messages": [ToolMessage, ToolMessage, ...]}
#       병합: add_messages로 ToolMessage들이 순서대로 누적
#
#     [agent 노드] (다시)
#       입력: 사람 + 이전 AIMessage + ToolMessage들 전부 포함된 messages
#       동작: LLM이 도구 결과를 읽고 다음 행동(또는 최종 텍스트) 생성
#       … tool_calls 없을 때까지 사이클
#
#   종료 후:
#     result["messages"][-1] 은 마지막 AIMessage로 두고 content를 최종 답으로 씀
#
# --- 4) Step 4 수동 루프와 1:1 대응 ---
#
#   Step 4  messages.append(response)     →  add_messages + 노드 return {"messages":[...]}
#   Step 4  for iteration + max_iter      →  그래프 재귀 + recursion_limit
#   Step 4  if not tool_calls: return      →  should_continue → "end" → END
#   Step 4  도구 for 루프                  →  tools_node 단일 노드로 캡슐화
#
# --- 5) 실수하기 쉬운 점 ---
#
#   - 조건 함수 반환 문자열("tools"/"end")과 add_conditional_edges의 dict 키가 불일치 → 런타임 오류
#   - reducer 없이 messages를 단순 list로 두고 노드가 전체 리스트를 잘못 반환하면 이전 메시지 유실
#   - tools_node는 "마지막 메시지"가 AIMessage라고 가정 — 그래프 구조가 바뀌면 수정 필요
#   - 마지막 메시지가 항상 "최종 답변 AIMessage"라는 가정: 구조 변경 시 종료 처리 로직도 조정
#
# =============================================================================
