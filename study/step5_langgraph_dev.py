import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# LangGraph 핵심 import
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ---------------------------------------------------------------------------
# 1) 도구 정의 (Step 4와 동일)
# ---------------------------------------------------------------------------
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

tools = [get_weather, recommend_clothing]
tools_by_name = {t.name: t for t in tools}

# ---------------------------------------------------------------------------
# 2) LLM (Step 4와 동일)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState) -> dict:
    """agent 노드: 전체 대화 messages를 LLM에 넣고 AIMessage 1개를 생성해 반환."""
    print("\n agent 노드 실행")
    # state 전체 확인
    # print(f"state: {state}")
    # state["messages"]: 지금까지 누적된 대화(사람/어시스턴트/도구결과...)
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"  📞 도구 호출 요청: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"  💬 최종 답변: {response.content[:50]}...")
    return {"messages": [response]}

def tools_node(state: AgentState) -> dict:
    """tools 노드: 직전 AIMessage의 tool_calls를 실행하고 ToolMessage들을 반환."""
    print("\n tools 노드 실행")
    # 가장 최근 메시지는 방금 agent가 낸 AIMessage (tool_calls 포함)라고 가정
    # 이게 왜필요하지?
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
        tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """agent 직후 마지막 메시지에 tool_calls가 있으면 'tools', 없으면 'end'."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
builder.add_edge("tools", "agent")
graph = builder.compile()

# 그래프 시작화
print("\n📊 그래프 구조:")
print(graph.get_graph().draw_ascii())

def run_agent(user_input: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )
    
    print("\n📜 전체 메시지 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        
        # AIMessage의 tool_calls
        tc_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_info = f"\n      tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}"
        
        # ToolMessage의 tool_call_id (매칭 확인용)
        tcid_info = ""
        if hasattr(msg, "tool_call_id"):
            tcid_info = f"\n      tool_call_id: {msg.tool_call_id}"
        
        content_preview = (msg.content or "")[:80]
        print(f"  [{i}] {msg_type}: {content_preview}{tc_info}{tcid_info}")
    
    final_message = result["messages"][-1]
    print(f"\n🎯 최종 답변: {final_message.content}")
    return final_message.content

if __name__ == "__main__":
    run_agent("서울 날씨 알려줘")