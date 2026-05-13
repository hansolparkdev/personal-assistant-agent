# step5_3_prebuilt_react.py
#
# ---------------------------------------------------------------------------
# Step 5-3: LangChain 1.0의 create_agent
# ---------------------------------------------------------------------------
# Step 5(직접 그래프 조립)와 동일한 동작을 한 줄에.
#
# 주의: 옛 langgraph.prebuilt.create_react_agent는 deprecated.
#       LangChain 1.0에서 langchain.agents.create_agent로 이동/통합됨.
#
# create_agent가 내부적으로 하는 일:
#   - StateGraph 생성
#   - agent 노드: LLM 호출 + bind_tools 자동
#   - tools 노드: ToolNode 사용
#   - 조건 엣지: tool_calls 있으면 tools로, 없으면 END
#   - tools → agent 사이클
#   - compile()
# ---------------------------------------------------------------------------

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# ★ 새 위치
from langchain.agents import create_agent

load_dotenv()


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


llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)


# ---------------------------------------------------------------------------
# ★ 핵심: create_agent (한 줄)
# ---------------------------------------------------------------------------
# 인자 변화:
#   - model (모델 또는 모델명 문자열)
#   - tools
#   - system_prompt (옛 'prompt' → 명확하게 'system_prompt'로 이름 변경)
#
# 기타 옵션: checkpointer, interrupt_before, debug, ...
agent = create_agent(
    model=llm,
    tools=[get_weather, recommend_clothing],
    system_prompt="너는 친절한 한국어 비서야. 사용자 질문에 도구를 활용해 정확하게 답해.",
)


print("\n📊 create_agent 그래프 구조:")
print(agent.get_graph().draw_ascii())


def run_agent(user_input: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )

    print("\n📜 전체 메시지 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        
        tc_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_info = f"\n      tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}"
        
        tcid_info = ""
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            tcid_info = f"\n      tool_call_id: {msg.tool_call_id}"
        
        content_preview = (msg.content or "")[:80]
        print(f"  [{i}] {msg_type}: {content_preview}{tc_info}{tcid_info}")
    
    final_message = result["messages"][-1]
    print(f"\n🎯 최종 답변: {final_message.content}")
    return final_message.content


if __name__ == "__main__":
    run_agent("서울 날씨 알려줘")
    run_agent("서울이랑 부산 날씨 알려줘")
    run_agent("서울 날씨 확인하고 거기 맞는 옷차림 추천해줘")
    run_agent("서울이랑 도쿄 날씨 비교해서 더 따뜻한 도시 옷차림 추천해줘")