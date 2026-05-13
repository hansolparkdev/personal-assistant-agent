# step4_5_agent_executor.py
#
# ---------------------------------------------------------------------------
# Step 4.5: AgentExecutor + create_tool_calling_agent
# ---------------------------------------------------------------------------
# Step 4(손코딩 루프)와 동일한 동작을 LangChain 추상화로 구현.
# 핵심: Agent(두뇌) + AgentExecutor(루프 런타임) 분리.
#
# ⚠️ AgentExecutor는 LangChain 1.0에서 deprecated 상태이지만,
#    레거시 코드/튜토리얼에서 여전히 자주 보이는 패턴이라 학습 가치 있음.
#    실전 신규 프로젝트는 4.7(create_agent) 이상에서 다룰 것.
# ---------------------------------------------------------------------------

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


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

tools = [get_weather, recommend_clothing]

# ---------------------------------------------------------------------------
# 2) LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

# ---------------------------------------------------------------------------
# 3) Prompt 템플릿
# ---------------------------------------------------------------------------
# AgentExcutor는 prompt 템플릿을 필요로 함
# 핵심 자리 표시자{placeholder} 두개
# - input: 사용자 입력
# - agent_scratchpad: 에이전트 메모리(도구 호출 결과 등)
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 한국어 비서야. 사용자 질문에 도구를 활용해 정확하게 답해."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ---------------------------------------------------------------------------
# 4) Agent
# ---------------------------------------------------------------------------
# create_tool_calling_agent가 하는 일:
#   - llm.bind_tools(tools) 자동
#   - prompt + bound_llm + 출력 파서를 하나의 Runnable로 묶음
#   - 입력: {"input": ..., "agent_scratchpad": ...}
#   - 출력: AgentAction(다음 도구 호출) 또는 AgentFinish(최종 답변)
#
# 이 객체 자체는 루프 안 돎. 한 턴만 돌리면 끝.
agent = create_tool_calling_agent(llm, tools, prompt)

# ---------------------------------------------------------------------------
# 5) AgentExecutor 생성 (= "몸뚱이": 루프 런타임)
# ---------------------------------------------------------------------------
# AgentExecutor가 하는 일 (= 너가 손코딩한 run_agent의 내부 동작):
#   1. agent.invoke로 한 턴 추론
#   2. 결과가 AgentFinish면 → 종료, 답변 반환
#   3. 결과가 AgentAction이면 → 도구 실행 → scratchpad에 추가 → 다시 1로
#   4. max_iterations 초과 시 강제 종료
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # 내부 동작을 콘솔에 출력 (디버깅용)
    max_iterations=10,      # 너 코드의 max_iterations와 동일
    return_intermediate_steps=True,  # 중간 도구 호출/결과까지 결과에 포함
)

# ---------------------------------------------------------------------------
# 6) 실행
# ---------------------------------------------------------------------------
def run_agent(user_input: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    # invoke 입력은 prompt의 자리표시자에 매핑되는 dict.
    # {input}만 채우면 됨. agent_scratchpad는 AgentExecutor가 자동 관리.
    result = agent_executor.invoke({"input": user_input})

    # result 구조:
    #   - "input": 원래 입력
    #   - "output": 최종 답변 (str)
    #   - "intermediate_steps": [(AgentAction, observation), ...] 도구 호출 기록
    print(f"\n🤖 최종 답변: {result['output']}")

    # 보너스: 중간 단계 출력 (학습용)
    if result.get("intermediate_steps"):
        print("\n📜 중간 단계:")
        for i, (action, observation) in enumerate(result["intermediate_steps"]):
            print(f"  [{i}] {action.tool}({action.tool_input}) → {observation}")

    return result["output"]


if __name__ == "__main__":
    # Step 4와 동일한 시나리오들
    run_agent("서울 날씨 알려줘")
    run_agent("서울이랑 부산 날씨 알려줘")
    run_agent("서울 날씨 확인하고 거기 맞는 옷차림 추천해줘")
    run_agent("서울이랑 도쿄 날씨 비교해서 더 따뜻한 도시 옷차림 추천해줘")