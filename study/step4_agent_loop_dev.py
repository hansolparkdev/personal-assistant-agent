import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

@tool(name="get_weather", description="특정 도시의 현재 날씨를 알려줍니다.")
def get_weather(city: str) -> str:
    fake_data = {
        "서울": "맑음, 18°C",
        "부산": "흐림, 22°C",
        "제주": "비, 20°C",
        "도쿄": "맑음, 24°C",
    }
    return fake_data.get(city, f"{city}의 날씨 정보 없음")

@tool(name="recommend_clothing", description="날씨 설명(예: '맑음, 18°C')을 받아 적절한 옷차림을 추천합니다.")
def recommend_clothing(weather_description: str) -> str:
    if "비" in weather_description:
        return "우산과 방수 자켓을 챙기세요."
    if "20" in weather_description or "18" in weather_description:
        return "얇은 가디건이나 긴팔 셔츠가 적당합니다."
    if "24" in weather_description or "22" in weather_description:
        return "반팔 티셔츠가 좋습니다."
    return "오늘 날씨에 맞는 편한 복장을 추천합니다."


model = os.environ.get("OPENAI_MODEL")
llm = ChatOpenAI(model=model, temperature=0)

tools = [get_weather, recommend_clothing]
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in tools}

def run_agent(user_input: str, max_iterations: int = 10) -> str:
    # 대화 시작 상태
    messages = [HumanMessage(content=user_input)]

    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 🔁 Iteration {iteration} ---")
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            print(f"\n최종 답변: {response.content}")
            return response.content

        # 도구 호출 단계
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name not in tools_by_name:
                error_msg = f"알 수 없는 도구: {tool_name}"
                print(f"  ⚠️  {error_msg}")
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                # continue 동작원리 설명
                # continue는 루프제어 키워드로 다음 tool_call 실행
                # 즉, 현재 tool_call 실행 중단하고 다음 tool_call 실행
                continue
            # 도구 실행 단계
            # Tool.invoke(args):
            # - args(dict)를 함수 파라미터로 매핑해 실제 함수 실행
            # - 반환값(result)은 문자열/JSON 직렬화 가능한 값이 일반적
            result = tools_by_name[tool_name].invoke(tool_args)
            print(f"  ⚙️  {tool_name}({tool_args}) → {result}")
            messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    print(f"\n⚠️  최대 반복 횟수({max_iterations}) 초과")
    return "에이전트가 작업을 완료하지 못했습니다."