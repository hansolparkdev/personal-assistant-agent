# step3_tool_full.py
#
# ---------------------------------------------------------------------------
# 이 예제의 목표(“Tool Calling” 흐름 이해)
# ---------------------------------------------------------------------------
# LangChain에서 "도구를 붙인(ChatOpenAI + tools) LLM"을 사용할 때의 전형적인 3단계 흐름입니다.
#
# 1) 사람(HumanMessage)이 질문한다.
# 2) LLM이 “답하려면 도구를 실행해야 함”을 판단하면, 즉시 답을 생성하지 않고
#    "어떤 도구를 어떤 인자로 호출해라"라는 구조화된 요청을 tool_calls로 반환한다.
# 3) 우리는 그 tool_calls를 실제 파이썬 함수로 실행한 뒤,
#    결과를 ToolMessage로 messages에 추가해서 다시 LLM에 주고,
#    LLM이 최종 자연어 답변(AIMessage.content)을 생성한다.
#
# 여기서 가장 중요한 점:
# - LLM과의 대화 기록은 messages 리스트에 “순서대로” 누적된다.
# - tool_calls에 들어있는 각 호출에는 id가 있고,
#   ToolMessage(tool_call_id=...)로 그 결과를 정확히 매칭해줘야 한다.
#
# 타입 감각(자주 헷갈리는 부분)
# - response / final_response: 보통 AIMessage 인스턴스(= 클래스 인스턴스 객체)
# - response.tool_calls: 보통 dict들의 리스트(list[dict])라서 tc["name"]처럼 키 접근을 한다
#   (즉, “메시지 객체” 안에 “dict 구조”가 섞여 있는 형태)

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# 0) 환경 변수 로드
# ---------------------------------------------------------------------------
# load_dotenv()는 .env 파일에 정의된 값을 현재 프로세스 환경(os.environ)에 넣어줍니다.
# 예: OPENAI_API_KEY, OPENAI_MODEL 등이 .env에 들어있고, 이후 코드에서 os.environ[...]로 읽습니다.
load_dotenv()


# ---------------------------------------------------------------------------
# 1) "도구" 정의: @tool로 파이썬 함수를 LangChain Tool로 감싸기
# ---------------------------------------------------------------------------
# @tool은 다음을 가능하게 합니다.
# - LLM에게 "이런 이름/입력/출력/설명을 가진 도구가 있다"를 전달
# - 도구 실행을 tool.invoke(...) 형태로 통일
# - 입력 dict({"city": "서울"})을 함수 시그니처(get_weather(city=...))에 자동 매핑
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 알려줍니다."""
    # 데모용 가짜 데이터(실서비스라면 여기서 실제 API 호출/DB 조회 등이 들어갈 자리)
    fake_data = {"서울": "맑음, 18°C", "부산": "흐림, 22°C", "제주": "비, 20°C"}
    return fake_data.get(city, f"{city}의 날씨 정보 없음")


# ---------------------------------------------------------------------------
# 2) LLM 생성 + tools 바인딩
# ---------------------------------------------------------------------------
# ChatOpenAI는 OpenAI 채팅 모델을 LangChain의 Runnable 인터페이스로 감싼 래퍼입니다.
#
# model:
# - 예: "gpt-4o-mini"
# - os.environ["OPENAI_MODEL"]처럼 대괄호 접근은 키가 없으면 KeyError를 냅니다(교육용으론 명확).
#
# temperature=0:
# - 샘플링 랜덤성을 낮춰, 같은 입력에 대해 더 안정적인 출력을 기대할 수 있습니다.
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

# bind_tools([get_weather]):
# - "이 모델은 get_weather 도구를 사용할 수 있다"를 붙인 새 Runnable을 만듭니다.
# - 이 runnable은 답변 대신 tool_calls를 포함한 AIMessage를 반환할 수 있습니다.
llm_with_tools = llm.bind_tools([get_weather])


# ---------------------------------------------------------------------------
# 3) 도구 라우팅(이름 -> Tool 객체)
# ---------------------------------------------------------------------------
# LLM이 tool_calls로 {"name": "..."} 형태를 주기 때문에,
# 이름으로 실제 Tool을 찾을 수 있게 dict로 매핑해둡니다.
tools_by_name = {"get_weather": get_weather}


# ---------------------------------------------------------------------------
# === 1턴: 도구 호출 명령 받기 ===
# ---------------------------------------------------------------------------
# messages는 "대화 히스토리"입니다. 한 번 invoke할 때마다 지금까지의 messages를 통째로 넣어
# 문맥을 유지합니다.
messages = [HumanMessage(content="서울이랑 제주 날씨 둘 다 알려줘")]

# invoke(messages):
# - 입력: list[BaseMessage] (여기서는 HumanMessage만 1개)
# - 출력: AIMessage (assistant 역할 메시지)
#
# 이 response는 'dict'가 아니라 'AIMessage 인스턴스'이고, 그래서 response.content처럼 점 접근을 합니다.
response = llm_with_tools.invoke(messages)

# 아주 중요: 1턴의 AIMessage(response)를 messages에 누적해야 3턴에서 모델이 자기 tool_calls 맥락을 기억합니다.
messages.append(response)

# response.tool_calls:
# - 보통 list[dict]이며,
#   예시(개략):
#   [
#     {"name": "get_weather", "args": {"city": "서울"}, "id": "call_..."},
#     {"name": "get_weather", "args": {"city": "제주"}, "id": "call_..."},
#   ]
print(f"response.tool_calls: {response.tool_calls}")

print("📞 LLM의 도구 호출 요청:")
for tc in response.tool_calls:
    # tc는 dict라서 tc["name"], tc["args"] 형태로 접근합니다.
    print(f"  - {tc['name']}({tc['args']})")


# ---------------------------------------------------------------------------
# === 2턴: 실제로 도구 실행 ===
# ---------------------------------------------------------------------------
# 이제 LLM이 요청한 tool_calls를 실제 파이썬 코드로 실행하고, 그 결과를 ToolMessage로 messages에 쌓습니다.
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]  # 예: "get_weather"
    tool_args = tool_call["args"]  # 예: {"city": "서울"}
    tool_id = tool_call["id"]  # 예: "call_..." (LLM이 부여한 호출 ID)

    # tools_by_name[tool_name]은 Tool 객체(@tool로 감싸진 것)이고,
    # invoke(tool_args)는 tool_args dict를 시그니처에 맞춰 함수 호출로 변환해 실행합니다.
    # get_weather의 반환 타입이 str이므로 result는 str입니다.
    result = tools_by_name[tool_name].invoke(tool_args)
    print(f"⚙️  실행: {tool_name}({tool_args}) → {result}")

    # ToolMessage(content=..., tool_call_id=...):
    # - 도구 실행 결과를 메시지 히스토리에 추가하는 전용 메시지
    # - tool_call_id는 반드시 tool_call["id"]와 같아야 “이 결과가 어떤 호출의 결과인지”가 맞춰집니다.
    messages.append(ToolMessage(content=result, tool_call_id=tool_id))


# ---------------------------------------------------------------------------
# === 3턴: 결과 보고 LLM이 최종 답변 ===
# ---------------------------------------------------------------------------
# 이제 messages에는 다음이 순서대로 들어있습니다.
# 1) HumanMessage: "서울이랑 제주..."
# 2) AIMessage: tool_calls를 포함한 LLM의 도구 호출 지시
# 3) ToolMessage들: 각 tool_call_id에 매칭되는 실행 결과
#
# 이 상태로 다시 invoke하면, LLM은 ToolMessage 내용을 근거로 사람에게 줄 최종 답변을 생성합니다.
final_response = llm_with_tools.invoke(messages)

# final_response도 보통 AIMessage 인스턴스입니다.
# 이제는 tool_calls가 비어 있고, 자연어 답이 final_response.content에 들어가는 경우가 일반적입니다.
print(f"\n🤖 최종 답변: {final_response.content}")