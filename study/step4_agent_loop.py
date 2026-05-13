# step4_agent_loop.py
#
# ---------------------------------------------------------------------------
# Step 4: 자동 루프 (The Agent Loop, Tool-Calling Agent의 기본 패턴)
# ---------------------------------------------------------------------------
# 이 파일은 "한 번 호출하고 끝"이 아니라, LLM이 스스로 필요할 때 도구를 부르고
# 도구 결과를 다시 읽어 최종 답을 만들 때까지 반복하는 최소 에이전트 루프 예제입니다.
#
# Step 3-2와의 차이
# - Step 3-2: 도구 호출 → 실행 → 최종 답변을 "고정된 턴 수"로 수동 작성
# - Step 4:   "종료 조건"만 정의하고, 나머지는 루프가 자동 수행
#
# 핵심 종료 조건
# - response.tool_calls가 비어 있다면:
#   LLM이 더 이상 도구가 필요 없다고 판단했고, response.content를 최종 답변으로 볼 수 있습니다.
# ---------------------------------------------------------------------------

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()
# .env에서 OPENAI_API_KEY, OPENAI_MODEL 등을 로드해 os.environ에서 읽을 수 있게 합니다.


# ---------------------------------------------------------------------------
# 1) 도구 정의: LLM이 호출할 "외부 능력" 선언
# ---------------------------------------------------------------------------
# @tool 데코레이터를 붙이면 함수가 LangChain Tool로 래핑됩니다.
# 모델은 함수 이름(name), 시그니처(args), 설명(docstring)을 보고 언제 호출할지 결정합니다.
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 알려줍니다."""
    # 데모 목적의 가짜 데이터 소스
    # 실전에서는 API/DB 조회로 대체할 수 있습니다.
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
    # 데모용 규칙 기반 로직:
    # - 날씨 문자열을 보고 간단한 키워드 매칭으로 답변 생성
    # - 실전에서는 규칙 엔진/별도 모델/벡터 검색 등으로 고도화 가능
    if "비" in weather_description:
        return "우산과 방수 자켓을 챙기세요."
    if "20" in weather_description or "18" in weather_description:
        return "얇은 가디건이나 긴팔 셔츠가 적당합니다."
    if "24" in weather_description or "22" in weather_description:
        return "반팔 티셔츠가 좋습니다."
    return "오늘 날씨에 맞는 편한 복장을 추천합니다."


# ---------------------------------------------------------------------------
# 2) LLM + 도구 바인딩: "모델이 어떤 도구를 쓸 수 있는지" 등록
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)
# ChatOpenAI.invoke(...) 반환은 보통 AIMessage 객체입니다.
# (dict가 아니라 객체이므로 response.content, response.tool_calls처럼 점 접근)

tools = [get_weather, recommend_clothing]
llm_with_tools = llm.bind_tools(tools)
# bind_tools 후에는 모델이 tool_calls를 생성할 수 있습니다.
# tool_calls는 보통 list[dict] 형태이고 각 dict에 name/args/id가 포함됩니다.

# 이름 → Tool 객체 매핑 (라우팅 테이블)
tools_by_name = {t.name: t for t in tools}
# LLM이 tool_call에서 문자열 name을 주면,
# 이 테이블로 실제 Tool 객체를 찾아 invoke(args) 합니다.


# ---------------------------------------------------------------------------
# 3) 에이전트 루프(핵심 패턴)
# ---------------------------------------------------------------------------
def run_agent(user_input: str, max_iterations: int = 10) -> str:
    """
    LLM이 더 이상 도구를 호출하지 않을 때까지 자동 반복 실행합니다.

    루프 불변식(매 반복마다 유지해야 하는 약속):
    - messages에는 지금까지의 대화/도구 결과가 순서대로 누적되어 있어야 한다.
    - LLM의 tool_call에 대해 반드시 대응 ToolMessage를 넣어줘야 다음 추론이 정상 동작한다.

    반복 흐름:
      A) llm_with_tools.invoke(messages)로 AIMessage(response) 받기
      B) response.tool_calls가 비어 있으면 종료(최종 답변)
      C) tool_calls가 있으면 각 호출을 실행하고 ToolMessage로 messages에 추가
      D) 다시 A로
    """
    # 대화 시작 상태: 사용자 입력 1개만 담긴 히스토리
    messages = [HumanMessage(content=user_input)]

    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    # max_iterations는 안전장치:
    # 모델이 비정상적으로 계속 도구를 부르는 무한 루프 상황을 강제 종료하기 위함입니다.
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 🔁 Iteration {iteration} ---")

        # -------------------------------------------------------------------
        # A) LLM 호출 단계
        # -------------------------------------------------------------------
        # messages 전체를 모델에 전달해 "다음 행동"을 받습니다.
        # response는 AIMessage이며, 여기엔 자연어 content 또는 tool_calls가 들어옵니다.
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        # response 자체도 히스토리에 저장해야 다음 반복에서 모델이 자기 이전 판단을 기억합니다.

        # -------------------------------------------------------------------
        # B) 종료 판정 단계
        # -------------------------------------------------------------------
        # tool_calls가 없다는 뜻:
        # - 모델이 더 이상 외부 도구를 쓸 필요가 없다고 판단
        # - 지금 content를 사용자에게 반환하면 됨
        if not response.tool_calls:
            print(f"\n🤖 최종 답변: {response.content}")
            return response.content

        # -------------------------------------------------------------------
        # C) 도구 실행 단계
        # -------------------------------------------------------------------
        # response.tool_calls는 list[dict] 형태.
        # 각 항목은 대체로 {"name": ..., "args": {...}, "id": ...} 구조입니다.
        print(f"📞 LLM이 {len(response.tool_calls)}개 도구 호출 요청")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # 안전장치: 모르는 도구를 LLM이 부르는 경우 (드물지만 가능)
            if tool_name not in tools_by_name:
                error_msg = f"알 수 없는 도구: {tool_name}"
                print(f"  ⚠️  {error_msg}")
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                # 에러도 ToolMessage로 되돌려줘야 모델이 상황을 인지하고 복구/재시도 추론 가능
                continue

            result = tools_by_name[tool_name].invoke(tool_args)
            # Tool.invoke(args):
            # - args(dict)를 함수 파라미터로 매핑해 실제 함수 실행
            # - 반환값(result)은 문자열/JSON 직렬화 가능한 값이 일반적
            print(f"  ⚙️  {tool_name}({tool_args}) → {result}")

            messages.append(ToolMessage(content=result, tool_call_id=tool_id))
            # 핵심: tool_call_id를 동일하게 넣어 호출-결과를 매칭
            # 이 연결이 깨지면 모델이 어떤 결과가 어떤 호출의 결과인지 이해하기 어려워집니다.

    # -----------------------------------------------------------------------
    # D) 안전 종료 단계
    # -----------------------------------------------------------------------
    # 여기까지 왔다는 것은 max_iterations를 초과했다는 의미.
    # 실전에서는 이 시점에 로그 저장, 알림, fallback 응답 등을 추가하는 것이 좋습니다.
    print(f"\n⚠️  최대 반복 횟수({max_iterations}) 초과")
    return "에이전트가 작업을 완료하지 못했습니다."


# ---------------------------------------------------------------------------
# 4) 테스트 시나리오: 패턴 검증용 샘플 프롬프트
# ---------------------------------------------------------------------------
# 시나리오를 난이도별로 구성했습니다.
# 1 -> 단일 호출, 2 -> 동일 라운드 복수 호출, 3 -> 순차 추론, 4 -> 복합 추론
if __name__ == "__main__":
    # 시나리오 1: 단순 - 도구 1번, 루프 2회 (호출 + 답변)
    run_agent("서울 날씨 알려줘")

    # 시나리오 2: 병렬 - 한 라운드에서 도구 2개 동시 호출
    run_agent("서울이랑 부산 날씨 알려줘")

    # 시나리오 3: 순차 추론 - 첫 결과 보고 다음 도구 결정 (라운드 2번 필요)
    run_agent("서울 날씨 확인하고 거기 맞는 옷차림 추천해줘")

    # 시나리오 4: 복합 - 병렬 + 순차 추론
    run_agent("서울이랑 도쿄 날씨 비교해서 더 따뜻한 도시 옷차림 추천해줘")


# ---------------------------------------------------------------------------
# [학습 메모] 이 파일에서 반드시 숙지할 에이전트 루프 패턴
# ---------------------------------------------------------------------------
# 1) "메시지 누적"이 상태 관리다
#    - 별도 상태 객체가 없어도 messages 히스토리가 곧 컨텍스트/메모리 역할을 합니다.
#
# 2) 종료 조건은 "도구 호출 없음"으로 단순화
#    - response.tool_calls 유무 하나로 제어 흐름이 명확해집니다.
#
# 3) 도구 실행 결과는 반드시 ToolMessage로 되돌린다
#    - tool_call_id 매칭까지 지켜야 모델이 다음 추론을 정확히 수행합니다.
#
# 4) 라우팅 테이블(tools_by_name)은 필수
#    - 모델이 문자열 name을 주기 때문에 실제 실행 가능한 Tool로 변환하는 브리지입니다.
#
# 5) 안전장치(max_iterations, unknown tool 처리)는 기본값으로 넣는다
#    - 에이전트 코드는 "예외 케이스가 정상"이라는 전제로 방어 코드를 먼저 설계해야 안정적입니다.
#
# 6) 확장 방향
#    - 도구 수 증가: tools 리스트와 라우팅 테이블만 확장
#    - 신뢰성 강화: 재시도 정책, 타임아웃, 로깅(입출력/토큰/지연시간) 추가
#    - 품질 개선: system prompt 설계(언제 어떤 도구를 쓸지 정책 명확화)
# ---------------------------------------------------------------------------