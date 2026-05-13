# step3_tool.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. 도구 정의 (그냥 함수 + @tool 데코레이터)
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 알려줍니다."""
    # 실제로는 API 호출, 지금은 가짜 데이터
    fake_data = {
        "서울": "맑음, 18°C",
        "부산": "흐림, 22°C",
        "제주": "비, 20°C",
    }
    return fake_data.get(city, f"{city}의 날씨 정보 없음")


# 2. LLM에 도구 바인딩
llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)
llm_with_tools = llm.bind_tools([get_weather])  # ← 도구 등록

# 3. 호출
response = llm_with_tools.invoke([
    HumanMessage(content="서울 날씨 어때?")
])

print("응답 객체:", response)
print("---")
print("내용:", response.content)
print("도구 호출:", response.tool_calls)