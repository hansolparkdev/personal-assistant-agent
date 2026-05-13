from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# OPENAI_MODEL=gpt-4o-mini
# 이렇게 env에 들어있음
# model을 env에서 가져오기
model = os.environ.get("OPENAI_MODEL")

# 설명
# temperature=0.0 : 출력의 예측 가능성을 최소화하여 예측 가능성이 높은 답변을 생성
# llm = ChatOpenAI(model=model, temperature=0.0)
llm_strict = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=1)
# Messages 객체 생성
# messages = [
#     SystemMessage(content="너는 친근한 개인 비서야. 한국어로 짧고 명확하게 답해. 한국어로 답변해줘."),
#     HumanMessage(content="안녕 너는 누구야?")
# ]

# 챗모델 호출
# reponse = llm.invoke(messages)

# content확인 및 reponse타입 확인
prompt = "너는 누군디?"
print("=== temperature=0 ===")
print(llm_strict.invoke(prompt).content)
print("\n=== temperature=1 ===")
print(llm_creative.invoke(prompt).content)
