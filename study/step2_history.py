import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = os.environ.get("OPENAI_MODEL")
llm = ChatOpenAI(model=model, temperature=0)

history = [
    SystemMessage(content="너는 친근한 개인 비서야. 한국어로 짧고 명확하게 답해. 한국어로 답변해줘."),
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break

    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))
    print("Assistant: ", response.content)