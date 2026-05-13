# test_main_agent.py

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from assistant.main_agent import main_agent


def run(user_input: str):
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")
    
    result = main_agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )
    
    print("\n📜 메인 에이전트 메시지 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        tc_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_info = f" [tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}]"
        content = (msg.content or "")[:150]
        print(f"  [{i}] {msg_type}{tc_info}")
        if content:
            print(f"      content: {content}")
    
    print(f"\n🎯 최종 답변: {result['messages'][-1].content}")


if __name__ == "__main__":
    # 노트 비서 단독
    run("오늘 배운 거: Supervisor 패턴은 멀티 에이전트의 한 종류야. 노트해줘")
    
    # 검색 비서 단독
    run("MCP가 뭐야? 검색해서 알려줘")
    
    # 두 비서 협업 (★ 핵심 시나리오)
    run("LangChain 1.0 새 기능을 검색해서 노트로 저장해줘")
    
    # 직접 답 (서브 안 거침)
    run("안녕")
    
    # 노트 검색 vs 웹 검색 구분
    run("LangGraph 관련해서 적어둔 거 있어?")  # ← 노트 비서
    run("LangGraph가 뭐야?")                   # ← 검색 비서 (애매할 수 있음)