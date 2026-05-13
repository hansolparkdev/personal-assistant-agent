from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from assistant.supervisor_graph import supervisor_graph

def run(user_input: str):
    print(f"\n{'=' * 60}")
    print(f"🧑 유저: {user_input}")
    print(f"{'=' * 60}")

    result = supervisor_graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )

    # print("\n📜 전체 메시지 흐름:")
    # for i, msg in enumerate(result["messages"]):
    #     msg_type = type(msg).__name__
    #     tc_info = ""
    #     if hasattr(msg, "tool_calls") and msg.tool_calls:
    #         tc_info = f" [tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}]"
    #     tcid_info = ""
    #     if hasattr(msg, "tool_call_id") and msg.tool_call_id:
    #         tcid_info = f" [tool_call_id: {msg.tool_call_id[:8]}]"
    #     content = (msg.content or "")[:120]
    #     print(f"  [{i}] {msg_type}{tc_info}{tcid_info}")
    #     if content:
    #         print(f"      content: {content}")

    print(f"\n최종 답변: {result['messages'][-1].content}")

if __name__ == "__main__":
    while True:
        user_input = input("🧑 유저: ")
        if user_input.lower() == "exit":
            break
        response = run(user_input)
        print(f"🤖 비서: {response}")