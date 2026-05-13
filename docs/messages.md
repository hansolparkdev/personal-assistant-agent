# Messages — LangChain 메시지 타입 완전 정리

LangChain · LangGraph 에이전트에서 LLM과 주고받는 모든 데이터는 **메시지(Message) 객체** 다.
이 프로젝트(`personal-assistant-agent`)도 전부 메시지로 동작한다.

이 문서를 다 읽으면 다음을 완벽히 이해할 수 있다:
- 4가지 메시지 타입(`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`)의 규격
- 각각 언제, 왜, 어떻게 쓰는지
- LLM 한 번의 호출에서 메시지들이 어떻게 흐르는지
- `tool_calls` ↔ `ToolMessage`의 1:1 매칭 규칙
- supervisor 그래프에서 메시지가 누적되는 실제 모습

---

## 1. 큰 그림 — 메시지는 "대화 한 줄" 이다

LLM은 stateless 다. 매 호출마다 **메시지 리스트 전체** 를 통째로 보낸다.
LangChain은 이 리스트의 각 항목을 역할(role)별로 클래스로 만들어둔 것뿐.

```
messages = [
    SystemMessage(content="너는 비서야"),       # ← role: "system"
    HumanMessage(content="MCP가 뭐야?"),        # ← role: "user"
    AIMessage(content="...", tool_calls=[...]), # ← role: "assistant"
    ToolMessage(content="검색결과", tool_call_id="..."),  # ← role: "tool"
    AIMessage(content="MCP는 ..."),            # ← role: "assistant" (최종 답변)
]

llm.invoke(messages)  # → 다음 AIMessage 한 개를 반환
```

이게 핵심이다. **메시지 리스트 = 대화 히스토리 = LLM의 "기억"**.

---

## 2. 4가지 메시지 타입 한눈에

| 타입 | role | 누가 만드나 | 언제 쓰나 |
|---|---|---|---|
| `SystemMessage` | system | 개발자가 작성 | 에이전트 페르소나 · 규칙 · 행동 지침 주입 (보통 맨 앞에 1개) |
| `HumanMessage` | user | 사용자 입력 | 사람의 질문, 또는 supervisor가 서브에 넘기는 쿼리 |
| `AIMessage` | assistant | LLM이 생성 | LLM의 응답. 텍스트 답변이거나 `tool_calls`(도구 호출 요청) |
| `ToolMessage` | tool | 코드(우리)가 생성 | 도구 실행 결과를 LLM에 돌려주는 메시지 (반드시 직전 `tool_calls`와 짝지어야 함) |

---

## 3. 각 메시지 타입 상세

### 3-1. `SystemMessage` — "너는 누구냐"

LLM에게 **정체성 · 행동 규칙**을 알려주는 메시지. 대화 맨 앞에 1번만 넣고, 이후 사용자 입력이 와도 보통 갱신하지 않는다.

**규격**:
```python
SystemMessage(content: str)
```

**언제 쓰나**:
- 에이전트의 역할 정의 ("너는 노트 비서야")
- 답변 톤 · 언어 ("한국어로 짧게")
- 도구 사용 규칙 ("X면 도구 A 써, Y면 도구 B 써")
- 출력 포맷 강제 ("JSON으로 답해")

**이 프로젝트에서**:
```python
# assistant/supervisor_graph.py:76
_SUPERVISOR_PROMPT = """너는 범용 비서를 총괄하는 supervisor야
너는 직접 답을 하지않아. 사용자 요청을 분석해서 적절한 서브 비서에게 위임해
...
"""

def supervisor_node(state):
    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *state["messages"]]
    response = _llm_with_handoff.invoke(messages)
```

> **왜 매번 앞에 붙이나?**: `state["messages"]`에는 시스템 프롬프트가 안 들어 있다. LangGraph는 보통 히스토리만 누적하고, 시스템 프롬프트는 노드에서 매번 prepend 한다. 그래야 프롬프트를 바꿔도 기존 대화에 영향이 없다.

> **`create_agent`를 쓸 때**: `create_agent(model, tools, system_prompt="...")`처럼 인자로 넘기면 내부에서 알아서 매 호출 앞에 붙여준다. (`assistant/agents/note_agent.py`, `search_agent.py`)

---

### 3-2. `HumanMessage` — "사람이 한 말"

**규격**:
```python
HumanMessage(content: str)
```

**언제 쓰나**:
- (당연) 사용자 입력
- **에이전트가 다른 에이전트에 일을 시킬 때**: supervisor → 서브에 쿼리를 넘길 때도 `HumanMessage`로 감싼다. 서브 입장에선 "누가 시켰든 사람이 시킨 것처럼" 보이는 게 가장 자연스럽기 때문.

**이 프로젝트에서**:

(1) 사용자 입력 시작점:
```python
# test_supervisor_graph.py:12
supervisor_graph.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    ...
)
```

(2) supervisor → 서브 에이전트로 위임:
```python
# assistant/supervisor_graph.py:142
result = note_agent.invoke({"messages": [HumanMessage(content=query)]})
```

> **왜 ToolMessage가 아니라 HumanMessage?**: 서브 에이전트는 자기 안에서 **독립된 새 대화**를 시작한다. supervisor의 `tool_calls` 컨텍스트는 모른다. 서브 입장에선 그냥 "사람이 질문한 것"으로 받아야 깔끔하다.

---

### 3-3. `AIMessage` — "LLM이 한 말"

가장 복잡하다. **두 가지 모드**가 있다.

**규격**:
```python
AIMessage(
    content: str,           # 텍스트 응답 (도구 호출 시에는 비어있을 수 있음)
    tool_calls: list[dict]  # 도구 호출 요청 (없으면 빈 리스트)
)
```

#### 모드 A — 최종 답변 (`tool_calls`가 비어있음)
```python
AIMessage(content="MCP는 ... 입니다.", tool_calls=[])
```
LLM이 "할 일 다 끝났고 이게 최종 답이다"라고 말하는 상태. 에이전트 루프가 종료된다.

#### 모드 B — 도구 호출 요청 (`tool_calls`가 있음)
```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "web_search",          # 호출할 도구 이름
            "args": {"query": "MCP"},      # 도구에 넘길 인자
            "id": "call_abc123",           # 이 호출의 고유 ID (ToolMessage가 매칭할 키)
        }
    ]
)
```
LLM이 "이 도구를 이런 인자로 실행해줘"라고 요청한 상태. 우리(코드)가 그 도구를 실제로 실행하고, 결과를 `ToolMessage`로 돌려줘야 한다.

> `tool_calls`는 **리스트** 다. 한 번에 여러 도구를 병렬 호출할 수도 있다. (이 프로젝트는 supervisor가 보통 1개씩 처리한다.)

**이 프로젝트에서**:

(1) supervisor가 라우팅 결정할 때:
```python
# supervisor_node가 출력
AIMessage(tool_calls=[
    {"name": "transfer_to_search_agent", "args": {"query": "MCP가 뭐야?"}, "id": "..."}
])
```

(2) 서브 에이전트가 실제 도구 호출:
```python
# search_agent 내부에서
AIMessage(tool_calls=[
    {"name": "web_search", "args": {"query": "MCP Model Context Protocol"}, "id": "..."}
])
```

(3) 최종 답변:
```python
# supervisor가 모든 위임 끝나고
AIMessage(content="MCP는 ... 입니다. 노트로 저장했어요.", tool_calls=[])
```

> **`bind_tools()`의 역할**: `_llm.bind_tools(handoff_tools)` 하면 LLM이 그 도구 스키마를 알게 되고, 필요할 때 `tool_calls`가 채워진 `AIMessage`를 반환할 수 있게 된다. `bind_tools` 안 했으면 LLM은 도구 호출을 할 수 없고 그냥 텍스트만 뱉는다.

---

### 3-4. `ToolMessage` — "도구가 돌려준 결과"

**규격**:
```python
ToolMessage(
    content: str,         # 도구 실행 결과 (LLM이 읽을 텍스트)
    tool_call_id: str,    # 어떤 tool_call에 대한 응답인지 (필수, AIMessage.tool_calls[i]["id"] 와 매칭)
)
```

**언제 쓰나**:
- 직전 `AIMessage`에 `tool_calls`가 있었고, 우리가 그 도구를 실행한 직후.
- **반드시 `tool_call_id`로 매칭**되어야 한다. 안 그러면 LLM이 어떤 호출에 대한 응답인지 모른다 (그리고 OpenAI API가 에러로 거부한다).

**규칙 — 절대 어기면 안 되는 것**:

1. `AIMessage`에 `tool_calls`가 N개 있으면 → 그 다음에 정확히 N개의 `ToolMessage`가 와야 한다.
2. 각 `ToolMessage.tool_call_id`는 `AIMessage.tool_calls[i]["id"]`와 1:1로 매칭되어야 한다.
3. `ToolMessage` 다음에는 다시 `AIMessage`가 와야 한다 (LLM이 결과를 보고 다음 액션을 결정).

**이 프로젝트에서**:

```python
# assistant/supervisor_graph.py:130 (note_node)
last_message = state["messages"][-1]
tool_call = next(
    tc for tc in last_message.tool_calls if tc["name"] == "transfer_to_note_agent"
)
query = tool_call["args"]["query"]
tool_call_id = tool_call["id"]     # ← 이 id를 반드시 기억

# 서브 에이전트 호출
result = note_agent.invoke({"messages": [HumanMessage(content=query)]})
response_content = result["messages"][-1].content

# ToolMessage로 supervisor에 반환 (id 매칭이 핵심)
return {
    "messages": [
        ToolMessage(content=response_content, tool_call_id=tool_call_id)
    ]
}
```

> **왜 `note_node`가 직접 `ToolMessage`를 만드나?**: supervisor가 `transfer_to_note_agent`라는 도구를 호출한 셈이 되어버렸기 때문(`tool_calls`에 있음). 그 호출에 응답을 안 주면 supervisor의 대화 히스토리가 깨진다. 그래서 노드는 결과를 받아서 `ToolMessage`로 supervisor에 "응답"해야 한다.

---

## 4. LLM이 한 번 호출될 때 보는 메시지의 흐름

LangGraph가 알아서 누적해주지만, 내부에서 어떻게 쌓이는지 정확히 알아야 디버깅이 된다.

### 시나리오: "MCP가 뭐야?"

#### 턴 0 — 사용자 입력만
```python
state["messages"] = [
    HumanMessage("MCP가 뭐야?"),
]
```

#### 턴 1 — supervisor 노드 실행
LLM이 본 것 (시스템 프롬프트는 노드가 prepend):
```python
[
    SystemMessage(_SUPERVISOR_PROMPT),
    HumanMessage("MCP가 뭐야?"),
]
```
LLM 출력:
```python
AIMessage(tool_calls=[
    {"name": "transfer_to_search_agent", "args": {"query": "MCP"}, "id": "call_001"}
])
```
이게 state에 append됨:
```python
state["messages"] = [
    HumanMessage("MCP가 뭐야?"),
    AIMessage(tool_calls=[{"name": "transfer_to_search_agent", "id": "call_001", ...}]),
]
```

#### 턴 2 — search_node 실행
- supervisor의 tool_call에서 query 추출
- 서브 에이전트 호출 (이건 별도 sub-state, 안에서 또 다른 대화)
- 결과를 받아 `ToolMessage`로 변환 (id 매칭!):
```python
ToolMessage(content="MCP는 ...", tool_call_id="call_001")
```
state는:
```python
state["messages"] = [
    HumanMessage("MCP가 뭐야?"),
    AIMessage(tool_calls=[{"id": "call_001", ...}]),
    ToolMessage(content="MCP는 ...", tool_call_id="call_001"),  # ← id 매칭됨
]
```

#### 턴 3 — supervisor 재진입
LLM이 본 것:
```python
[
    SystemMessage(_SUPERVISOR_PROMPT),
    HumanMessage("MCP가 뭐야?"),
    AIMessage(tool_calls=[{"id": "call_001", ...}]),
    ToolMessage(content="MCP는 ...", tool_call_id="call_001"),
]
```
LLM 출력 (도구 호출 더 안 함):
```python
AIMessage(content="MCP는 ... 입니다.", tool_calls=[])
```
→ `route_after_supervisor`가 `"end"` 반환 → 사용자에게 최종 답변.

---

## 5. 메시지 흐름 다이어그램

```
   사용자                                              LLM
     │                                                  │
     │ "MCP가 뭐야?"                                     │
     │ ───────► HumanMessage ─┐                          │
     │                        │                          │
     │                        │  [SystemMessage,         │
     │                        │   HumanMessage]          │
     │                        │ ───────────────────────► │
     │                        │                          │ (라우팅 결정)
     │                        │ ◄─── AIMessage(tool_calls=[search, id="A"])
     │                        │                          │
     │           ┌────────────┘                          │
     │           ▼                                       │
     │     [search_node]                                 │
     │      도구 실행 (web_search)                       │
     │      결과 → ToolMessage(tool_call_id="A")          │
     │           │                                       │
     │           │  [Sys, Hum, AI(tool_calls), Tool]     │
     │           │ ─────────────────────────────────────►│
     │                                                  │ (이제 답할 차례)
     │           ◄────────── AIMessage(content="MCP는...")│
     │                                                  │
     │ ◄────── 최종 답변                                  │
```

---

## 6. 자주 발생하는 실수와 해결

### ❌ 실수 1: `ToolMessage` 빼먹기
```python
# 잘못된 예
state["messages"] = [
    HumanMessage("..."),
    AIMessage(tool_calls=[{"id": "X", ...}]),
    AIMessage(content="..."),  # ❌ ToolMessage 없이 바로 AIMessage
]
```
**에러**: `Tool call X has no corresponding tool result.`
**해결**: `tool_calls`가 있는 `AIMessage` 뒤엔 반드시 짝맞는 `ToolMessage`.

### ❌ 실수 2: `tool_call_id` 누락 또는 불일치
```python
ToolMessage(content="결과")  # ❌ tool_call_id 없음
ToolMessage(content="결과", tool_call_id="wrong_id")  # ❌ 매칭 안됨
```
**해결**: 항상 직전 `AIMessage.tool_calls[i]["id"]`를 그대로 복사해서 넣기.

### ❌ 실수 3: `SystemMessage`를 사용자 메시지 중간에 끼워넣기
```python
[HumanMessage("..."), SystemMessage("새 규칙"), HumanMessage("...")]  # ❌
```
일부 모델은 받아주지만 동작이 일관되지 않는다.
**해결**: `SystemMessage`는 무조건 맨 앞에 1개. 규칙을 바꾸고 싶으면 새 노드에서 시스템 프롬프트만 갈아끼우고 다시 prepend.

### ❌ 실수 4: 서브 에이전트에 supervisor의 히스토리 통째로 전달
```python
note_agent.invoke({"messages": state["messages"]})  # ❌ tool_calls 매칭 깨짐
```
서브에는 `tool_call_id`에 대응하는 `AIMessage`가 없으니 에러.
**해결**: 이 프로젝트처럼 `query`만 추출해 `HumanMessage`로 감싸서 넘기기.

### ❌ 실수 5: `AIMessage(content="...", tool_calls=[...])` 둘 다 채우기
가능은 하다. LLM이 "도구 호출하면서 동시에 사용자에게 코멘트"하는 경우. 다만 OpenAI/Anthropic은 둘 다 있는 응답을 잘 안 만든다. 보통은 둘 중 하나만 채워진다.

---

## 7. 빠르게 외워둘 체크리스트

```
✅ 매 LLM 호출은 SystemMessage 1개로 시작 (노드에서 prepend)
✅ HumanMessage는 진짜 사람 입력이거나, 에이전트 간 위임 시
✅ AIMessage에 tool_calls가 있으면 → 다음 메시지는 반드시 ToolMessage
✅ ToolMessage.tool_call_id는 직전 AIMessage.tool_calls[i]["id"]와 정확히 일치
✅ 서브 에이전트로 위임할 땐 query만 뽑아서 HumanMessage로 새로 시작
✅ 노드 반환값은 dict, "messages" 키에 추가할 메시지(들)의 리스트
```

---

## 8. 이 프로젝트의 메시지 흐름 한 줄 요약

> **사용자 `HumanMessage`** → supervisor가 `AIMessage(tool_calls=[transfer_*])` 생성 → 노드가 서브 에이전트를 새 `HumanMessage`로 호출 → 서브가 내부에서 `web_search`/`save_note` 도구 돌리며 자기 `AIMessage`/`ToolMessage` 사이클 수행 → 서브의 최종 텍스트를 `ToolMessage(tool_call_id=...)`로 supervisor에 반환 → supervisor가 더 일할 게 없으면 `AIMessage(content="...", tool_calls=[])` 으로 종료.

이 한 문장이 머릿속에 그려지면 이 프로젝트는 완벽히 이해한 거다.

---

## 9. 더 깊이 — 실제 라이브러리 위치

- `langchain_core.messages`
  - `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`
  - 모두 `BaseMessage`를 상속. `.content`, `.type`(role), 그리고 각자 추가 필드.
- `langgraph.graph.MessagesState`
  - `{"messages": list[BaseMessage]}` 형태의 state 스키마.
  - **append 동작이 자동**: 노드가 `{"messages": [...]}` 반환하면 기존 리스트에 append (reducer가 처리).
  - 그래서 노드는 "추가할 메시지만" 반환하면 된다 (기존 히스토리는 안 건드림).
