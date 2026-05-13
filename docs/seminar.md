# Personal Assistant Agent 세미나

> LangChain 1.0 / LangGraph로 만드는 멀티에이전트
> 흐름: **도구 → 서브에이전트 → 메인에이전트 → 위임**

---

## 0. 오늘 다룰 것

박한솔 개인 비서 에이전트를 만들면서 다음을 배웁니다.

1. **도구(Tool)** 만드는 법 — `@tool` 데코레이터
2. **서브에이전트** 만드는 법 — `create_agent` 사용법 (노트 / 검색 2종)
3. **메인 에이전트(Supervisor)** 만드는 법 — 서브를 도구로 wrapping
4. `create_agent`가 내부적으로 만드는 **LangGraph 그래프** 이해
5. **서브가 2개 이상**일 때의 라우팅과 순차 호출

> 파이썬을 잘 모르는 분들도 있으니, 필요한 문법(데코레이터, 타입힌트, import)은 1챕터에서 짧게 짚고 갑니다.

### 전체 그림

```
사용자
  │
  ▼
┌──────────────┐
│ main_agent   │   Supervisor — 어느 서브에게 보낼지 결정
│ (Supervisor) │
└──┬─────────┬─┘
   │         │
   │         └──── call_search_agent(query) ──┐
   │                                          ▼
   │                                  ┌──────────────┐
   │                                  │ search_agent │   웹 검색 전문
   │                                  └──────┬───────┘
   │                                         │
   │                                         ▼
   │                                    web_search       (ddgs)
   │
   └──── call_note_agent(query) ────────────┐
                                            ▼
                                    ┌──────────────┐
                                    │  note_agent  │   노트 전문
                                    └──────┬───────┘
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                     save_note      search_notes     list_today_notes
                          │                │                │
                          └────────── data/notes.json ──────┘
```

### 프로젝트 구조

```
personal-assistant-agent/
├── assistant/
│   ├── main_agent.py             # 메인 에이전트 (Supervisor)
│   └── agents/
│       ├── note_agent.py         # 노트 서브에이전트
│       └── search_agent.py       # 검색 서브에이전트
├── tools/
│   ├── note_tools.py             # 노트 도구 3개 (save/search/list)
│   └── search_tools.py           # 검색 도구 1개 (web_search)
├── data/notes.json               # 노트 저장소
└── test_note_agent.py            # 실행 테스트
```

---

## 1. 파이썬 사전지식

### 1.1 데코레이터 `@`

```python
@tool
def save_note(...):
    ...
```

`@tool`은 **함수를 감싸서 변형**합니다. 위 코드는 아래와 같습니다.

```python
def save_note(...):
    ...
save_note = tool(save_note)   # 똑같은 의미
```

→ `@tool`이 붙으면 평범한 함수가 "LLM이 호출할 수 있는 Tool 객체"로 바뀝니다.

### 1.2 타입힌트 + Docstring

```python
def save_note(title: str, content: str, tags: list[str] = None) -> str:
    """학습 내용을 노트로 저장합니다."""
```

- `title: str` → title은 문자열이라는 표시
- `-> str` → 반환값이 문자열이라는 표시
- `"""..."""` → 함수 설명 (docstring)

**중요**: LangChain은 이 타입힌트와 docstring을 읽어서 **LLM에게 도구 사용법을 자동으로 알려줍니다.** 그래서 docstring을 잘 써야 LLM이 도구를 정확히 부릅니다.

### 1.3 import 경로

```python
from tools.note_tools import note_tools
from assistant.agents.note_agent import note_agent
```

→ 디렉토리 구조와 import 경로가 1:1로 매칭됩니다.

---

## 2. Layer 1 — 도구 만들기

### 2.1 핵심 컨셉

> **도구 = 데코레이터 `@tool` + 타입힌트 + docstring이 붙은 일반 파이썬 함수**

LLM은 함수 안의 코드를 읽지 못합니다. 대신 **함수 시그니처와 docstring**을 보고 "이 도구는 뭘 하는지", "어떤 인자를 줘야 하는지" 판단합니다.

### 2.2 노트 도구 — `tools/note_tools.py`

```python
from langchain_core.tools import tool

@tool
def save_note(title: str, content: str, tags: list[str] = None) -> str:
    """학습 내용, 메모, 일기 등을 노트로 저장합니다.

    Args:
        title: 노트 제목 (짧고 명확하게).
        content: 노트 본문.
        tags: 태그 리스트 (예: ['langchain', 'agent']). 선택사항.

    Returns:
        저장 결과 메시지.
    """
    notes = _load_notes()
    new_note = {
        "id": f"note_{len(notes) + 1:04d}",
        "title": title,
        "content": content,
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
    }
    notes.append(new_note)
    _save_notes(notes)
    return f"노트 저장 완료. ID: {new_note['id']}, 제목: '{title}'"

# 마지막에 모아서 export
note_tools = [save_note, search_notes, list_today_notes]
```

| 도구 | 역할 |
|---|---|
| `save_note` | 제목/본문/태그를 JSON에 저장 |
| `search_notes` | 키워드로 노트 검색 |
| `list_today_notes` | 오늘 작성한 노트 전체 조회 |

### 2.3 검색 도구 — `tools/search_tools.py`

웹 검색은 외부 라이브러리(`ddgs` — DuckDuckGo Search)를 wrapping한 단일 도구입니다.

```python
from ddgs import DDGS
from langchain_core.tools import tool

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색합니다. 최신정보, 정의, 뉴스, 기술 자료 등

    이 도구를 사용해야 하는 경우:
    - 사용자가 모르는 개념/용어를 설명해야할 때
    - 최신 정보가 필요한 질문 (라이브러리 버전, 뉴스 등)
    - 노트에 저장되지 않은 정보를 찾아야할 때

    Args:
        query: 검색 키워드.
        max_results: 최대 결과 수 (기본 5, 최대 10).

    Returns:
        검색 결과 요약. 제목 + 요약 + URL.
    """
    max_results = min(max_results, 10)
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
    except Exception as e:
        return f"검색 중 오류 발생: {e}"

    if not results:
        return f"'{query}'에 대한 검색 결과가 없습니다."

    lines = [f"'{query}' 검색 결과 ({len(results)}개):"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '제목 없음')}")
        lines.append(f"  요약: {r.get('body', '요약 없음')}")
        lines.append(f"  URL: {r.get('url', 'URL 없음')}")
    return "\n".join(lines)

search_tools = [web_search]
```

### 2.4 도구 개발 패턴 정리

- **`@tool` 한 줄**이 함수를 LLM 호출 가능한 도구로 변환
- **docstring이 LLM의 사용 설명서** — "언제 써야 하는지"를 명확히
- 반환값은 **사람이 읽을 수 있는 문자열** — LLM이 그대로 다음 사고에 사용
- 외부 의존(ddg, datetime, json 등)은 도구 내부에 격리 → 에이전트 코드는 깨끗하게 유지
- 파일 끝에 `xxx_tools = [...]` 리스트로 묶어 export

### 2.5 모듈화 방안

도구는 `tools/` 폴더에 **도메인별 파일**로 모읍니다.

```
tools/
├── note_tools.py      # note_tools = [save_note, search_notes, list_today_notes]
├── search_tools.py    # search_tools = [web_search]
├── calendar_tools.py  # (앞으로) calendar_tools = [...]
└── mail_tools.py      # (앞으로) mail_tools = [...]
```

각 파일은 **그 도메인 외부 라이브러리도 자기 안에서 책임**집니다. 검색 도메인이 ddg를 쓰는 걸 노트 도메인은 알 필요가 없음.

---

## 3. Layer 2 — 서브에이전트

### 3.1 핵심 컨셉

> **서브에이전트 = 도구 묶음 + LLM + 도메인 전문 프롬프트**

`create_agent` 함수 한 줄이면 ReAct 패턴 에이전트가 완성됩니다.

### 3.2 노트 에이전트 — `assistant/agents/note_agent.py`

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools.note_tools import note_tools

_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

_SYSTEM_PROMPT = """너는 학습 노트 전문 비서야. 사용자의 학습 내용/메모/일기를 저장하고 검색해.

원칙:
- 사용자가 뭔가를 "기록하고 싶다", "정리해줘", "저장해줘" 라고 하면 → save_note 사용
- "찾아줘", "어디 적었지", "전에 뭐라고 했지" 같은 검색은 → search_notes 사용
- "오늘 뭐 배웠지", "오늘 적은 거" 같은 일일 조회는 → list_today_notes 사용
- 저장할 때 제목은 짧고 명확하게, 태그는 1~3개 정도로 자동 추출해
- 한국어로 답해. 기술 용어는 영어 그대로.
"""

note_agent = create_agent(
    model=_llm,
    tools=note_tools,
    system_prompt=_SYSTEM_PROMPT,
)
```

### 3.3 검색 에이전트 — `assistant/agents/search_agent.py`

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools.search_tools import search_tools

_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

_SYSTEM_PROMPT = """너는 웹 검색 전문 비서야. 사용자의 질문에 최신 정보로 답해.

원칙:
- 모르는 개념이나 최신 정보 질문이면 무조건 web_search 사용
- 검색 결과를 그대로 나열하지 말고, 질문에 맞게 정리해서 답해
- 검색 결과 중 신뢰할 만한 것 (공식 문서, 위키피디아, 큰회사 블로그) 우선
- 너무 길지 않게 핵심만 짧게
- 한국어로 답해. 기술 용어는 영어 그대로.
- 답에 출처 URL 1~2개 포함해
"""

search_agent = create_agent(
    model=_llm,
    tools=search_tools,
    system_prompt=_SYSTEM_PROMPT,
)
```

### 3.4 `create_agent` 사용법

세 가지 인자가 핵심입니다.

| 인자 | 역할 |
|---|---|
| `model` | LLM 인스턴스 (`ChatOpenAI` 등) |
| `tools` | `@tool` 함수들의 리스트 |
| `system_prompt` | 에이전트의 성격/역할/규칙 |

추가로 `temperature=0`: 답변의 무작위성을 0으로. 도구 호출 안정성을 위해 0 권장.

### 3.5 프롬프트 작성 팁 — 두 에이전트 비교

**공통 구조**:
1. "너는 ~~ 전문 비서야" (역할 정의)
2. "원칙:" (어느 도구를 언제 쓸지 규칙)
3. 답변 형식 (언어, 길이, 출처 등)

**노트 에이전트**: 도구가 3개라서 **"어떤 표현이면 어떤 도구"** 매핑이 핵심.
```
기록/저장/메모 → save_note
찾아줘/검색 → search_notes
오늘 뭐 적었지 → list_today_notes
```

**검색 에이전트**: 도구가 1개라서 도구 선택이 아닌 **"결과 가공 방식"** 이 핵심.
```
검색 결과를 그대로 나열하지 말고 정리
신뢰할 만한 출처 우선
출처 URL 1~2개 포함
```

→ **도메인마다 프롬프트 패턴이 다릅니다.** 도구가 많은 에이전트는 "선택 규칙", 도구가 적은 에이전트는 "결과 가공"에 무게.

---

## 4. `create_agent` 내부 — LangGraph 그래프

### 4.1 핵심 포인트

> **`create_agent`는 LangChain 1.0 API지만, 내부는 LangGraph 그래프다.**

`note_agent.invoke(...)`를 호출하면 사실 LangGraph 상태 그래프가 돕니다.

### 4.2 그래프 구조

```
       ┌───────┐
       │ START │
       └───┬───┘
           ▼
       ┌────────┐    tool_calls 있음    ┌────────┐
       │ agent  │ ───────────────────▶ │ tools  │
       │ (LLM)  │ ◀─────────────────── │ (실행)  │
       └───┬────┘    도구 결과            └────────┘
           │
           │ tool_calls 없음 (최종 답변)
           ▼
        ┌─────┐
        │ END │
        └─────┘
```

### 4.3 한 턴 동안 일어나는 일

1. **agent 노드**: LLM이 메시지 히스토리를 보고 응답 생성
   - 도구 호출 필요 → `AIMessage`에 `tool_calls` 붙음
   - 그냥 답변 → `tool_calls` 비어있음
2. **분기**:
   - `tool_calls` 있음 → `tools` 노드로
   - `tool_calls` 없음 → `END`로
3. **tools 노드**: 호출된 도구를 실제로 실행 → 결과를 `ToolMessage`로 메시지에 추가
4. 다시 **agent 노드**로 (루프)

### 4.4 실제 메시지 흐름 예시 (노트 에이전트)

```
[0] HumanMessage  : "create_agent 학습 내용 노트해줘"
[1] AIMessage     : tool_calls=[save_note(title="...", content="...", tags=[...])]
[2] ToolMessage   : "노트 저장 완료. ID: note_0001"
[3] AIMessage     : "저장했어요! 제목은 ..."
```

도구 호출이 끝나야 그래프가 종료됩니다. 그래서 `recursion_limit=25`로 무한 루프 방지.

### 4.5 그래프 시각화

```python
print(note_agent.get_graph().draw_ascii())
print(search_agent.get_graph().draw_ascii())
print(main_agent.get_graph().draw_ascii())
```

→ 세 에이전트 모두 동일한 ReAct 토폴로지(`agent ↔ tools`)임을 눈으로 확인.

---

## 5. Layer 3 — 메인 에이전트 (Supervisor)

### 5.1 핵심 트릭

> **서브에이전트를 다시 `@tool`로 감싸서 메인의 "도구"로 등록한다.**

메인 LLM 입장에선 `call_note_agent`, `call_search_agent`는 그냥 함수 둘일 뿐. 안에서 또 다른 에이전트가 도는 건 모릅니다.

### 5.2 서브 2개를 도구로 wrapping — `assistant/main_agent.py`

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from assistant.agents.note_agent import note_agent
from assistant.agents.search_agent import search_agent

@tool
def call_note_agent(query: str) -> str:
    """학습 내용, 메모, 일기 등 **노트** 관련 작업을 노트 비서에게 위임합니다.

    이 도구를 사용해야 하는 경우:
    - 사용자가 뭔가를 "기록/저장/메모/노트"하고 싶다고 할 때
    - "어디 적었지", "전에 뭐라고 했지", "오늘 뭐 배웠지" 같은 검색/조회
    - 학습 내용 정리, 일기, 메모 관련 모든 요청

    이 도구를 사용하면 안 되는 경우:
    - 웹에서 정보를 찾아야 할 때 (그건 call_search_agent)

    Args:
        query: 노트 비서에게 전달할 자연어 요청.
    """
    result = note_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content   # 서브의 최종 답변만 반환


@tool
def call_search_agent(query: str) -> str:
    """**웹 검색** 관련 작업을 검색 비서에게 위임합니다.

    이 도구를 사용해야 하는 경우:
    - 사용자가 모르는 개념/용어를 설명해야할 때
    - 최신 정보가 필요한 질문 (라이브러리 버전, 뉴스 등)
    - 노트에 저장되지 않은 정보를 찾아야할 때

    Args:
        query: 검색 비서에게 전달할 자연어 요청.
            예: "MCP가 뭐야?", "LangChain 새 기능 알려줘"
    """
    result = search_agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content
```

### 5.3 wrapper 작성의 두 가지 포인트

1. **"사용해야 하는 경우"** 만 적지 말고 **"사용하면 안 되는 경우"** 도 적기
   - 도구가 늘어나면 LLM이 헷갈리기 시작함 → 도구끼리 경계를 명시해야 라우팅 정확도가 올라감
   - 예: `call_note_agent`의 docstring에 "웹에서 정보 찾을 땐 call_search_agent 쓰라"고 못박음
2. **`query` 인자는 자연어 그대로 넘김**
   - 서브에이전트가 알아서 ReAct loop 돌리므로, 메인은 "통째로 위임"만 하면 됨

### 5.4 메인 에이전트 생성

```python
_SYSTEM_PROMPT = """너는 박한솔의 범용 비서야. 다양한 전문 서브 비서들을 갖고 있어.

원칙:
- 사용자 요청을 분석해서 어느 서브 비서에게 위임할지 결정해.
- 너가 직접 답하기보다는, 가능하면 서브 비서를 활용해.
- 여러 서브가 필요한 작업이면 순차/병렬로 호출.
  예: "MCP 검색해서 노트로 저장" → search → note
- 서브 비서가 일을 끝내면 그 결과를 자연스럽게 사용자에게 전달.
- 단순 인사나 일상 대화는 직접 답해도 됨.
- 한국어로 답해. 짧고 직설적으로.

현재 사용 가능한 서브 비서:
- 노트 비서 (call_note_agent): 학습 내용/메모/일기 저장 및 검색
- 검색 비서 (call_search_agent): 웹 검색으로 최신 정보 조회
"""

main_agent = create_agent(
    model=_llm,
    tools=[call_note_agent, call_search_agent],   # ← 서브 추가 시 여기에 늘림
    system_prompt=_SYSTEM_PROMPT,
)
```

### 5.5 서브 2개일 때의 진짜 가치 — **순차 호출**

서브가 하나면 그냥 wrapping이지만, **둘 이상**이면 메인이 진짜 일을 합니다.

**예시 요청**: `"MCP 검색해서 노트로 저장해줘"`

```
메인 LLM 사고:
  1) "검색이 필요하네" → call_search_agent("MCP가 뭐야?")
  2) (검색 결과 받음)
  3) "이걸 노트로 저장하라네" → call_note_agent("MCP는 ... 라는 내용을 저장")
  4) 사용자에게 최종 답변
```

→ 메인 그래프 안에서 **agent ↔ tools 루프가 두 번** 돕니다. ReAct 패턴이 서브 호출에도 그대로 적용됨.

### 5.6 확장 패턴

새 도메인(일정, 메일)을 추가할 때:

1. `tools/calendar_tools.py` 작성 (`@tool` 함수들)
2. `assistant/agents/calendar_agent.py` 작성 (`create_agent`로 서브 만들기)
3. `main_agent.py`에 `call_calendar_agent` wrapper 추가 + tools 리스트에 추가
4. 메인 프롬프트의 "사용 가능한 서브 비서"에 한 줄 추가

**메인 코드 변경 최소화**가 핵심. 메인은 라우터 역할만 함.

---

## 6. End-to-End 흐름 (`test_note_agent.py`)

### 6.1 실행 코드

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from assistant.main_agent import main_agent

def run(user_input: str):
    result = main_agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25},
    )

    print("📜 메시지 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        tc_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc_info = f" [tool_calls: {[(tc['name'], tc['args']) for tc in msg.tool_calls]}]"
        print(f"  [{i}] {msg_type}{tc_info}")

    print(f"🎯 최종 답변: {result['messages'][-1].content}")
```

### 6.2 한 요청이 두 서브를 거치는 흐름

`"MCP 검색해서 노트로 저장해줘"` 한 줄을 던지면:

```
[메인 그래프]
  HumanMessage("MCP 검색해서 노트로 저장")
  ↓
  agent 노드 (메인 LLM)
  → tool_calls: call_search_agent(query="MCP가 뭐야?")
  ↓
  tools 노드: call_search_agent 실행
      [검색 서브 그래프]
        agent → web_search → agent → END
      [최종 답변: "MCP는 ... 이런 프로토콜이야 (출처: ...)"]
  ↓
  ToolMessage("MCP는 ...")
  ↓
  agent 노드 (메인 LLM)
  → tool_calls: call_note_agent(query="MCP는 ... 내용을 저장해줘")
  ↓
  tools 노드: call_note_agent 실행
      [노트 서브 그래프]
        agent → save_note → agent → END
      [최종 답변: "노트 저장 완료. ID: note_0004"]
  ↓
  ToolMessage("노트 저장 완료...")
  ↓
  agent 노드 (메인 LLM)
  → "MCP 정보를 찾아서 노트에 저장했어요!"
  ↓
  END
```

### 6.3 메시지 흐름 출력 예시

```
[0] HumanMessage
[1] AIMessage [tool_calls: [('call_search_agent', {'query': 'MCP가 뭐야'})]]
[2] ToolMessage              # 검색 서브의 최종 답변
[3] AIMessage [tool_calls: [('call_note_agent', {'query': 'MCP 정보 저장...'})]]
[4] ToolMessage              # 노트 서브의 최종 답변
[5] AIMessage                # 메인의 최종 답변
```

→ **서브 한 번 호출이 메인 입장에선 "도구 한 번 호출"로 보임.** 그게 핵심.

---

## 7. 정리 — 핵심 4가지

### ① `@tool` = 도구
파이썬 함수에 `@tool`을 붙이고 docstring/타입힌트를 잘 쓰면 LLM이 부를 수 있는 도구가 된다.

### ② `create_agent` = 서브에이전트
`model + tools + system_prompt` 3가지를 넣으면 ReAct 루프(`agent ↔ tools`)가 도는 LangGraph 에이전트가 만들어진다.

### ③ 서브를 `@tool`로 wrapping = 메인
서브에이전트의 `invoke()` 호출을 `@tool` 함수 안에 넣으면, 메인이 서브를 "도구"처럼 부를 수 있다 → Supervisor 패턴.

### ④ 서브 2개 이상 = 진짜 라우팅 + 순차 호출
서브가 둘 이상이면 메인이 (a) 어느 서브를 부를지 분류하고, (b) 필요하면 여러 서브를 순차로 부른다. wrapper docstring에 **"사용하면 안 되는 경우"** 도 적어 경계를 명확히 하는 게 라우팅 정확도의 핵심.

---

## 8. 라이브 데모

```bash
source venv/bin/activate
python test_note_agent.py
```

### 데모 시나리오

1. **노트 저장** (단일 서브)
   `"오늘 배운 거: create_agent는 LangChain에 있지만 내부는 LangGraph 그래프야. 노트해줘"`
   → 메인 → `call_note_agent` → `save_note`

2. **노트 검색** (단일 서브)
   `"LangGraph 관련해서 적은 거 검색해줘"`
   → 메인 → `call_note_agent` → `search_notes`

3. **웹 검색** (단일 서브, 다른 도메인)
   `"MCP가 뭐야?"`
   → 메인 → `call_search_agent` → `web_search`

4. **검색 + 저장** (두 서브 순차 호출 — 메인 라우팅의 진가)
   `"MCP 검색해서 노트로 저장해줘"`
   → 메인 → `call_search_agent` → 메인 → `call_note_agent`

5. **잡담** (도구 없이 직접)
   `"안녕"`
   → 메인이 도구 없이 답변

### 그래프 시각화

```python
print(main_agent.get_graph().draw_ascii())
print(note_agent.get_graph().draw_ascii())
print(search_agent.get_graph().draw_ascii())
```

세 그래프 모두 동일한 ReAct 토폴로지임을 강조 → "구조는 같고 도구만 다르다"가 메시지.

---

## 9. Q&A 대비

- **Q. 메인이 두 서브를 동시에 호출할 수 있나?**
  A. LangGraph는 같은 턴에 여러 `tool_calls`가 있으면 병렬 실행함. 다만 "검색해서 저장"처럼 결과 의존성이 있으면 LLM이 알아서 순차로 만듦.

- **Q. 서브가 서브를 또 부를 수 있나?**
  A. 가능. wrapper를 그 서브의 tools에도 등록하면 됨. 다만 무한 위임 위험이 있으니 보통 메인→서브 단방향.

- **Q. 대화가 이어지나?**
  A. 지금은 안 됨. 매 invoke가 독립. LangGraph의 `Checkpointer`(MemorySaver, SqliteSaver)를 붙이면 됨.

- **Q. 어떻게 LLM이 docstring을 읽나?**
  A. LangChain이 함수 시그니처 + docstring을 JSON schema로 변환해 LLM의 tool spec에 넘김. LLM은 그 spec을 보고 호출 형식을 생성.

- **Q. 도구가 많아지면 어떻게 정리?**
  A. 도메인별 파일 분리 → 도메인별 서브에이전트 → 메인이 서브만 보면 됨. 도구가 30개라도 메인 입장에선 서브 N개만 보임.
