# Architecture — 전체 흐름과 코드 읽는 순서

이 문서는 `personal-assistant-agent`의 동작 원리, 코드를 어떤 순서로 읽으면 좋은지,
그리고 각 에이전트가 실행될 때 어떤 도구 · 어떤 정보를 참조하는지를 정리한다.

> 메시지 타입(`HumanMessage`, `SystemMessage`, `AIMessage`, `ToolMessage`)이 헷갈리면 먼저 [messages.md](messages.md)를 읽고 오면 이해가 훨씬 빠르다.

---

## 1. 한눈에 보는 전체 흐름

```
┌───────────────────────────────────────────────────────────────────────┐
│  사용자 입력 ("MCP가 뭐야? 그리고 노트로 저장해줘")                       │
└───────────────────────────────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────┐         test_supervisor_graph.py 가 진입점
   │  supervisor_graph.invoke │◄─────── HumanMessage 1개를 state에 넣고 시작
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │   supervisor_node        │  ── _llm_with_handoff (bind_tools)
   │   (LLM이 라우팅 결정)       │     · _SUPERVISOR_PROMPT 읽음
   │                          │     · 사용 가능한 도구: transfer_to_note_agent,
   │                          │                          transfer_to_search_agent
   └────────────┬─────────────┘
                │ tool_calls가 있으면 → route_after_supervisor가 분기
                ▼
        ┌───────┴────────┐
        ▼                ▼
  ┌──────────┐     ┌──────────┐
  │ note_node│     │search_node│   ── ToolMessage 형태로 결과 반환
  └────┬─────┘     └─────┬────┘
       │                 │
       ▼                 ▼
  note_agent       search_agent     ── 각자 create_agent로 만든 ReAct 루프
  (서브)            (서브)
       │                 │
       ▼                 ▼
  note_tools       search_tools     ── 실제 작업 (파일 I/O, 웹 호출)
   save_note         web_search
   search_notes
   list_today_notes
       │                 │
       ▼                 ▼
  data/notes.json    DuckDuckGo
       │                 │
       └────────┬────────┘
                │ 결과를 ToolMessage로 supervisor에 반환
                ▼
   ┌──────────────────────────┐
   │   supervisor_node (재진입)│  ── 더 할 일이 있으면 다시 라우팅,
   │                          │     없으면 tool_calls 없는 최종 응답 생성
   └────────────┬─────────────┘
                │ tool_calls 없음 → "end"
                ▼
              END (사용자에게 최종 답변)
```

핵심 포인트
- **메인은 직접 일을 안 한다**: 분류 · 위임 · 결과 전달만 담당.
- **핸드오프 도구는 라우팅 신호**: `transfer_to_*` 도구는 실제로 빈 문자열을 반환하고, LLM의 `tool_calls` 자체가 "어디로 갈지"의 신호가 된다.
- **반복 가능**: 서브가 끝나면 supervisor로 돌아오기 때문에 "검색 → 노트 저장" 같은 복합 요청도 가능.

---

## 2. 코드 읽는 순서 (입문자 → 전체 구조 파악)

학습 곡선을 따라가도록 단순 → 복잡 순으로 정리했다.

### 🟢 Step 0. 전체 진입점부터 보기

| # | 파일 | 무엇을 보나 |
|---|---|---|
| 1 | `README.md` | 프로젝트가 뭐 하는 건지 큰 그림 |
| 2 | `requirements.txt` | 어떤 라이브러리를 쓰는지 (langchain, langgraph, ddgs) |
| 3 | `.env.example` | 어떤 환경 변수가 필요한지 |

### 🟢 Step 1. 가장 바닥부터 — 도구(Tool)

에이전트가 실제로 "할 수 있는 일"이 정의된 곳. 여기를 먼저 이해해야 위층이 보인다.

| # | 파일 | 핵심 |
|---|---|---|
| 4 | `tools/search_tools.py` | `@tool web_search` — DuckDuckGo 호출, 결과 텍스트 정리 |
| 5 | `tools/note_tools.py` | `@tool save_note / search_notes / list_today_notes` — JSON 파일 CRUD |

> **포인트**: `@tool` 데코레이터는 함수의 docstring · 시그니처를 LLM이 읽을 스키마로 변환한다. **LLM은 docstring을 보고 어떤 도구를 호출할지 결정한다.** 그래서 docstring이 곧 프롬프트의 일부다.

### 🟢 Step 2. 단일 서브 에이전트

도구를 들고 자기 도메인만 처리하는 에이전트.

| # | 파일 | 핵심 |
|---|---|---|
| 6 | `assistant/agents/search_agent.py` | `create_agent(model, tools=search_tools, system_prompt=...)` — 도구 1개짜리 ReAct 에이전트 |
| 7 | `assistant/agents/note_agent.py` | 도구 3개짜리 노트 전문 에이전트 |

> **포인트**: 두 파일은 거의 똑같이 생겼다. `create_agent`가 안에서 "LLM 호출 → tool_calls 있으면 도구 실행 → 결과 다시 LLM에 전달" 루프를 자동으로 돌려준다.

### 🟢 Step 3. Supervisor — 구현 A (도구 호출 방식)

| # | 파일 | 핵심 |
|---|---|---|
| 8 | `assistant/main_agent.py` | 서브 에이전트를 `@tool`로 감싸서 메인의 "도구"로 등록 |
| 9 | `test_note_agent.py` | 노트 에이전트 단독 실행 (가장 단순한 테스트) |

> **핵심 트릭**: `call_note_agent`, `call_search_agent`라는 `@tool` 함수 안에서 서브 에이전트를 `.invoke()` 한다. 메인 LLM 입장에선 그냥 도구 호출 한 번 한 것처럼 보이지만, 실제로는 그 안에서 또 다른 에이전트 루프가 돌아간다.

### 🟢 Step 4. Supervisor — 구현 B (StateGraph 방식)

`main_agent.py`와 같은 일을 하지만 흐름을 **명시적인 그래프**로 만든 버전.

| # | 파일 | 핵심 |
|---|---|---|
| 10 | `assistant/supervisor_graph.py` | `StateGraph` 직접 구성. 노드/엣지/조건 라우터 전부 손으로 정의 |
| 11 | `test_supervisor_graph.py` | REPL로 supervisor_graph 실행 |

> **차이점**:
> - `main_agent.py`는 "메인도 그냥 도구 가진 에이전트"
> - `supervisor_graph.py`는 "메인은 라우팅 노드, 서브는 별도 노드"로 분리
>
> 후자가 흐름이 더 명확하고 디버깅이 쉽다. 실무에서는 보통 후자 패턴을 쓴다.

### 🟢 Step 5. 처음부터 따라 짜보기 — `study/`

위 구조에 도달하기까지의 단계별 학습 스크립트.

```
hello_llm.py
  → step2_history.py
    → step3_tool.py / step3_tool_full.py
      → step4_agent_loop.py / step4_agent_loop_dev.py
        → step4_5_agent_executor.py
          → step5_langgraph.py / step5_langgraph_dev.py
            → step5_3_prebuilt_react.py
              → (assistant/ 본 구현으로 이어짐)
```

처음 보는 사람은 **study/부터 → tools/ → assistant/agents/ → assistant/supervisor_graph.py** 순서가 가장 부드럽다.

---

## 3. 각 에이전트가 실행될 때 "무엇을 읽는가"

LLM은 매 호출마다 다음을 **모두 한 번에** 본다. 각 에이전트별로 정리.

### 3-1. Supervisor (메인)

`assistant/supervisor_graph.py` 의 `supervisor_node`에서 LLM이 보는 것:

| 항목 | 출처 | 내용 |
|---|---|---|
| **시스템 프롬프트** | `_SUPERVISOR_PROMPT` (라인 76) | "너는 supervisor야 / 직접 답 X / 라우팅 규칙 / 한국어 짧고 직설적" |
| **대화 히스토리** | `state["messages"]` (MessagesState) | 사용자 입력 + 지금까지의 모든 메시지 |
| **사용 가능한 도구 스키마** | `_llm.bind_tools(handoff_tools)` | `transfer_to_note_agent`, `transfer_to_search_agent` 의 docstring + 인자 |

**산출물**: `AIMessage` 한 개
- `tool_calls`가 있으면 → 서브로 위임 (route 함수가 분기)
- `tool_calls`가 없으면 → 최종 답변 (END)

> 메인이 도구 결과를 받으면, **이전 ToolMessage까지 모두 포함한 새 히스토리**로 다시 호출된다. 그래서 "검색 결과 받고 → 그걸 노트로 저장" 같은 후속 위임이 가능.

### 3-2. note_agent (서브 — 노트 비서)

`assistant/agents/note_agent.py`. `create_agent`가 만든 ReAct 루프.

| 항목 | 출처 | 내용 |
|---|---|---|
| **시스템 프롬프트** | `_SYSTEM_PROMPT` (라인 23) | "학습 노트 전문 비서 / save_note/search_notes/list_today_notes 언제 쓸지" |
| **입력 메시지** | supervisor가 `query` 문자열로 전달 → `HumanMessage`로 래핑 | supervisor가 압축해서 넘긴 자연어 요청 (예: "MCP 학습 내용 노트로 저장") |
| **사용 가능한 도구** | `note_tools` = `[save_note, search_notes, list_today_notes]` | 각 함수의 docstring · 인자 시그니처 |

**도구별로 실제 읽는 것/쓰는 것**:

| 도구 | 읽는 파일 | 쓰는 파일 | 동작 |
|---|---|---|---|
| `save_note(title, content, tags)` | `data/notes.json` (기존 노트 로드) | `data/notes.json` (append 후 전체 저장) | id 자동 생성 (`note_0001` 형식), `created_at` 현재 시각 |
| `search_notes(query, limit=5)` | `data/notes.json` | — | 제목/본문/태그 부분 일치, 최근 순 정렬, 상위 `limit`개 |
| `list_today_notes()` | `data/notes.json` | — | `created_at`이 오늘 날짜로 시작하는 노트만 |

> **데이터 위치**: `tools/note_tools.py`의 `DATA_DIR = Path(__file__).parent.parent / "data"` → 프로젝트 루트의 `data/notes.json`. 파일이 없으면 `_ensure_data_file()`이 자동 생성.

### 3-3. search_agent (서브 — 검색 비서)

`assistant/agents/search_agent.py`.

| 항목 | 출처 | 내용 |
|---|---|---|
| **시스템 프롬프트** | `_SYSTEM_PROMPT` (라인 22) | "웹 검색 전문 / 무조건 web_search 호출 / 출처 URL 포함" |
| **입력 메시지** | supervisor가 전달한 `query` → `HumanMessage` | 예: "MCP가 뭐야?" |
| **사용 가능한 도구** | `search_tools = [web_search]` | DuckDuckGo 검색 |

**도구가 실제 하는 일**:

| 도구 | 외부 의존 | 동작 |
|---|---|---|
| `web_search(query, max_results=5)` | `ddgs` (DuckDuckGo, API 키 불필요) | 최대 10개로 클램프, `title/body/url` 추출 후 텍스트로 포맷팅 |

> **장점**: API 키가 필요 없어서 즉시 동작. 단점: rate limit, 정확도가 상용 검색 API보다 떨어질 수 있음.

---

## 4. 실제 예시로 따라가기

사용자 입력: **"MCP가 뭐야? 그거 노트로 저장해줘"**

```
[1] HumanMessage("MCP가 뭐야? 그거 노트로 저장해줘")
       │
       ▼
[2] supervisor_node
    LLM이 본 것: _SUPERVISOR_PROMPT + [1]
    LLM 출력: AIMessage(tool_calls=[transfer_to_search_agent(query="MCP가 뭐야?")])
       │
       ▼ route_after_supervisor → "search"
[3] search_node
    - tool_call에서 query="MCP가 뭐야?" 추출
    - search_agent.invoke({"messages": [HumanMessage("MCP가 뭐야?")]})
        ├─ search_agent의 LLM이 본 것: _SYSTEM_PROMPT + 위 메시지
        ├─ LLM 출력: tool_calls=[web_search(query="MCP Model Context Protocol")]
        ├─ web_search 실행 → DuckDuckGo 호출 → 텍스트 결과
        └─ LLM이 결과 받아 한국어로 정리한 AIMessage 반환
    - search_node는 그 결과를 ToolMessage(tool_call_id=...)로 supervisor에 반환
       │
       ▼ edge: search → supervisor
[4] supervisor_node (재진입)
    LLM이 본 것: _SUPERVISOR_PROMPT + [1] + [2] AIMessage + [3] ToolMessage
    LLM 판단: "검색 끝났네, 이제 노트 저장해야지"
    LLM 출력: AIMessage(tool_calls=[transfer_to_note_agent(query="MCP는 ... 노트로 저장")])
       │
       ▼ route → "note"
[5] note_node
    - note_agent.invoke(...)
        ├─ note_agent의 LLM이 본 것: 노트 _SYSTEM_PROMPT + HumanMessage(query)
        ├─ LLM 출력: tool_calls=[save_note(title="MCP", content="...", tags=["MCP","AI"])]
        ├─ save_note 실행 → data/notes.json append → "노트 저장 완료. ID: note_0042"
        └─ LLM이 "저장했어요" 같은 자연어 AIMessage 반환
    - ToolMessage로 supervisor에 반환
       │
       ▼ edge: note → supervisor
[6] supervisor_node (재진입)
    LLM 판단: "할 일 다 끝났다"
    LLM 출력: AIMessage(content="MCP는 ... 노트로 저장했어요.", tool_calls=[])
       │
       ▼ route → "end"
[7] END → 사용자에게 [6]의 content 전달
```

---

## 5. 자주 헷갈리는 포인트

### Q1. `transfer_to_*` 도구는 왜 빈 문자열을 반환하나?
이 도구들은 **실제 실행되지 않는다**. supervisor LLM이 이걸 호출하려고 한 순간(`tool_calls` 발생) 우리가 그래프에서 가로채서 다른 노드로 라우팅한다. 즉, **"도구 이름 = 라우팅 라벨"** 이라는 트릭.

### Q2. `main_agent.py` 와 `supervisor_graph.py` 둘 다 있는데 뭘 써야 하나?
- **`main_agent.py`**: 짧고 간단. `create_agent`가 다 알아서 해준다.
- **`supervisor_graph.py`**: 흐름이 보이고, 노드별 로깅 · 커스텀 로직 삽입이 쉽다.

학습용으로 둘 다 둔 것. 실 서비스로 확장한다면 `supervisor_graph.py` 쪽이 유연하다.

### Q3. 서브 에이전트는 supervisor의 대화 히스토리 전체를 보나?
**아니다.** supervisor가 추출한 `query` 문자열만 `HumanMessage`로 받는다. 즉 서브는 supervisor가 압축해서 넘긴 한 문장만 보고 일한다. (=context 격리)

### Q4. 노트는 어디 저장되나?
`data/notes.json` 한 파일. 구조:
```json
[
  {
    "id": "note_0001",
    "title": "...",
    "content": "...",
    "tags": ["..."],
    "created_at": "2026-05-13T15:41:00.123456"
  }
]
```
`.gitignore`에 포함되어 있어 GitHub에는 올라가지 않는다 (개인 데이터 보호).

### Q5. `recursion_limit=25` 는 뭐지?
한 invoke 안에서 노드를 최대 25번 거치도록 제한. 무한 루프 방지용. 복잡한 멀티 스텝 작업이 필요하면 늘릴 수 있다.

---

## 6. 확장 가이드 — 서브 에이전트 추가하기

새 서브 비서(예: 일정 비서)를 추가하려면:

1. `tools/calendar_tools.py` — `@tool` 함수 작성, `calendar_tools` 리스트 export
2. `assistant/agents/calendar_agent.py` — `create_agent(model, tools=calendar_tools, system_prompt=...)`
3. `assistant/supervisor_graph.py`:
   - `@tool transfer_to_calendar_agent` 추가
   - `handoff_tools`에 추가
   - `calendar_node` 함수 추가
   - `route_after_supervisor`에 분기 추가
   - `builder.add_node("calendar", calendar_node)` + 엣지 추가
   - `_SUPERVISOR_PROMPT` 의 "사용 가능한 서브 비서" 목록 갱신

`main_agent.py` 쪽도 동일한 패턴으로 `call_calendar_agent` `@tool`만 추가하면 된다.
