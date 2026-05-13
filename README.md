# Personal Assistant Agent

LangChain · LangGraph 기반의 **Supervisor 멀티 에이전트** 개인 비서.
사용자의 요청을 분석해 적절한 서브 비서(노트 / 검색)에게 위임하는 라우터 패턴으로 동작한다.

---

## ✨ 주요 기능

- **Supervisor 패턴**: 메인 에이전트가 직접 답하지 않고, 요청을 분류해 서브 에이전트에게 위임
- **노트 비서** (`note_agent`): 학습 내용 · 메모 · 일기를 `data/notes.json`에 저장하고 검색
- **검색 비서** (`search_agent`): DuckDuckGo 기반 웹 검색으로 최신 정보 조회 (API 키 불필요)
- **두 가지 구현체**
  - `assistant/main_agent.py` — `create_agent` 기반 (도구 호출로 핸드오프)
  - `assistant/supervisor_graph.py` — `StateGraph` 직접 구성 (명시적 노드 · 조건 엣지)

---

## 🏗 아키텍처

```
                ┌──────────────┐
   user ──────► │  Supervisor  │ ◄────────┐
                └──────┬───────┘          │
                       │ 라우팅 결정       │
            ┌──────────┴──────────┐       │
            ▼                     ▼       │
     ┌────────────┐         ┌────────────┐│
     │ note_agent │         │search_agent││ 결과 반환
     └─────┬──────┘         └─────┬──────┘│
           │                      │       │
           └──────────────────────┴───────┘
```

- **Supervisor**: 어느 서브로 갈지 결정 (`transfer_to_*` 핸드오프 도구의 `tool_calls`가 라우팅 신호)
- **서브 에이전트**: 각자 자기 도메인 도구만 갖고 작업 수행 후 결과를 supervisor로 반환
- **종료 조건**: supervisor가 더 이상 도구를 호출하지 않으면 최종 답변 반환

---

## 📁 프로젝트 구조

```
personal-assistant-agent/
├── assistant/
│   ├── main_agent.py           # create_agent 기반 supervisor
│   ├── supervisor_graph.py     # StateGraph 기반 supervisor
│   └── agents/
│       ├── note_agent.py       # 노트 서브 에이전트
│       └── search_agent.py     # 검색 서브 에이전트
├── tools/
│   ├── note_tools.py           # 노트 CRUD (JSON 파일 영속화)
│   └── search_tools.py         # DuckDuckGo 웹 검색
├── study/                      # 단계별 학습 스크립트 (LLM → 도구 → 에이전트 → 그래프)
├── data/                       # 노트 저장소 (notes.json — gitignore)
├── docs/
├── test_note_agent.py          # 노트 에이전트 단독 실행
├── test_supervisor_graph.py    # supervisor 전체 흐름 실행 (REPL)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 시작하기

### 1. 저장소 클론 & 가상환경 생성

```bash
git clone https://github.com/hansolparkdev/personal-assistant-agent.git
cd personal-assistant-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env.example`을 복사해 `.env`를 만들고 OpenAI API 키를 채워 넣는다.

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### 4. 실행

대화형 REPL로 supervisor 그래프 실행:

```bash
python test_supervisor_graph.py
```

```
🧑 유저: MCP가 뭐야?
🧑 유저: 방금 답변 노트로 저장해줘
🧑 유저: 내가 저장한 MCP 노트 보여줘
🧑 유저: exit
```

---

## 💡 사용 예시

| 사용자 요청 | 라우팅 결과 |
|---|---|
| "LangGraph state에 대해 학습한 거 메모해줘" | → `note_agent` (저장) |
| "내가 적은 supervisor 패턴 노트 보여줘" | → `note_agent` (검색) |
| "MCP가 뭐야?" | → `search_agent` |
| "LangChain 1.0 새 기능 알려줘" | → `search_agent` |
| "MCP 검색해서 노트로 저장해줘" | → `search_agent` → `note_agent` (순차) |
| "안녕" | supervisor가 직접 답변 |

---

## 🧪 학습 스크립트 (`study/`)

LLM에서 멀티 에이전트로 점진적으로 발전시킨 학습용 스크립트들.

| 파일 | 내용 |
|---|---|
| `hello_llm.py` | 가장 기본적인 ChatOpenAI 호출 |
| `step2_history.py` | 대화 히스토리 유지 |
| `step3_tool*.py` | LLM의 도구 호출 (function calling) |
| `step4_agent_loop*.py` | 직접 구현하는 ReAct 에이전트 루프 |
| `step4_5_agent_executor.py` | `create_agent`로 자동화된 루프 |
| `step5_langgraph*.py` | LangGraph `StateGraph` 입문 |
| `step5_3_prebuilt_react.py` | `create_react_agent` 사용 |

---

## 🛠 기술 스택

- **LLM**: OpenAI (`gpt-4o-mini` 기본)
- **에이전트**: `langchain` 1.x, `langgraph` 1.x
- **검색**: `ddgs` (DuckDuckGo, API 키 불필요)
- **저장소**: JSON 파일 (`data/notes.json`)

---

## 📚 더 읽을거리

- [docs/architecture.md](docs/architecture.md) — 전체 흐름 · 코드 읽는 순서 · 각 에이전트가 실행될 때 무엇을 읽는지 상세 정리
- [docs/messages.md](docs/messages.md) — LangChain 메시지 타입(`SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`) 완전 정리. 언제·왜·어떻게 쓰는지, `tool_calls` ↔ `ToolMessage` 매칭 규칙

---

## 📝 라이선스

개인 학습 프로젝트.
