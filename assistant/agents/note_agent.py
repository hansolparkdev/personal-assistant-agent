# assistant/agents/note_agent.py
#
# ---------------------------------------------------------------------------
# 노트 서브에이전트
# ---------------------------------------------------------------------------
# create_agent로 만든 단일 ReAct 에이전트.
# 도구 3개(save/search/list)를 들고 학습 노트를 관리한다.
# ---------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools.note_tools import note_tools


# 노트 에이전트 전용 LLM
# (메인 에이전트와 동일 모델을 써도 되고, 더 작은/저렴한 모델로 바꿔도 됨)
_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)


# 노트 전문 시스템 프롬프트
_SYSTEM_PROMPT = """너는 학습 노트 전문 비서야. 사용자의 학습 내용/메모/일기를 저장하고 검색해.

원칙:
- 사용자가 뭔가를 "기록하고 싶다", "정리해줘", "저장해줘" 라고 하면 → save_note 사용
- "찾아줘", "어디 적었지", "전에 뭐라고 했지" 같은 검색은 → search_notes 사용
- "오늘 뭐 배웠지", "오늘 적은 거" 같은 일일 조회는 → list_today_notes 사용
- 저장할 때 제목은 짧고 명확하게, 태그는 1~3개 정도로 자동 추출해
- 한국어로 답해. 기술 용어는 영어 그대로.
"""

# 서브에이전트 인스턴스
note_agent = create_agent(
    model=_llm,
    tools=note_tools,
    system_prompt=_SYSTEM_PROMPT,
)