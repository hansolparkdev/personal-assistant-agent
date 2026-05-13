# tools/note_tools.py
#
# ---------------------------------------------------------------------------
# 노트 관리 도구
# ---------------------------------------------------------------------------
# JSON 파일에 노트를 영속화하는 단순 CRUD 도구.
# 추후 SQLite 등으로 마이그레이션 가능. 지금은 단순함이 우선.
# ---------------------------------------------------------------------------

import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# 데이터 파일 경로
DATA_DIR = Path(__file__).parent.parent / "data"
NOTES_FILE = DATA_DIR / "notes.json"


def _ensure_data_file():
    """data/notes.json이 없으면 생성."""
    DATA_DIR.mkdir(exist_ok=True)
    if not NOTES_FILE.exists():
        NOTES_FILE.write_text("[]", encoding="utf-8")


def _load_notes() -> list[dict]:
    """노트 전체를 로드."""
    _ensure_data_file()
    return json.loads(NOTES_FILE.read_text(encoding="utf-8"))


def _save_notes(notes: list[dict]):
    """노트 전체를 저장."""
    _ensure_data_file()
    NOTES_FILE.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


@tool
def search_notes(query: str, limit: int = 5) -> str:
    """노트를 키워드로 검색합니다. 제목/본문/태그에서 매칭.
    
    Args:
        query: 검색 키워드.
        limit: 최대 결과 수 (기본 5).
    
    Returns:
        매칭된 노트들의 요약.
    """
    notes = _load_notes()
    query_lower = query.lower()
    
    matched = []
    for note in notes:
        # 제목/본문/태그 중 어디든 매칭되면 포함
        if (query_lower in note["title"].lower() or
            query_lower in note["content"].lower() or
            any(query_lower in tag.lower() for tag in note["tags"])):
            matched.append(note)
    
    if not matched:
        return f"'{query}'에 매칭되는 노트가 없습니다."
    
    # 최근 순으로 정렬
    matched.sort(key=lambda n: n["created_at"], reverse=True)
    matched = matched[:limit]
    
    # 결과 포맷팅
    lines = [f"'{query}' 검색 결과 ({len(matched)}개):"]
    for note in matched:
        date = note["created_at"][:10]  # YYYY-MM-DD만
        tags = ", ".join(note["tags"]) if note["tags"] else "(태그 없음)"
        # content 미리보기 100자
        preview = note["content"][:100] + ("..." if len(note["content"]) > 100 else "")
        lines.append(f"\n[{note['id']}] {note['title']} ({date})")
        lines.append(f"  태그: {tags}")
        lines.append(f"  내용: {preview}")
    
    return "\n".join(lines)


@tool
def list_today_notes() -> str:
    """오늘 작성한 노트 전체를 조회합니다.
    
    Returns:
        오늘의 노트들.
    """
    notes = _load_notes()
    today = datetime.now().date().isoformat()  # YYYY-MM-DD
    
    today_notes = [n for n in notes if n["created_at"].startswith(today)]
    
    if not today_notes:
        return "오늘 작성한 노트가 없습니다."
    
    lines = [f"오늘({today}) 작성한 노트 ({len(today_notes)}개):"]
    for note in today_notes:
        tags = ", ".join(note["tags"]) if note["tags"] else "(태그 없음)"
        lines.append(f"\n[{note['id']}] {note['title']}")
        lines.append(f"  태그: {tags}")
        lines.append(f"  내용: {note['content'][:150]}{'...' if len(note['content']) > 150 else ''}")
    
    return "\n".join(lines)


# 모듈에서 export할 도구 모음
note_tools = [save_note, search_notes, list_today_notes]