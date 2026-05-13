# tools/search_tools.py
#
# ---------------------------------------------------------------------------
# 웹 검색 도구
# ---------------------------------------------------------------------------
# DuckDuckGo Search (ddgs) 사용. API 키 불필요.
# 검색 결과를 LLM이 이해하기 쉽게 정리된 텍스트로 반환.
# ---------------------------------------------------------------------------

# ddg란?
# DuckDuckGo Search (ddgs) 사용. API 키 불필요.
# 검색 결과를 LLM이 이해하기 쉽게 정리된 텍스트로 반환.

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
        # DDGS()는 context manager이므로 with 문으로 사용
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            # results는 list[dict]
            # 예시: [{"title": "검색 결과 1", "body": "검색 결과 1 요약", "url": "https://www.google.com"}]
    except Exception as e:
        return f"검색 중 오류 발생: {e}"

    if not results:
        return f"'{query}'에 대한 검색 결과가 없습니다."

    # LLM이 이해하기 쉽게 정리된 텍스트로 반환
    lines = [f"'{query}' 검색 결과 ({len(results)}개):"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "제목 없음")
        body = r.get("body", "요약 없음")
        url = r.get("url", "URL 없음")
        lines.append(f"[{i}] {title}")
        lines.append(f"  요약: {body}")
        lines.append(f"  URL: {url}")
    
    return "\n".join(lines)

# 모듈 export
search_tools = [web_search]