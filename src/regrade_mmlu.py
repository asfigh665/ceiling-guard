"""Match extracted answers to MMLU-Pro option letters or content."""
import re
from lt1.data import _fold
from lt1.debate import parse_response

LETTERS = "ABCDEFGHIJ"
_NEG = ("not sure", "cannot determine", "unknown", "no idea", "unable to",
        "insufficient", "impossible to")


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = _fold(s)
    return re.sub(r"[^a-z0-9]", "", s)


def extract_v3(raw: str, choices):
    '返回 (letter, method)。'
    ans, _ = parse_response(raw)
    a = (ans or "").strip()

    if not a:
        return "", "empty"
    low_all = a.lower()
    if any(w in (raw or "").lower() for w in _NEG) and len(a) > 30:
        return "", "refused"

    # 1) answer 字段就是一个字母（可能带括号/句点）
    m = re.fullmatch(r"[\s\[\(]*([A-Ja-j])[\s\]\)\.:]*", a)
    if m:
        return m.group(1).upper(), "single-letter"

    # 2) "E. Penetration pricing" / "(E) something"
    m = re.match(r"^\s*[\[\(]?([A-Ja-j])[\]\)\.:]\s+\S", a)
    if m:
        return m.group(1).upper(), "letter-then-text"

    # 3) answer 字段里显式给出 "answer is X"
    m = re.search(r"(?:answer\s*(?:is|:))\s*[\[\(]?([A-Ja-j])\b", low_all)
    if m:
        return m.group(1).upper(), "answer-phrase"

    # 4) 与选项内容做等值匹配
    na = _norm(a)
    if na and choices:
        for i, ch in enumerate(choices):
            if _norm(ch) == na:
                return LETTERS[i], "content-equal"
        # 5) 包含匹配（较长者优先）
        best = None
        for i, ch in enumerate(choices):
            n = _norm(ch)
            if len(n) >= 8 and (n in na or na in n):
                if best is None or len(n) > best[1]:
                    best = (i, len(n))
        if best:
            return LETTERS[best[0]], "content-contains"

    # 6) answer 字段里唯一的孤立大写字母
    letters = re.findall(r"\b([A-J])\b", a)
    if len(set(letters)) == 1 and len(a) < 120:
        return letters[0], "loose-letter"

    return "", "failed"


