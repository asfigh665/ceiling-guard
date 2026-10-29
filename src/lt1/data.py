"""Answer normalization and exact numeric equivalence."""
from __future__ import annotations
import re
from fractions import Fraction
from typing import Any, Optional

_WS = re.compile(r"\s+")
_DROP_CMD = re.compile(r"\\(left|right)\s*")
_MMA_OPT = re.compile(
    r"(?:answer\s*(?:is|:)\s*|option\s*|^|\()\s*[\(\[]?([A-Ja-j])[\)\].:]?\s*$", re.I)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def _fold(s: str) -> str:
    '把转义后的反斜杠折叠回单个，去 LaTeX 噪声，移除所有空白。'
    s = s.replace("\\\\", "\\")
    s = _DROP_CMD.sub("", s)
    s = s.replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\leq", "\\le").replace("\\geq", "\\ge")
    s = s.replace("\u2212", "-")            # Unicode minus
    s = s.replace("\u03c0", "\\pi")
    s = _WS.sub("", s)
    s = s.rstrip(".")
    while s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    return s


def normalize_answer(x: Any, kind: str = "generic") -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if kind == "mmlu_pro":
        return _extract_option(s)
    return _fold(s)


def _extract_option(s: str) -> str:
    '从自然语言回复里可靠地抽出 A–J 选项字母。抽不到返回空串（判错）。'
    if not s:
        return ""
    low = s.lower()
    if any(w in low for w in ("not sure", "cannot determine", "unknown",
                              "no idea", "unable to", "insufficient")):
        return ""

    b = _BOXED.findall(s)
    if b:
        cand = _fold(b[-1]).upper()
        if len(cand) == 1 and cand in "ABCDEFGHIJ":
            return cand

    patterns = (
        r"answer\s*(?:is|:)\s*[\(\[]?([a-j])\b",
        r"option\s*[\(\[]?([a-j])\b",
        r"^\s*[\(\[]([a-j])[\)\]]",
        r"^\s*([a-j])[\)\].:]",
    )
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return m.group(1).upper()

    core = _fold(s).upper()
    if len(core) == 1 and core in "ABCDEFGHIJ":
        return core

    # 句末独立字母（"I think C" / "it is D."）——要求前面是空白，避免命中 "sure" 的末字母
    m = re.search(r"(?:^|\s)([a-j])\s*[.!]?\s*$", low)
    if m:
        return m.group(1).upper()
    return ""


def answers_equal(pred: Any, gold: Any, kind: str = "generic") -> bool:
    '严格但规范化后判等。空答案一律 False。'
    if pred is None or gold is None:
        return False
    a_raw, b_raw = str(pred).strip(), str(gold).strip()
    if not a_raw or not b_raw:
        return False
    a = normalize_answer(a_raw, kind)
    b = normalize_answer(b_raw, kind)
    if not a or not b:
        return False
    if _cmp(a, b):
        return True
    # MATH-500 的 gold 常见 "x=5" 形式；模型只写 "5" 也应算对（反之亦然）
    av, bv = _assignment_value(a), _assignment_value(b)
    if av is not None and bv is None:
        return _cmp(av, b)
    if bv is not None and av is None:
        return _cmp(a, bv)
    if av is not None and bv is not None:
        return _cmp(av, bv)
    return False


_ASSIGN = re.compile(r"^(?:[A-Za-z]|\\?[A-Za-z]_\d+)=(.+)$")


def _assignment_value(s: str) -> Optional[str]:
    m = _ASSIGN.match(s)
    return m.group(1) if m else None


def _cmp(a: str, b: str) -> bool:
    '已规范化的两个答案是否等价（字符串或数值）。'
    if not a or not b:
        return False
    if a == b:
        return True
    fa, fb = _as_fraction(a), _as_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb
    return False


_NUM = re.compile(r"^[+-]?\d+(\.\d+)?$")
_FRAC = re.compile(r"^([+-]?)\\frac\{([^{}]*)\}\{([^{}]*)\}$")
# 真机输出里极常见：\frac{14}{3} 被模型写成 14/3、-14/3、(-14)/3
_SLASH = re.compile(r"^([+-]?)\(?([+-]?\d+)\)?/\(?([+-]?\d+)\)?$")
_DECFRAC = re.compile(r"^([+-]?)(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$")


def _as_fraction(s: str) -> Optional[Fraction]:
    '把 '
    if not s:
        return None
    m = _FRAC.match(s)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        try:
            num = Fraction(m.group(2))
            den = Fraction(m.group(3))
            return sign * num / den if den else None
        except (ValueError, ZeroDivisionError):
            return None
    for pat in (_SLASH, _DECFRAC):
        m = pat.match(s)
        if m:
            sign = -1 if m.group(1) == "-" else 1
            try:
                num = Fraction(m.group(2))
                den = Fraction(m.group(3))
                return sign * num / den if den else None
            except (ValueError, ZeroDivisionError):
                return None
    if _NUM.match(s):
        try:
            return Fraction(s)
        except ValueError:
            return None
    return None


# ------------------------------------------------------------ loaders
