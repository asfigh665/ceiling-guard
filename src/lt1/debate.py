"""Parse answers and confidence from stored model responses."""
from __future__ import annotations
import json
import re
TRUNCATED_MARK = "\n<<TRUNCATED>>"

ABSTAIN = "__ABSTAIN__"

_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(.*?)"', re.S)
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_EMBEDDED_OBJ = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.S)
# 模型常写 {"answer": -3} —— 数字/词不加引号，json.loads 会失败。用宽容正则兜底。
_LOOSE_ANSWER = re.compile(r'"answer"\s*:\s*(.+?)\s*(?:,\s*"confidence"|\}\s*$)', re.S)
_LOOSE_CONF = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')
_LAST_EQ = re.compile(r"=\s*([^=\n]+?)\s*$")


def _try_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _clean_answer_value(v: str) -> str:
    "从 'answer' 字段的原始文本里清出答案本身。"
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1].strip()
    # 宽容正则可能把分隔逗号带进来；只去掉一个逗号，不碰花括号
    if v.endswith(","):
        v = v[:-1].rstrip()
    return v.strip()


def parse_response(raw: str) -> tuple[str, float]:
    '从模型输出里抽 (answer, confidence)。'
    text = (raw or "").replace(TRUNCATED_MARK, "")
    ans = None
    conf = 0.5

    # 1) 整体或文中最后一个 JSON 对象
    obj = _try_json(text.strip())
    if not isinstance(obj, dict):
        hits = _EMBEDDED_OBJ.findall(text)
        obj = _try_json(hits[-1]) if hits else None
    if isinstance(obj, dict) and obj.get("answer") is not None:
        ans = _clean_answer_value(str(obj["answer"]))
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5

    # 2) 宽容兜底：未加引号的 answer 值
    if not ans:
        m = _LOOSE_ANSWER.findall(text)
        if m:
            ans = _clean_answer_value(m[-1])
            mc = _LOOSE_CONF.findall(text)
            if mc:
                try:
                    conf = float(mc[-1])
                except ValueError:
                    pass

    # 3) \boxed{}
    if not ans:
        boxes = _BOXED_RE.findall(text)
        if boxes:
            ans = boxes[-1].strip()

    # 4) 以 "= X" 结尾
    if not ans:
        for line in reversed([l for l in text.strip().splitlines() if l.strip()]):
            m = _LAST_EQ.search(line)
            if m:
                ans = m.group(1).strip()
                break

    # 5) 最后手段：倒序找第一条较短的、像答案的行
    if not ans:
        for line in reversed([l.strip() for l in text.strip().splitlines() if l.strip()]):
            if len(line) <= 60:
                ans = line
                break
    if not ans:
        ans = text.strip().split("\n")[0][:200] if text.strip() else ""

    conf = min(max(conf, 0.0), 1.0)
    return ans or "", conf


