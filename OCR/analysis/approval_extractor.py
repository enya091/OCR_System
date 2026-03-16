from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import fitz

ROLE_KEYWORDS = [
    "代理科長",
    "政風室主任",
    "書記官長",
    "檢察長",
    "會計室主任",
    "主任檢察官",
]

ROLE_FRAGMENTS = [
    "主任",
    "科長",
    "檢察長",
    "檢察官",
    "書記官長",
    "會計室",
    "政風室",
    "代理",
]

IGNORE_NAME_WORDS = {
    "承辦單位",
    "會辦單位",
    "決行",
    "說",
    "簽",
    "會",
    "可底",
    "推人",
}

APPROVAL_NOTE_TOKENS = ["可", "准", "核", "閱"]
CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")


@dataclass
class TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    role: str = ""
    name: str = ""
    date_tuple: tuple[int, int, int] | None = None

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _find_role(text: str) -> str:
    compact = text.replace(" ", "")
    for role in sorted(ROLE_KEYWORDS, key=len, reverse=True):
        if role in compact:
            return role
    return ""


def _find_name(text: str) -> str:
    candidates = CJK_RE.findall(text)
    valid: list[str] = []

    for candidate in candidates:
        if candidate in ROLE_KEYWORDS:
            continue
        if candidate in IGNORE_NAME_WORDS:
            continue
        if any(fragment in candidate for fragment in ROLE_FRAGMENTS):
            continue
        valid.append(candidate)

    if not valid:
        return ""

    # In stamp-like strings, the last CJK token is usually the person's name.
    return valid[-1]


def _find_date_tuple(text: str) -> tuple[int, int, int] | None:
    numbers = re.findall(r"\d{1,4}", text)
    if len(numbers) < 3:
        return None

    year = int(numbers[0])
    month = int(numbers[1])
    day = int(numbers[2])
    return year, month, day


def _is_valid_date(date_tuple: tuple[int, int, int] | None) -> bool:
    if not date_tuple:
        return False
    _, month, day = date_tuple
    return 1 <= month <= 12 and 1 <= day <= 31


def _format_date(date_tuple: tuple[int, int, int] | None) -> str:
    if not date_tuple:
        return ""
    year, month, day = date_tuple
    return f"{year}.{month:02d}.{day:02d}"


def _collect_blocks(page: fitz.Page) -> list[TextBlock]:
    blocks: list[TextBlock] = []

    for raw_block in page.get_text("blocks"):
        x0, y0, x1, y1, raw_text, *_ = raw_block
        text = _normalize_text(raw_text)
        if not text:
            continue

        blocks.append(
            TextBlock(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                text=text,
                role=_find_role(text),
                name=_find_name(text),
                date_tuple=_find_date_tuple(text),
            )
        )

    return blocks


def _pair_role_and_name(blocks: list[TextBlock], page_number: int) -> list[dict[str, str | int]]:
    role_blocks = [
        block
        for block in blocks
        if block.role and 110 <= block.y0 <= 560
    ]

    name_blocks = [
        block
        for block in blocks
        if block.name and not block.role
    ]

    used_name_indices: set[int] = set()
    entries: list[dict[str, str | int]] = []

    for role_block in sorted(role_blocks, key=lambda item: (item.y0, item.x0)):
        candidates: list[tuple[float, int, TextBlock]] = []

        for idx, name_block in enumerate(name_blocks):
            if idx in used_name_indices:
                continue

            dx = abs(name_block.x0 - role_block.x0)
            dy = name_block.y0 - role_block.y0

            # Same column and just below/near the role title.
            if dx > 130:
                continue
            if dy < -10 or dy > 140:
                continue

            score = (dx * 2.0) + abs(dy)
            if _is_valid_date(name_block.date_tuple):
                score -= 20

            candidates.append((score, idx, name_block))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, matched_idx, matched_name_block = candidates[0]
            used_name_indices.add(matched_idx)

            entries.append(
                {
                    "role": role_block.role,
                    "name": matched_name_block.name,
                    "date": _format_date(matched_name_block.date_tuple),
                    "date_tuple": matched_name_block.date_tuple,
                    "notes": "",
                    "page": page_number,
                    "anchor_x": role_block.x0,
                    "anchor_y": role_block.y0,
                }
            )
        else:
            entries.append(
                {
                    "role": role_block.role,
                    "name": "",
                    "date": "",
                    "date_tuple": None,
                    "notes": "",
                    "page": page_number,
                    "anchor_x": role_block.x0,
                    "anchor_y": role_block.y0,
                }
            )

    return entries


def _normalize_duplicate_names(entries: list[dict[str, str | int]]) -> None:
    # If same role appears with partial name (e.g. 黃元 / 黃元冠), keep longest name.
    role_to_longest_name: dict[str, str] = {}
    for entry in entries:
        role = str(entry["role"])
        name = str(entry["name"])
        if not name:
            continue

        current = role_to_longest_name.get(role, "")
        if len(name) > len(current):
            role_to_longest_name[role] = name

    for entry in entries:
        role = str(entry["role"])
        name = str(entry["name"])
        longest = role_to_longest_name.get(role, "")
        if not name or not longest:
            continue
        if longest.startswith(name):
            entry["name"] = longest


def _repair_dates(entries: list[dict[str, str | int]]) -> None:
    valid_dates = [
        tuple(entry["date_tuple"])  # type: ignore[arg-type]
        for entry in entries
        if _is_valid_date(entry["date_tuple"])  # type: ignore[arg-type]
    ]

    if not valid_dates:
        return

    canonical_date, _ = Counter(valid_dates).most_common(1)[0]

    for entry in entries:
        date_tuple = entry["date_tuple"]
        if not _is_valid_date(date_tuple):
            entry["date_tuple"] = canonical_date
            entry["date"] = _format_date(canonical_date)
            continue

        year, month, day = date_tuple  # type: ignore[misc]
        can_year, can_month, can_day = canonical_date

        if year == can_year and month == can_month and abs(day - can_day) >= 20:
            entry["date_tuple"] = canonical_date
            entry["date"] = _format_date(canonical_date)


def _attach_notes(entries: list[dict[str, str | int]], blocks: list[TextBlock]) -> None:
    note_blocks = []
    for block in blocks:
        if block.role or block.name:
            continue

        text_compact = block.text.replace(" ", "")
        found_tokens = [token for token in APPROVAL_NOTE_TOKENS if token in text_compact]
        if not found_tokens:
            continue

        note_blocks.append((block, "".join(found_tokens)))

    if not note_blocks or not entries:
        return

    for note_block, note_text in note_blocks:
        best_idx = -1
        best_score = float("inf")

        for idx, entry in enumerate(entries):
            if int(entry["page"]) < 0:
                continue

            dx = abs(note_block.x0 - float(entry["anchor_x"]))
            dy = note_block.y0 - float(entry["anchor_y"])

            if dx > 220:
                continue
            if dy < -160 or dy > 220:
                continue

            score = (dx * 1.5) + abs(dy)
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            existing = str(entries[best_idx]["notes"])
            if note_text not in existing:
                entries[best_idx]["notes"] = (existing + note_text).strip()


def _finalize(entries: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    cleaned: list[dict[str, str | int]] = []

    for entry in entries:
        role = str(entry["role"])
        name = str(entry["name"])
        date_text = str(entry["date"])
        notes = str(entry["notes"])

        handwritten_parts: list[str] = []
        if date_text:
            handwritten_parts.append(date_text)
        if notes:
            handwritten_parts.append(notes)

        cleaned.append(
            {
                "頁碼": int(entry["page"]),
                "職位": role,
                "姓名": name,
                "手寫內容": "；".join(handwritten_parts),
            }
        )

    return cleaned


def extract_approval_entries_from_pdf_bytes(pdf_bytes: bytes) -> list[dict[str, str | int]]:
    """
    Extract role-name-handwriting entries from approval-like pages.
    """
    all_entries: list[dict[str, str | int]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
        for page_index, page in enumerate(pdf_doc, start=1):
            blocks = _collect_blocks(page)
            if not blocks:
                continue

            page_entries = _pair_role_and_name(blocks, page_index)
            if not page_entries:
                continue

            _normalize_duplicate_names(page_entries)
            _repair_dates(page_entries)
            _attach_notes(page_entries, blocks)
            all_entries.extend(page_entries)

    if not all_entries:
        return []

    return _finalize(all_entries)
