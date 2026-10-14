"""
maintenance_analyzer.py
-----------------------
Requirement 3: Maintenance Notes as a Second Signal
Scans the shared maintenance log for entries related to a specific arm.
Uses keyword matching + simple scoring — NOT sensor numbers.
"""

import re
from datetime import datetime

# Keywords that suggest a maintenance concern (weighted)
CONCERN_KEYWORDS = {
    # High-concern terms
    "overheating":   3,
    "overheat":      3,
    "bearing wear":  3,
    "bearing":       2,
    "grinding":      3,
    "squealing":     2,
    "rattling":      2,
    "hot":           2,
    "dangerously":   3,
    "dangerous":     3,
    "immediate":     3,
    "urgent":        3,
    "fail":          3,
    "failure":       3,
    # Medium-concern terms
    "vibration":     2,
    "unusual":       2,
    "noise":         2,
    "rough":         2,
    "stiff":         2,
    "resistance":    2,
    "hotter":        2,
    "warm":          1,
    "spike":         2,
    "spikes":        2,
    # Low-concern / monitoring terms
    "monitoring":    1,
    "flagged":       1,
    "inspection":    1,
    "lubricated":    1,
    "recommend":     2,
}

# Threshold to decide if maintenance signal is "concerning"
CONCERN_THRESHOLD = 3


def parse_log(log_path: str) -> list[dict]:
    """
    Parse the maintenance log into structured entries.

    Each entry is a dict with:
        - 'timestamp' : datetime | None
        - 'arm_id'    : str | None
        - 'text'      : str  (full note text)
    """
    entries = []
    pattern = re.compile(
        r"\[(?P<ts>[^\]]+)\]\s+(?P<arm>ARM_\d+)\s+-\s+(?P<text>.+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                try:
                    ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M")
                except ValueError:
                    ts = None
                entries.append({
                    "timestamp": ts,
                    "arm_id":    m.group("arm"),
                    "text":      m.group("text"),
                })
            else:
                # Unstructured line — still store it
                entries.append({"timestamp": None, "arm_id": None, "text": line})

    return entries


def score_entry(text: str) -> int:
    """Return the concern score for a single log entry."""
    text_lower = text.lower()
    score = 0
    for keyword, weight in CONCERN_KEYWORDS.items():
        if keyword in text_lower:
            score += weight
    return score


def analyze_maintenance_log(log_path: str, arm_id: str) -> dict:
    """
    Analyze the maintenance log for entries related to the given arm.

    Parameters
    ----------
    log_path : str  – path to maintenance_log.txt
    arm_id   : str  – e.g. 'ARM_07'

    Returns
    -------
    dict with:
        - 'maintenance_concern' : bool
        - 'total_score'         : int
        - 'matched_entries'     : list[str]  – relevant log notes
        - 'keywords_found'      : list[str]
    """
    entries = parse_log(log_path)
    arm_entries = [e for e in entries if e.get("arm_id") == arm_id]

    total_score   = 0
    matched_notes = []
    keywords_found = set()

    for entry in arm_entries:
        text  = entry["text"]
        score = score_entry(text)
        total_score += score
        if score > 0:
            matched_notes.append(text)
            text_lower = text.lower()
            for kw in CONCERN_KEYWORDS:
                if kw in text_lower:
                    keywords_found.add(kw)

    return {
        "maintenance_concern": total_score >= CONCERN_THRESHOLD,
        "total_score":         total_score,
        "matched_entries":     matched_notes,
        "keywords_found":      sorted(keywords_found),
    }
