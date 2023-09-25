from typing import List


def find_repeating_suffix(s):
    max_len = min((len(s) // 2 + 1, 500))
    for size in range(1, max_len):
        if s[-size:] == s[-2*size:-size]:
            return s[-size:]
    return None


def remove_repeating_suffix(s):
    repeating_suffix = find_repeating_suffix(s)
    if not repeating_suffix:
        return s
    while True:
        if not s.endswith(repeating_suffix):
            break
        s = s[:-len(repeating_suffix)]
    return s


def combine_text(text: List[str]):
    cleaned = ""
    for item in text:
        cleaned += remove_repeating_suffix(item)
    return cleaned