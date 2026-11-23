"""Check generated device instructions; this is not native GPU execution."""
from pathlib import Path
import re
import sys


def validate_ptx(content):
    text = re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', content)
    target = re.search(r'\.target\s+(sm_\d+)', text)
    entry = re.search(r'\.entry\s+\S*barrett_kernel[^\{]*\{', text)
    if target is None or entry is None:
        raise ValueError('missing target or Barrett kernel')
    start, depth, end = entry.end(), 1, entry.end()
    while end < len(text) and depth:
        if text[end] == '{':depth += 1
        elif text[end] == '}':depth -= 1
        end += 1
    if depth:
        raise ValueError('unterminated Barrett kernel')
    body = text[start:end-1]
    if re.search(r'\b(?:div|rem)\.|\bcall(?:\.|\s)', body):
        raise ValueError('Barrett kernel contains division, remainder or a helper call')
    count = len(re.findall(r'\bmul\.hi\.u64\b', body))
    if count < 1:
        raise ValueError('missing native high-half multiplication')
    return target[1], count


if __name__ == '__main__':
    if len(sys.argv) < 2:raise SystemExit('usage: check_barrett_ptx.py PTX [PTX ...]')
    for filename in sys.argv[1:]:
        architecture, count = validate_ptx(Path(filename).read_text())
        print(f'BARRETT PTX PASS: {architecture}, {count} mul.hi.u64 instructions, no division/remainder/helper calls')
