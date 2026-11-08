#!/usr/bin/env python3
"""Create a draft daily entry without overwriting existing work."""

import argparse
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def create_entry(entry_date, title):
    parsed_date = date.fromisoformat(entry_date)
    title = title.strip()
    if not title or "\n" in title or "\r" in title:
        raise ValueError("Use a nonempty, single-line title")
    template = (ROOT / "docs/journal/_template.md").read_text()
    content = template.replace("YYYY-MM-DD", parsed_date.isoformat())
    content = content.replace("Replace with the day's title", title)
    destination = ROOT / "docs/journal" / f"{parsed_date.isoformat()}.md"
    # Exclusive creation ensures an existing entry is never truncated.
    with destination.open("x") as entry:
        entry.write(content)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Calendar date in Asia/Shanghai: YYYY-MM-DD")
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    try:
        destination = create_entry(args.date, args.title)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot create journal entry: {error}") from error
    print(f"Created draft: {destination}")
    print("Complete the six sections, set status: published, and rebuild the site.")


if __name__ == "__main__":
    main()
