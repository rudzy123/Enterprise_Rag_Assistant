import os
import re
from pathlib import Path

CURATED_DIR = Path("data/curated")

def split_markdown_sections(text: str):
    """
    Splits markdown text into sections.
    Prefers ### sections; falls back to ## if no ### exist.
    Returns a list of (section_title, section_text).
    """

    # Try splitting on ### first
    sections = re.split(r"\n###\s+", text)

    if len(sections) > 1:
        # First chunk is preamble; discard if empty
        results = []
        for section in sections[1:]:
            lines = section.strip().splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                results.append((title, body))
        return results

    # Fallback to ##
    sections = re.split(r"\n##\s+", text)
    results = []
    for section in sections[1:]:
        lines = section.strip().splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            results.append((title, body))

    return results


def ingest_curated_markdown():
    all_chunks = []

    for md_file in CURATED_DIR.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        sections = split_markdown_sections(text)

        for idx, (title, content) in enumerate(sections):
            chunk = {
                "source_file": md_file.name,
                "section_title": title,
                "text": content,
                "chunk_id": f"{md_file.stem}_{idx}"
            }
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_curated_markdown()

    print(f"\nIngested {len(chunks)} chunks:\n")

    for chunk in chunks:
        print("=" * 80)
        print(f"Source:  {chunk['source_file']}")
        print(f"Section: {chunk['section_title']}")
        print("-" * 80)
        print(chunk["text"][:500])  # preview only
        print()
