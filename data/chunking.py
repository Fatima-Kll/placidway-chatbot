
import html
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/pages.json")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 1800

# Headings likely to appear in the approved PlacidWay pages.
KNOWN_HEADINGS = {
    "profile overview",
    "program type",
    "included in the inpatient program",
    "included in the in-patient program",
    "included in the outpatient program",
    "included in the out-patient program",
    "excluded services and contingency costs",
    "optional procedures and their costs",
    "optional procedures",
    "follow-up support",
    "coordination with existing doctors",
    "contact information",
    "frequently asked questions",
}


def clean_text(text, page_title=""):
    """Normalize text and remove known interface noise."""
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    title = page_title.lower()

    # Remove interface content before the main content on these pages.
    if (
        "immunity therapy center" in title
        and "cancer treatment in tijuana mexico" in title
    ):
        marker = "profile overview"
        position = text.lower().find(marker)
        if position != -1:
            text = text[position:]

    elif "alternative medicine cost abroad" in title:
        marker = "alternative medicine cost abroad"
        position = text.lower().find(marker)
        if position != -1:
            text = text[position:]

    interface_labels = {
        "select time",
        "select date",
        "select time zone",
        "request a video call",
    }

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line or line.lower() in interface_labels:
            continue

        lines.append(line)

    return "\n".join(lines)



def is_heading(line):
    """Recognize explicit section headings without splitting table rows."""
    normalized = line.strip().lower().rstrip(":")

    if normalized in KNOWN_HEADINGS:
        return True

    heading_prefixes = (
        "included in the ",
        "excluded services",
        "optional procedures",
        "the itc philosophy",
        "week 1 -",
        "week 2 -",
        "week 3 -",
        "program overview",
        "treatment overview",
        "cost breakdown",
        "cost of alternative ",
        "price list",
        "contact itc",
        "frequently asked questions",
        "profile overview",
        "attractions worth exploring",
        "how to get to ",
        "by plane",
        "by train",
    )

    if normalized.startswith(heading_prefixes):
        return True

    # Do not interpret table values as section headings.
    if re.fullmatch(
        r"\d+\s+weeks?|\$[\d,]+(?:\.\d{2})?",
        line.strip(),
        re.IGNORECASE,
    ):
        return False

    # Avoid guessing that ordinary short labels are headings.
    return False



def extract_sections(text, page_title):
    """Split a page into heading-labelled sections."""
    sections = []
    current_heading = page_title or "Main content"
    current_lines = []

    for line in text.splitlines():
        if is_heading(line):
            content = "\n".join(current_lines).strip()

            if content:
                sections.append((current_heading, content))

            current_heading = line.rstrip(":").strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_heading, content))

    return sections


def split_long_section(text, heading):
    """Split oversized sections at paragraph or sentence boundaries."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n+", text)
        if paragraph.strip()
    ]

    chunks = []
    current = heading.strip()

    for paragraph in paragraphs:
        # Break a single oversized paragraph into sentences if possible.
        if len(paragraph) > CHUNK_SIZE:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current}\n{sentence}".strip()

            if len(candidate) <= CHUNK_SIZE:
                current = candidate
                continue

            if current.strip() and current.strip() != heading.strip():
                chunks.append(current.strip())

            # Repeat the heading so the chunk remains understandable.
            current = f"{heading}\n{sentence}".strip()

            # Extremely long sentences cannot be split reliably by
            # sentence boundaries, so split them at word boundaries.
            if len(current) > CHUNK_SIZE:
                words = sentence.split()
                current = heading

                for word in words:
                    candidate = f"{current} {word}".strip()

                    if len(candidate) > CHUNK_SIZE and current != heading:
                        chunks.append(current.strip())
                        current = f"{heading} {word}".strip()
                    else:
                        current = candidate

    if current.strip() and current.strip() != heading.strip():
        chunks.append(current.strip())

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run ingest.py first."
        )

    pages = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    all_chunks = []

    for page_number, page in enumerate(pages, start=1):
        page_title = page.get("title", "Untitled page")
        text = clean_text(page.get("text", ""), page_title)

        sections = extract_sections(text, page_title)
        page_chunks = []

        for heading, section_text in sections:
            section_chunks = split_long_section(section_text, heading)

            for chunk_text in section_chunks:
                if chunk_text.strip():
                    page_chunks.append({
                        "section_heading": heading,
                        "text": chunk_text,
                    })

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "chunk_id": (
                    f"page_{page_number:02d}_"
                    f"chunk_{chunk_number:03d}"
                ),
                "source_url": page["url"],
                "page_title": page_title,
                "section_heading": chunk["section_heading"],
                "chunk_index": chunk_number,
                "text": chunk["text"],
            })

        print(f"{page_title}: {len(page_chunks)} chunks")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nPages processed: {len(pages)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
