"""
Preprocess Orbis Tertius articles for EDC input format.

This script converts article files into the format expected by EDC:
- One text unit per line in the main input file
- Metadata JSON mapping line indices to article provenance

Usage:
    python src/extraction/edc/preprocess_orbis.py --mode pilot
    python src/extraction/edc/preprocess_orbis.py --mode full

Note: Requires raw article texts in data/raw/articles/ directory.
See data/README.md for acquisition instructions.
"""

import os
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # publication/
ARTICLES_DIR = PROJECT_ROOT / "data" / "raw" / "articles"
TOPIC_FILE = PROJECT_ROOT / "data" / "topic_assignments.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "edc_input"

# Chunking configuration
MAX_CHUNK_CHARS = 3000  # Characters per chunk (conservative for API limits)
CHUNK_OVERLAP = 200     # Overlap between chunks
MIN_PARAGRAPH_CHARS = 100  # Minimum paragraph size to keep


def fix_ocr_hyphenation(text: str) -> str:
    """
    Fix OCR artifacts where words are split across lines with hyphens.

    Examples:
        "deliberada-\nmente" -> "deliberadamente"
        "Capi-\ntal" -> "Capital"
    """
    def rejoin_word(match):
        before = match.group(1)
        after = match.group(2)
        if before[-1].isdigit() or after[0].isupper() or len(after) <= 1:
            return match.group(0)
        return before + after

    pattern = r'(\w+)-\s*\n\s*(\w+)'
    return re.sub(pattern, rejoin_word, text)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form for consistent accent handling."""
    return unicodedata.normalize('NFC', text)


def load_topic_assignments() -> Dict[str, Dict]:
    """Load topic assignments from CSV."""
    topics = {}
    if TOPIC_FILE.exists():
        with open(TOPIC_FILE, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    filename, topic, year = parts[0], parts[1], parts[2]
                    topics[filename] = {"topic": topic, "year": year}
    return topics


def extract_year_from_path(filepath: Path) -> str:
    """Extract year from directory name like '1(1996)' -> '1996'."""
    parent_name = filepath.parent.name
    match = re.search(r'\((\d{4})\)', parent_name)
    return match.group(1) if match else "unknown"


def clean_text(text: str) -> str:
    """Clean article text while preserving content."""
    text = normalize_unicode(text)
    text = fix_ocr_hyphenation(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= MIN_PARAGRAPH_CHARS]


def chunk_article(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split article into chunks by paragraphs."""
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return [text[:max_chars]] if text else []

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        if current_length + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def collect_articles() -> List[Tuple[Path, str]]:
    """Collect all article paths with their content."""
    articles = []

    if not ARTICLES_DIR.exists():
        logger.error(f"Articles directory not found: {ARTICLES_DIR}")
        logger.info("See data/README.md for instructions on obtaining raw articles.")
        return articles

    for year_dir in sorted(ARTICLES_DIR.iterdir()):
        if not year_dir.is_dir():
            continue

        for article_file in sorted(year_dir.glob("*.txt")):
            try:
                try:
                    content = article_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    content = article_file.read_text(encoding='latin-1')

                content = clean_text(content)
                if content:
                    articles.append((article_file, content))
            except Exception as e:
                logger.warning(f"Error reading {article_file}: {e}")

    return articles


def preprocess_for_edc(
    mode: str = "pilot",
    pilot_size: int = 20,
    chunk_mode: str = "paragraph"
) -> Tuple[Path, Path]:
    """
    Main preprocessing function.

    Args:
        mode: "pilot" (20 articles) or "full" (all 472)
        pilot_size: Number of articles for pilot mode
        chunk_mode: How to split articles
            - "full": One line per article
            - "paragraph": Split by paragraphs, group into chunks
            - "fixed": Fixed character chunks with overlap

    Returns:
        Tuple of (input_file_path, metadata_file_path)
    """
    logger.info(f"Starting preprocessing: mode={mode}, chunk_mode={chunk_mode}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = collect_articles()
    if not articles:
        raise FileNotFoundError(
            "No articles found. See data/README.md for acquisition instructions."
        )

    logger.info(f"Found {len(articles)} articles")

    if mode == "pilot":
        articles = articles[:pilot_size]
        logger.info(f"Pilot mode: using first {pilot_size} articles")

    topic_map = load_topic_assignments()

    text_lines = []
    metadata = []

    for article_path, content in articles:
        filename = article_path.stem
        year = extract_year_from_path(article_path)
        topic_info = topic_map.get(filename, {"topic": "-1", "year": year})

        if chunk_mode == "full":
            chunks = [content[:MAX_CHUNK_CHARS * 2]]
        elif chunk_mode == "paragraph":
            chunks = chunk_article(content)
        else:
            chunks = [content[i:i+MAX_CHUNK_CHARS]
                     for i in range(0, len(content), MAX_CHUNK_CHARS - CHUNK_OVERLAP)]

        for chunk_idx, chunk in enumerate(chunks):
            line_idx = len(text_lines)
            text_lines.append(chunk.replace('\n', ' '))

            metadata.append({
                "line_index": line_idx,
                "filename": filename,
                "filepath": str(article_path.relative_to(PROJECT_ROOT)),
                "year": topic_info.get("year", year),
                "topic": topic_info.get("topic", "-1"),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
            })

    suffix = f"_{mode}" if mode == "pilot" else ""
    input_file = OUTPUT_DIR / f"orbis_articles{suffix}.txt"
    metadata_file = OUTPUT_DIR / f"orbis_metadata{suffix}.json"

    with open(input_file, 'w', encoding='utf-8') as f:
        for line in text_lines:
            f.write(line + '\n')

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "preprocessing_config": {
                "mode": mode,
                "chunk_mode": chunk_mode,
                "max_chunk_chars": MAX_CHUNK_CHARS,
                "total_articles": len(articles),
                "total_chunks": len(text_lines),
            },
            "chunks": metadata
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Created {input_file} with {len(text_lines)} lines")
    logger.info(f"Created {metadata_file}")

    return input_file, metadata_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Orbis articles for EDC")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                       help="Processing mode: pilot (20 articles) or full (all)")
    parser.add_argument("--chunk-mode", choices=["full", "paragraph", "fixed"],
                       default="paragraph", help="How to chunk articles")
    parser.add_argument("--pilot-size", type=int, default=20,
                       help="Number of articles for pilot mode")

    args = parser.parse_args()

    input_file, metadata_file = preprocess_for_edc(
        mode=args.mode,
        pilot_size=args.pilot_size,
        chunk_mode=args.chunk_mode
    )

    print(f"\nPreprocessing complete!")
    print(f"Input file: {input_file}")
    print(f"Metadata file: {metadata_file}")
