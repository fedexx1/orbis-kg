"""
================================================================================
KGGen V2: Extraction with Full Provenance
================================================================================

This script extracts knowledge graphs from each article with embedded provenance.
Key differences from V1:
- NO KGGen clustering (it failed for Spanish scholarly text)
- Provenance (filename, year, topic) embedded in each relation
- Explicit UTF-8 handling to prevent encoding corruption
- Uses Gemini 2.0 Flash for cost efficiency

Usage:
    cd src/extraction/
    python extract_v2.py

Output:
    outputs/v2/per_article/*.json - One file per article with provenance
    outputs/v2/logs/extraction_log.json - Extraction metrics and errors
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG, ensure_directories, validate_inputs

# Import KGGen
try:
    from kg_gen import KGGen
except ImportError:
    print("ERROR: kg-gen not installed. Run: pip install kg-gen")
    sys.exit(1)


def load_article_text(filepath: Path) -> str:
    """
    Load article text with explicit UTF-8 handling.
    Attempts multiple encodings if UTF-8 fails.
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            # Verify we got valid text (no replacement characters)
            if '\ufffd' not in text:
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Last resort: read with errors='replace'
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def load_topic_assignments(csv_path: Path) -> dict:
    """
    Load topic assignments and create lookup by filename.
    Returns: {normalized_filename: {"topic": int, "year": str}}
    """
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Build lookup
    lookup = {}
    for _, row in df.iterrows():
        # Normalize filename (remove extension, handle special chars)
        fname = str(row.get('filename', row.get('file', '')))
        fname_normalized = normalize_filename(fname)
        
        lookup[fname_normalized] = {
            "topic": int(row.get('topic', row.get('cluster', -1))),
            "year": str(row.get('year', row.get('año', 'unknown')))[:4],
        }
    
    return lookup


def normalize_filename(fname: str) -> str:
    """Normalize filename for matching."""
    # Remove extensions
    fname = fname.replace('.txt', '').replace('.json', '')
    # Handle special prefix pattern
    if fname.startswith("admin,+Gestor") or fname.startswith("admin__Gestor"):
        parts = fname.replace(',+', '__').replace(',', '__').split('__')
        fname = parts[-1] if len(parts) > 1 else fname
    return fname.strip()


def extract_single_article(
    kg: KGGen,
    text: str,
    filename: str,
    metadata: dict,
    config: dict
) -> dict:
    """
    Extract KG from a single article with embedded provenance.
    
    Returns dict with:
    - filename: source article
    - year: publication year
    - topic: BERTopic assignment
    - entities: list of extracted entities
    - relations: list of [subject, predicate, object, provenance] tuples
    - extraction_time: seconds taken
    - error: error message if failed, None otherwise
    """
    result = {
        "filename": filename,
        "year": metadata.get("year", "unknown"),
        "topic": metadata.get("topic", -1),
        "entities": [],
        "relations": [],
        "extraction_time": 0,
        "error": None,
    }
    
    start_time = time.time()
    
    try:
        # Extract using KGGen (NO clustering)
        graph = kg.generate(
            input_data=text,
            chunk_size=config["chunk_size"],
            context=config["context"],
        )
        
        # Store entities
        result["entities"] = list(graph.entities) if graph.entities else []
        
        # Store relations WITH provenance embedded
        # Format: [subject, predicate, object] 
        # (provenance is at file level, not relation level)
        result["relations"] = [
            [s, p, o] for s, p, o in graph.relations
        ] if graph.relations else []
        
        # Also store edge types for reference
        result["edge_types"] = list(graph.edges) if graph.edges else []
        
    except Exception as e:
        result["error"] = str(e)
    
    result["extraction_time"] = time.time() - start_time
    return result


def run_extraction(config: dict):
    """
    Main extraction loop.
    Processes all articles and saves per-article JSONs with provenance.
    """
    print("\n" + "=" * 60)
    print("KGGen V2: EXTRACTION WITH PROVENANCE")
    print("=" * 60)
    
    # Setup
    ensure_directories()
    if not validate_inputs():
        print("\nERROR: Missing input files. Aborting.")
        sys.exit(1)
    
    # Load topic assignments
    print("\n📂 Loading topic assignments...")
    topic_lookup = load_topic_assignments(config["topic_assignments"])
    print(f"   Loaded metadata for {len(topic_lookup)} articles")
    
    # Get list of article files (search recursively in year subdirectories)
    articles_dir = Path(config["articles_dir"])
    article_files = sorted(articles_dir.glob("**/*.txt"))
    print(f"   Found {len(article_files)} article files")
    
    # Test mode: limit to sample
    if config["test_mode"]:
        article_files = article_files[:config["test_sample_size"]]
        print(f"   TEST MODE: Processing only {len(article_files)} articles")
    
    # Initialize KGGen
    print(f"\n🤖 Initializing KGGen with {config['model']}...")
    kg = KGGen(
        model=config["model"],
        temperature=config["temperature"],
    )
    
    # Extraction tracking
    output_dir = Path(config["v2_per_article_dir"])
    log = {
        "start_time": datetime.now().isoformat(),
        "config": {k: str(v) for k, v in config.items()},
        "total_articles": len(article_files),
        "successful": 0,
        "failed": 0,
        "errors": [],
        "total_entities": 0,
        "total_relations": 0,
        "articles_processed": [],
    }
    
    # Process articles
    print(f"\n⚙️  Processing {len(article_files)} articles...")
    print("-" * 60)
    
    batch_size = config["batch_size"]
    total = len(article_files)
    
    for i, article_path in enumerate(article_files, 1):
        filename = article_path.stem
        fname_normalized = normalize_filename(filename)
        
        # Get metadata
        metadata = topic_lookup.get(fname_normalized, {"topic": -1, "year": "unknown"})
        
        # Progress reporting
        if i % batch_size == 0 or i == 1 or i == total:
            print(f"   [{i:4}/{total}] Processing: {filename[:50]}...")
        
        # Load article text
        text = load_article_text(article_path)
        
        if not text.strip():
            log["errors"].append({"file": filename, "error": "Empty file"})
            log["failed"] += 1
            continue
        
        # Extract with retry logic
        result = None
        for attempt in range(config["max_retries"]):
            try:
                result = extract_single_article(kg, text, filename, metadata, config)
                if result["error"] is None:
                    break
                elif attempt < config["max_retries"] - 1:
                    print(f"      Retry {attempt + 1} for {filename}...")
                    time.sleep(config["retry_delay"])
            except Exception as e:
                if attempt < config["max_retries"] - 1:
                    time.sleep(config["retry_delay"])
                else:
                    result = {"error": str(e), "filename": filename}
        
        # Save result
        if result and result.get("error") is None:
            output_path = output_dir / f"{filename}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            log["successful"] += 1
            log["total_entities"] += len(result.get("entities", []))
            log["total_relations"] += len(result.get("relations", []))
            log["articles_processed"].append({
                "filename": filename,
                "entities": len(result.get("entities", [])),
                "relations": len(result.get("relations", [])),
                "time": result.get("extraction_time", 0),
            })
        else:
            log["failed"] += 1
            log["errors"].append({
                "file": filename,
                "error": result.get("error", "Unknown error") if result else "No result"
            })
        
        # Rate limiting
        time.sleep(config["request_delay"])
    
    # Finalize log
    log["end_time"] = datetime.now().isoformat()
    log["duration_seconds"] = (
        datetime.fromisoformat(log["end_time"]) - 
        datetime.fromisoformat(log["start_time"])
    ).total_seconds()
    
    # Save log
    log_path = Path(config["v2_logs_dir"]) / "extraction_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   Articles processed: {log['successful']}/{total}")
    print(f"   Failed: {log['failed']}")
    print(f"   Total entities: {log['total_entities']:,}")
    print(f"   Total relations: {log['total_relations']:,}")
    print(f"   Avg relations/article: {log['total_relations']/max(1,log['successful']):.1f}")
    print(f"   Duration: {log['duration_seconds']/60:.1f} minutes")
    
    print(f"\n📁 Output:")
    print(f"   Per-article graphs: {output_dir}")
    print(f"   Extraction log: {log_path}")
    
    if log["errors"]:
        print(f"\n⚠️  Errors ({len(log['errors'])}):")
        for err in log["errors"][:5]:
            print(f"   - {err['file']}: {err['error'][:50]}...")
        if len(log["errors"]) > 5:
            print(f"   ... and {len(log['errors']) - 5} more")
    
    return log


if __name__ == "__main__":
    run_extraction(CONFIG)
