"""
================================================================================
KGGen V2: Build Native Entity Curation
================================================================================

This script analyzes V2's raw extracted entities (before curation) and 
generates a V2-native curation list for comparison with V1.

The problem: V1 curation was based on V1's entity frequencies, but V2's 
extraction may have different high-frequency entities.

This script:
1. Loads all V2 per-article graphs
2. Aggregates entity frequencies (degree, document frequency)
3. Compares with V1 curation list
4. Generates V2-native curation recommendation
5. Outputs Excel file for manual review

Usage:
    python build_v2_curation.py

Input:
    outputs/v2/per_article/*.json - V2 per-article graphs
    outputs/curation/entity_curation.xlsx - V1 curation (for comparison)

Output:
    outputs/v2/curation/v2_entity_analysis.xlsx - Full entity frequency analysis
    outputs/v2/curation/v2_curation_recommendation.xlsx - Recommended curation list
    outputs/v2/curation/v1_v2_curation_comparison.md - Comparison report
================================================================================
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
import re

# Configuration - adjust paths as needed
CONFIG = {
    "v2_per_article_dir": Path("outputs/v2/per_article"),
    "v1_curation_file": Path("outputs/curation/entity_curation.xlsx"),
    "output_dir": Path("outputs/v2/curation"),
    "min_degree": 50,  # Lower threshold to capture more candidates
    "min_doc_frequency": 5,  # Minimum articles mentioning entity
}


def sanitize_for_excel(text):
    """Remove characters that are illegal in Excel worksheets."""
    if not isinstance(text, str):
        return text
    # Remove control characters (0x00-0x1F except tab, newline, carriage return)
    # Also remove other problematic characters
    illegal_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
    return illegal_chars.sub('', text)


def load_v2_graphs(graphs_dir: Path) -> list:
    """Load all V2 per-article graphs."""
    graphs = []
    graph_files = sorted(graphs_dir.glob("*.json"))
    
    for gf in graph_files:
        try:
            with open(gf, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            graph['_filename'] = gf.stem
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Could not load {gf.name}: {e}")
    
    return graphs


def analyze_entities(graphs: list) -> pd.DataFrame:
    """
    Analyze entity frequencies across all V2 graphs.
    
    Returns DataFrame with columns:
    - entity: entity name
    - degree: total number of relation mentions
    - doc_frequency: number of articles mentioning entity
    - avg_degree_per_doc: average mentions per article
    """
    # Count entity occurrences
    entity_degree = Counter()
    entity_docs = defaultdict(set)
    
    for graph in graphs:
        filename = graph.get('_filename', graph.get('filename', 'unknown'))
        
        # Count from entities list
        for entity in graph.get('entities', []):
            entity_docs[entity].add(filename)
        
        # Count from relations (more accurate for degree)
        for rel in graph.get('relations', []):
            if len(rel) >= 3:
                s, p, o = rel[0], rel[1], rel[2]
                entity_degree[s] += 1
                entity_degree[o] += 1
                entity_docs[s].add(filename)
                entity_docs[o].add(filename)
    
    # Build DataFrame
    data = []
    for entity in set(entity_degree.keys()) | set(entity_docs.keys()):
        data.append({
            "entity": entity,
            "degree": entity_degree.get(entity, 0),
            "doc_frequency": len(entity_docs.get(entity, set())),
            "avg_degree_per_doc": (
                entity_degree.get(entity, 0) / max(1, len(entity_docs.get(entity, set())))
            ),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("degree", ascending=False).reset_index(drop=True)
    
    return df


def load_v1_curation(filepath: Path) -> dict:
    """Load V1 curation for comparison."""
    if not filepath.exists():
        return {"entities": set(), "canonical": {}, "types": {}}
    
    df = pd.read_excel(filepath)
    
    entities = set(df['entity'].str.strip())
    canonical = {}
    types = {}
    
    for _, row in df.iterrows():
        entity = str(row['entity']).strip()
        canon = str(row['canonical']).strip() if pd.notna(row.get('canonical')) else entity
        etype = str(row['type']).strip() if pd.notna(row.get('type')) else "unknown"
        
        canonical[entity] = canon
        types[canon] = etype
    
    return {
        "entities": entities,
        "canonical": canonical,
        "types": types,
        "canonical_set": set(canonical.values()),
    }


def generate_curation_recommendation(
    entity_df: pd.DataFrame,
    v1_curation: dict,
    min_degree: int,
    min_doc_freq: int
) -> pd.DataFrame:
    """
    Generate curation recommendation based on V2 frequencies.
    """
    # Filter by thresholds
    candidates = entity_df[
        (entity_df['degree'] >= min_degree) | 
        (entity_df['doc_frequency'] >= min_doc_freq)
    ].copy()
    
    # Check overlap with V1
    v1_entities = v1_curation['entities']
    v1_canonical = v1_curation['canonical_set']
    
    def check_v1_status(entity):
        if entity in v1_entities:
            return "in_v1_original"
        if entity in v1_canonical:
            return "in_v1_canonical"
        # Check fuzzy match
        for v1_e in v1_entities:
            if len(v1_e) >= 4 and v1_e.lower() in entity.lower():
                return f"fuzzy_match:{v1_e}"
            if len(entity) >= 4 and entity.lower() in v1_e.lower():
                return f"fuzzy_match:{v1_e}"
        return "NEW_IN_V2"
    
    candidates['v1_status'] = candidates['entity'].apply(check_v1_status)
    candidates['is_new'] = candidates['v1_status'] == "NEW_IN_V2"
    
    # Add empty columns for manual curation
    candidates['type'] = ""
    candidates['keep'] = "yes"
    candidates['canonical'] = ""
    candidates['notes'] = ""
    
    return candidates


def generate_comparison_report(
    entity_df: pd.DataFrame,
    v1_curation: dict,
    candidates: pd.DataFrame,
    output_path: Path
):
    """Generate markdown comparison report."""
    
    new_in_v2 = candidates[candidates['is_new']]
    in_v1 = candidates[~candidates['is_new']]
    
    report = f"""# V1 vs V2 Entity Curation Comparison

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | V1 | V2 |
|--------|-----|-----|
| Total unique entities | {len(v1_curation['entities'])} | {len(entity_df)} |
| Entities with degree ≥ 50 | {len(v1_curation['entities'])} | {len(candidates)} |
| Overlap with V1 | - | {len(in_v1)} |
| **New in V2** | - | **{len(new_in_v2)}** |

## Key Finding

V2 extraction produced **{len(new_in_v2)} entities** that meet the curation threshold 
but were NOT in V1's curation list. These may represent:
- Entities V1 missed
- Different extraction patterns
- Important additions to consider

## New V2 Entities (Top 30 by degree)

| Entity | Degree | Doc Freq | Notes |
|--------|--------|----------|-------|
"""
    
    for _, row in new_in_v2.head(30).iterrows():
        report += f"| {row['entity']} | {row['degree']} | {row['doc_frequency']} | |\n"
    
    report += f"""

## V1 Entities Not Found in V2 (Top Frequency)

These V1 curated entities did not appear prominently in V2:

"""
    
    v2_entities = set(entity_df['entity'])
    v1_missing = []
    for e in v1_curation['entities']:
        if e not in v2_entities:
            # Check if canonical is there
            canon = v1_curation['canonical'].get(e, e)
            if canon not in v2_entities:
                v1_missing.append(e)
    
    report += f"Total V1 entities not in V2: {len(v1_missing)}\n\n"
    for e in sorted(v1_missing)[:20]:
        report += f"- {e}\n"
    
    report += """

## Recommendations

1. **Review new V2 entities** - Add important ones to curation
2. **Check V1 entities missing in V2** - Verify extraction quality
3. **Consider merged curation** - Combine V1 + V2 top entities
4. **Re-run curation application** - After updating curation list

## Next Steps

1. Open `v2_curation_recommendation.xlsx`
2. Review `is_new` column for V2-specific entities
3. Assign types and keep/exclude decisions
4. Save and re-run apply_curation.py
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    print("=" * 60)
    print("Building V2-Native Entity Curation")
    print("=" * 60)
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Load V2 graphs
    print(f"\n📂 Loading V2 per-article graphs from {CONFIG['v2_per_article_dir']}...")
    graphs = load_v2_graphs(CONFIG["v2_per_article_dir"])
    
    if not graphs:
        print("❌ No graphs found. Check the path.")
        sys.exit(1)
    
    print(f"   Loaded {len(graphs)} article graphs")
    
    # Analyze entities
    print("\n📊 Analyzing entity frequencies...")
    entity_df = analyze_entities(graphs)
    print(f"   Total unique entities: {len(entity_df)}")
    print(f"   Entities with degree ≥ {CONFIG['min_degree']}: {len(entity_df[entity_df['degree'] >= CONFIG['min_degree']])}")
    
    # Save full analysis (sanitize entity names for Excel)
    analysis_path = CONFIG["output_dir"] / "v2_entity_analysis.xlsx"
    entity_df_clean = entity_df.copy()
    entity_df_clean['entity'] = entity_df_clean['entity'].apply(sanitize_for_excel)
    entity_df_clean.to_excel(analysis_path, index=False)
    print(f"   ✓ Full analysis saved: {analysis_path}")
    
    # Load V1 curation
    print(f"\n📂 Loading V1 curation from {CONFIG['v1_curation_file']}...")
    v1_curation = load_v1_curation(CONFIG["v1_curation_file"])
    print(f"   V1 curated entities: {len(v1_curation['entities'])}")
    
    # Generate recommendation
    print("\n🎯 Generating curation recommendation...")
    candidates = generate_curation_recommendation(
        entity_df, 
        v1_curation,
        CONFIG["min_degree"],
        CONFIG["min_doc_frequency"]
    )
    
    new_count = len(candidates[candidates['is_new']])
    print(f"   Candidates: {len(candidates)}")
    print(f"   New in V2: {new_count}")
    
    # Save recommendation (sanitize for Excel)
    rec_path = CONFIG["output_dir"] / "v2_curation_recommendation.xlsx"
    candidates_clean = candidates.copy()
    for col in ['entity', 'v1_status', 'notes']:
        if col in candidates_clean.columns:
            candidates_clean[col] = candidates_clean[col].apply(sanitize_for_excel)
    candidates_clean.to_excel(rec_path, index=False)
    print(f"   ✓ Recommendation saved: {rec_path}")
    
    # Generate comparison report
    print("\n📝 Generating comparison report...")
    report_path = CONFIG["output_dir"] / "v1_v2_curation_comparison.md"
    generate_comparison_report(entity_df, v1_curation, candidates, report_path)
    print(f"   ✓ Report saved: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📊 Key Findings:")
    print(f"   V2 unique entities: {len(entity_df)}")
    print(f"   Curation candidates: {len(candidates)}")
    print(f"   NEW entities in V2: {new_count}")
    print(f"\n📁 Outputs:")
    print(f"   {analysis_path}")
    print(f"   {rec_path}")
    print(f"   {report_path}")
    print(f"\n⚠️  Next: Review {rec_path.name} and update curation")


if __name__ == "__main__":
    main()
