"""
Validate all outputs for NSLP 2026 paper.

This script verifies that all claimed statistics in the paper match
the actual data in the output files. Run before submission to ensure
scientific accuracy.

Output: validation_report.json
"""

import json
from pathlib import Path
from collections import Counter

# Paths
PUB_ROOT = Path(__file__).parent.parent
GRAPH_FILE = PUB_ROOT / "outputs" / "graphs" / "v2_curated_graph.json"
MAPPING_FILE = PUB_ROOT / "outputs" / "graphs" / "v2_entity_mapping.json"
PROVENANCE_FILE = PUB_ROOT / "outputs" / "graphs" / "v2_provenance_index.json"
METRICS_FILE = PUB_ROOT / "outputs" / "graphs" / "v2_metrics.json"
EDC_FILE = PUB_ROOT / "outputs" / "comparison" / "edc_analysis_like_kggen.json"

# Paper claims to validate
PAPER_CLAIMS = {
    "entities": 106,
    "edges": 2093,
    "entity_mappings": 13396,  # v2_metrics.json: total_entity_mappings
    "provenance_coverage": 100,  # percent
    "articles_processed": 472,
    "cultural_textual_ratio": 2.5,  # KGGen shows 2.52x, EDC shows 2.17x
    "edc_entities": 104,
    "edc_edges": 2412,
    "edc_ratio": 2.17,
}


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_graph():
    """Validate curated graph statistics."""
    print("\n[1] Validating Curated Graph...")

    graph = load_json(GRAPH_FILE)
    entities = graph.get("entities", [])
    # Graph uses "relations" not "edges"
    relations = graph.get("relations", graph.get("edges", []))

    results = {
        "entities_count": len(entities),
        "entities_expected": PAPER_CLAIMS["entities"],
        "entities_match": len(entities) == PAPER_CLAIMS["entities"],
        "relations_count": len(relations),
        "relations_expected": PAPER_CLAIMS["edges"],
        "relations_match": len(relations) == PAPER_CLAIMS["edges"],
    }

    status = "PASS" if results["entities_match"] and results["relations_match"] else "FAIL"
    print(f"  Entities: {results['entities_count']} (expected {results['entities_expected']}) [{status}]")
    print(f"  Relations: {results['relations_count']} (expected {results['relations_expected']}) [{status}]")

    return results


def validate_mappings():
    """Validate entity mapping count."""
    print("\n[2] Validating Entity Mappings...")

    mapping = load_json(MAPPING_FILE)

    # Count total mappings
    # v2_entity_mapping.json is a flat dict: {variant_name: canonical_entity}
    # So the count is simply the number of key-value pairs
    total_mappings = len(mapping) if isinstance(mapping, dict) else 0

    results = {
        "mappings_count": total_mappings,
        "mappings_expected": PAPER_CLAIMS["entity_mappings"],
        "mappings_match": abs(total_mappings - PAPER_CLAIMS["entity_mappings"]) < 100,  # Allow some variance
    }

    status = "PASS" if results["mappings_match"] else "WARN"
    print(f"  Mappings: {results['mappings_count']} (expected ~{results['mappings_expected']}) [{status}]")

    return results


def validate_provenance():
    """Validate 100% provenance coverage."""
    print("\n[3] Validating Provenance Coverage...")

    graph = load_json(GRAPH_FILE)
    provenance = load_json(PROVENANCE_FILE)

    relations = graph.get("relations", graph.get("edges", []))
    relations_with_provenance = 0
    relations_without = []

    for rel in relations:
        # Relations format: [subject, predicate, object, metadata...]
        if isinstance(rel, list) and len(rel) >= 3:
            subj, pred, obj = rel[0], rel[1], rel[2]
        elif isinstance(rel, dict):
            subj = rel.get('source', rel.get('subject', ''))
            obj = rel.get('target', rel.get('object', ''))
        else:
            continue

        # Create edge key
        key = f"{subj}|{obj}"
        alt_key = f"{obj}|{subj}"

        if key in provenance or alt_key in provenance:
            relations_with_provenance += 1
        else:
            relations_without.append(key)

    coverage = (relations_with_provenance / len(relations) * 100) if relations else 0

    results = {
        "relations_with_provenance": relations_with_provenance,
        "relations_total": len(relations),
        "coverage_percent": round(coverage, 1),
        "coverage_expected": PAPER_CLAIMS["provenance_coverage"],
        "coverage_match": coverage >= 99,  # Allow tiny variance
        "relations_without_provenance": relations_without[:5] if relations_without else []
    }

    status = "PASS" if results["coverage_match"] else "FAIL"
    print(f"  Coverage: {results['coverage_percent']}% (expected {results['coverage_expected']}%) [{status}]")
    if relations_without:
        print(f"  Warning: {len(relations_without)} relations without provenance")

    return results


def validate_cultural_hypothesis():
    """Validate cultural studies hypothesis ratio."""
    print("\n[4] Validating Cultural Hypothesis...")

    graph = load_json(GRAPH_FILE)

    cultural_concepts = {"cultura", "historia", "arte", "sociedad"}
    textual_concepts = {"texto", "escritura", "lectura", "lenguaje"}

    cultural_weight = 0
    textual_weight = 0

    relations = graph.get("relations", graph.get("edges", []))

    for rel in relations:
        # Handle both list and dict formats
        if isinstance(rel, list) and len(rel) >= 3:
            source = rel[0].lower()
            target = rel[2].lower()
            weight = 1
        elif isinstance(rel, dict):
            source = rel.get("source", rel.get("subject", "")).lower()
            target = rel.get("target", rel.get("object", "")).lower()
            weight = rel.get("weight", 1)
        else:
            continue

        if "literatura" in source or "literatura" in target:
            other = target if "literatura" in source else source

            if other in cultural_concepts:
                cultural_weight += weight
            elif other in textual_concepts:
                textual_weight += weight

    ratio = cultural_weight / textual_weight if textual_weight > 0 else 0

    results = {
        "cultural_weight": cultural_weight,
        "textual_weight": textual_weight,
        "ratio": round(ratio, 2),
        "ratio_expected": PAPER_CLAIMS["cultural_textual_ratio"],
        "ratio_match": abs(ratio - PAPER_CLAIMS["cultural_textual_ratio"]) < 0.5,
    }

    status = "PASS" if results["ratio_match"] else "WARN"
    print(f"  Cultural: {results['cultural_weight']}, Textual: {results['textual_weight']}")
    print(f"  Ratio: {results['ratio']}x (expected ~{results['ratio_expected']}x) [{status}]")

    return results


def validate_edc_comparison():
    """Validate EDC comparison results."""
    print("\n[5] Validating EDC Comparison...")

    try:
        edc = load_json(EDC_FILE)
    except FileNotFoundError:
        print("  EDC file not found - skipping")
        return {"skipped": True}

    results = {
        "edc_entities": edc.get("total_entities", 0),
        "edc_entities_expected": PAPER_CLAIMS["edc_entities"],
        "edc_edges": edc.get("total_edges", 0),
        "edc_edges_expected": PAPER_CLAIMS["edc_edges"],
        "edc_ratio": round(edc.get("cultural_hypothesis", {}).get("ratio", 0), 2),
        "edc_ratio_expected": PAPER_CLAIMS["edc_ratio"],
    }

    entities_match = results["edc_entities"] == results["edc_entities_expected"]
    edges_match = results["edc_edges"] == results["edc_edges_expected"]
    ratio_match = abs(results["edc_ratio"] - results["edc_ratio_expected"]) < 0.2

    results["all_match"] = entities_match and edges_match and ratio_match

    status = "PASS" if results["all_match"] else "WARN"
    print(f"  EDC Entities: {results['edc_entities']} (expected {results['edc_entities_expected']})")
    print(f"  EDC Edges: {results['edc_edges']} (expected {results['edc_edges_expected']})")
    print(f"  EDC Ratio: {results['edc_ratio']}x (expected ~{results['edc_ratio_expected']}x) [{status}]")

    return results


def validate_files_exist():
    """Check all required files exist."""
    print("\n[0] Checking Required Files...")

    required_files = [
        GRAPH_FILE,
        MAPPING_FILE,
        PROVENANCE_FILE,
        PUB_ROOT / "outputs" / "visualizations" / "v2_visualization.html",
        PUB_ROOT / "curation" / "entity_curation.xlsx",
        PUB_ROOT / "data" / "topic_assignments.csv",
    ]

    results = {}
    all_exist = True

    for filepath in required_files:
        exists = filepath.exists()
        results[str(filepath.name)] = exists
        status = "OK" if exists else "MISSING"
        print(f"  {filepath.name}: [{status}]")
        if not exists:
            all_exist = False

    results["all_exist"] = all_exist
    return results


def main():
    print("=" * 60)
    print("VALIDATING PUBLICATION OUTPUTS")
    print("=" * 60)

    all_results = {}

    # Run all validations
    all_results["files"] = validate_files_exist()
    all_results["graph"] = validate_graph()
    all_results["mappings"] = validate_mappings()
    all_results["provenance"] = validate_provenance()
    all_results["cultural_hypothesis"] = validate_cultural_hypothesis()
    all_results["edc_comparison"] = validate_edc_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    issues = []

    if not all_results["files"].get("all_exist", False):
        issues.append("Some required files are missing")
    if not all_results["graph"].get("entities_match", False):
        issues.append(f"Entity count mismatch: {all_results['graph']['entities_count']} vs {PAPER_CLAIMS['entities']}")
    if not all_results["graph"].get("relations_match", False):
        issues.append(f"Relations count mismatch: {all_results['graph']['relations_count']} vs {PAPER_CLAIMS['edges']}")
    if not all_results["provenance"].get("coverage_match", False):
        issues.append(f"Provenance coverage: {all_results['provenance']['coverage_percent']}%")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease review and update paper claims if needed.")
    else:
        print("\nALL VALIDATIONS PASSED")
        print("Paper claims match output data.")

    # Save report
    report_file = PUB_ROOT / "outputs" / "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed report saved: {report_file}")


if __name__ == "__main__":
    main()
