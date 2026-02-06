"""
Analyze EDC graph using the same metrics as KGGen V2.

This script replicates the KGGen V2 analyses on EDC data to compare:
1. Cultural studies hypothesis (literatura ↔ cultura vs literatura ↔ texto)
2. Top entities by connectivity
3. Author-to-author network
4. Concept-to-concept network
5. Temporal evolution

IMPORTANT: For fair comparison, this script filters EDC data to only include
entities that match the KGGen V2 curated entity list (106 entities with degree >= 100).
This ensures we're measuring the same entities across both methodologies.

Output:
- outputs/comparison/edc_analysis_like_kggen.txt
- outputs/comparison/edc_analysis_like_kggen.json
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

# Paths
KGGEN_ROOT = Path(__file__).parent.parent
EDC_RESULTS = KGGEN_ROOT / "outputs" / "edc" / "full_20260116_161616" / "results" / "iter0" / "result_at_each_stage.json"
EDC_METADATA = KGGEN_ROOT / "edc" / "datasets" / "orbis_metadata.json"
KGGEN_CURATED = KGGEN_ROOT / "outputs" / "v2" / "curated" / "v2_curated_graph.json"
OUTPUT_DIR = KGGEN_ROOT / "outputs" / "comparison"

# Entity classification for EDC (Spanish entities)
KNOWN_AUTHORS = {
    "borges", "jorge luis borges", "cortázar", "julio cortázar", "cortazar",
    "saer", "juan josé saer", "puig", "manuel puig", "piglia", "ricardo piglia",
    "arlt", "roberto arlt", "sarmiento", "domingo sarmiento", "echeverría",
    "hernández", "josé hernández", "lugones", "leopoldo lugones", "girondo",
    "gombrowicz", "bioy casares", "bioy", "macedonio fernández", "macedonio",
    "marechal", "leopoldo marechal", "lamborghini", "aira", "césar aira",
    "ocampo", "silvina ocampo", "walsh", "rodolfo walsh", "quiroga",
    "güiraldes", "darío", "rubén darío", "rojas", "ricardo rojas",
    "martínez estrada", "groussac", "paul groussac"
}

KNOWN_CRITICS = {
    "sarlo", "beatriz sarlo", "ludmer", "josefina ludmer", "jitrik", "noé jitrik",
    "viñas", "david viñas", "gramuglio", "panesi", "jorge panesi", "amícola",
    "rama", "ángel rama", "prieto", "martín prieto"
}

KNOWN_THEORISTS = {
    "barthes", "roland barthes", "foucault", "michel foucault", "derrida",
    "jacques derrida", "lacan", "jacques lacan", "freud", "sigmund freud",
    "benjamin", "walter benjamin", "adorno", "bajtín", "bajtin", "mijail bajtín",
    "kristeva", "julia kristeva", "genette", "gérard genette", "deleuze",
    "gilles deleuze", "lotman", "iuri lotman", "shklovski", "tyniánov",
    "jameson", "fredric jameson", "nietzsche", "sartre"
}

# Concepts for cultural studies hypothesis
CULTURAL_CONCEPTS = {"cultura", "historia", "sociedad", "política", "arte", "memoria", "nación", "identidad"}
TEXTUAL_CONCEPTS = {"texto", "escritura", "lectura", "lenguaje", "narración", "narrativa", "discurso", "palabra"}
LITERATURE_TERMS = {"literatura", "literaria", "literario"}


def normalize(text: str) -> str:
    """Normalize text for matching."""
    return text.lower().strip()


def classify_entity(entity: str) -> str:
    """Classify entity into type."""
    entity_lower = normalize(entity)

    for author in KNOWN_AUTHORS:
        if author in entity_lower or entity_lower in author:
            return "Author (literary)"

    for critic in KNOWN_CRITICS:
        if critic in entity_lower or entity_lower in critic:
            return "Author (critic)"

    for theorist in KNOWN_THEORISTS:
        if theorist in entity_lower or entity_lower in theorist:
            return "Author (theory)"

    if entity_lower in CULTURAL_CONCEPTS:
        return "Concept: Cultural"

    if entity_lower in TEXTUAL_CONCEPTS:
        return "Concept: Literary/Textual"

    if entity_lower in LITERATURE_TERMS:
        return "Concept: Literature"

    # Place detection
    places = ["argentina", "buenos aires", "francia", "españa", "méxico", "europa",
              "latinoamérica", "américa latina", "uruguay", "chile", "cuba", "madrid",
              "barcelona", "parís", "montevideo"]
    for place in places:
        if place in entity_lower:
            return "Place"

    return "Other"


def load_kggen_curated_entities() -> Set[str]:
    """Load the curated entity list from KGGen V2."""
    print("Loading KGGen V2 curated entities...")

    with open(KGGEN_CURATED, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities = set()
    for entity in data.get("entities", []):
        entities.add(normalize(entity))

    print(f"  Loaded {len(entities)} curated entities")
    return entities


def load_edc_data() -> Tuple[List[Dict], Dict[int, Dict]]:
    """Load EDC results and metadata."""
    print("Loading EDC data...")

    with open(EDC_RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = {}
    if EDC_METADATA.exists():
        with open(EDC_METADATA, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            for chunk in meta.get("chunks", []):
                metadata[chunk["line_index"]] = chunk

    print(f"  Loaded {len(data)} chunks")
    return data, metadata


def match_to_curated(entity: str, curated_entities: Set[str]) -> str:
    """
    Try to match an EDC entity to a KGGen curated entity.
    Returns the matched curated entity or None if no match.
    """
    entity_lower = entity.lower().strip()

    # Direct match
    if entity_lower in curated_entities:
        return entity_lower

    # Try substring matching (entity contains curated or vice versa)
    for curated in curated_entities:
        # EDC entity contains curated entity name
        if curated in entity_lower and len(curated) >= 4:
            return curated
        # Curated entity contains EDC entity name
        if entity_lower in curated and len(entity_lower) >= 4:
            return curated

    return None


def build_aggregated_graph(data: List[Dict], metadata: Dict[int, Dict],
                          curated_entities: Set[str] = None) -> Dict:
    """
    Build aggregated graph from EDC triplets.

    If curated_entities is provided, only include edges where BOTH entities
    can be matched to a curated entity. This ensures fair comparison with KGGen.
    """
    print("Building aggregated graph...")
    if curated_entities:
        print(f"  Filtering to {len(curated_entities)} curated entities...")

    # Aggregate edges
    edge_data = defaultdict(lambda: {
        "weight": 0,
        "predicates": [],
        "years": set(),
        "articles": set()
    })

    entity_degrees = Counter()
    total_triplets = 0
    matched_triplets = 0

    for chunk in data:
        chunk_idx = chunk.get("index", 0)
        chunk_meta = metadata.get(chunk_idx, {})
        year = chunk_meta.get("year", "unknown")
        article = chunk_meta.get("filename", f"chunk_{chunk_idx}")

        triplets = chunk.get("schema_canonicalization", [])
        for triplet in triplets:
            if len(triplet) >= 3:
                subj, pred, obj = triplet[0], triplet[1], triplet[2]
                total_triplets += 1

                # Skip very short entities
                if len(subj) < 2 or len(obj) < 2:
                    continue

                # If curated filter is active, match to curated entities
                if curated_entities:
                    subj_matched = match_to_curated(subj, curated_entities)
                    obj_matched = match_to_curated(obj, curated_entities)

                    # Skip if either entity doesn't match a curated entity
                    if not subj_matched or not obj_matched:
                        continue

                    subj_norm = subj_matched
                    obj_norm = obj_matched
                    matched_triplets += 1
                else:
                    subj_norm = normalize(subj)
                    obj_norm = normalize(obj)

                edge_key = (subj_norm, obj_norm)
                edge_data[edge_key]["weight"] += 1
                if pred not in edge_data[edge_key]["predicates"]:
                    edge_data[edge_key]["predicates"].append(pred)
                edge_data[edge_key]["years"].add(year)
                edge_data[edge_key]["articles"].add(article)

                entity_degrees[subj_norm] += 1
                entity_degrees[obj_norm] += 1

    print(f"  Total triplets processed: {total_triplets}")
    if curated_entities:
        print(f"  Triplets matching curated entities: {matched_triplets} ({100*matched_triplets/total_triplets:.1f}%)")
    print(f"  Unique entities in filtered graph: {len(entity_degrees)}")
    print(f"  Unique edges in filtered graph: {len(edge_data)}")

    return {
        "edge_data": dict(edge_data),
        "entity_degrees": entity_degrees
    }


def analyze_cultural_hypothesis(edge_data: Dict, entity_degrees: Counter) -> Dict:
    """
    Analyze cultural studies hypothesis.
    Compare: literatura ↔ cultural concepts vs literatura ↔ textual concepts
    """
    print("\nAnalyzing cultural studies hypothesis...")

    # Find literatura-related entities
    lit_entities = [e for e in entity_degrees.keys()
                    if any(lit in e for lit in LITERATURE_TERMS)]

    print(f"  Literature-related entities: {lit_entities[:5]}...")

    cultural_weight = 0
    cultural_connections = []
    textual_weight = 0
    textual_connections = []

    for (subj, obj), data in edge_data.items():
        weight = data["weight"]

        # Check if one end is literatura-related
        is_lit_subj = any(lit in subj for lit in LITERATURE_TERMS)
        is_lit_obj = any(lit in obj for lit in LITERATURE_TERMS)

        if is_lit_subj or is_lit_obj:
            other = obj if is_lit_subj else subj

            # Check if other end is cultural or textual concept
            if other in CULTURAL_CONCEPTS:
                cultural_weight += weight
                cultural_connections.append((subj, obj, weight))
            elif other in TEXTUAL_CONCEPTS:
                textual_weight += weight
                textual_connections.append((subj, obj, weight))

    ratio = cultural_weight / textual_weight if textual_weight > 0 else float('inf')

    return {
        "cultural_weight": cultural_weight,
        "textual_weight": textual_weight,
        "ratio": ratio,
        "cultural_connections": sorted(cultural_connections, key=lambda x: -x[2])[:10],
        "textual_connections": sorted(textual_connections, key=lambda x: -x[2])[:10]
    }


def analyze_temporal_evolution(edge_data: Dict) -> Dict:
    """Analyze cultural/textual ratio by decade."""
    print("\nAnalyzing temporal evolution...")

    decades = {
        "1996-1999": (1996, 1999),
        "2000-2009": (2000, 2009),
        "2010-2019": (2010, 2019),
        "2020-2024": (2020, 2024)
    }

    results = {}

    for decade_name, (start, end) in decades.items():
        cultural_weight = 0
        textual_weight = 0

        for (subj, obj), data in edge_data.items():
            # Check years
            years = [int(y) for y in data["years"] if y.isdigit()]
            if not any(start <= y <= end for y in years):
                continue

            weight = data["weight"]

            # Check if literatura-related
            is_lit_subj = any(lit in subj for lit in LITERATURE_TERMS)
            is_lit_obj = any(lit in obj for lit in LITERATURE_TERMS)

            if is_lit_subj or is_lit_obj:
                other = obj if is_lit_subj else subj

                if other in CULTURAL_CONCEPTS:
                    cultural_weight += weight
                elif other in TEXTUAL_CONCEPTS:
                    textual_weight += weight

        ratio = cultural_weight / textual_weight if textual_weight > 0 else float('inf')
        results[decade_name] = {
            "cultural": cultural_weight,
            "textual": textual_weight,
            "ratio": ratio
        }

    return results


def analyze_author_network(edge_data: Dict, entity_degrees: Counter) -> List[Dict]:
    """Find strongest author-to-author connections."""
    print("\nAnalyzing author-to-author network...")

    author_connections = []

    for (subj, obj), data in edge_data.items():
        subj_type = classify_entity(subj)
        obj_type = classify_entity(obj)

        # Both must be authors (literary, critic, or theory)
        if "Author" in subj_type and "Author" in obj_type:
            author_connections.append({
                "source": subj,
                "target": obj,
                "weight": data["weight"],
                "predicates": data["predicates"][:3],
                "source_type": subj_type,
                "target_type": obj_type
            })

    # Sort by weight
    author_connections.sort(key=lambda x: -x["weight"])

    return author_connections[:30]


def analyze_concept_network(edge_data: Dict) -> List[Dict]:
    """Find strongest concept-to-concept connections."""
    print("\nAnalyzing concept-to-concept network...")

    all_concepts = CULTURAL_CONCEPTS | TEXTUAL_CONCEPTS | LITERATURE_TERMS
    concept_connections = []

    for (subj, obj), data in edge_data.items():
        if subj in all_concepts and obj in all_concepts:
            concept_connections.append({
                "source": subj,
                "target": obj,
                "weight": data["weight"],
                "predicates": data["predicates"][:3]
            })

    concept_connections.sort(key=lambda x: -x["weight"])
    return concept_connections[:20]


def get_top_entities(entity_degrees: Counter, n: int = 20) -> List[Dict]:
    """Get top entities by degree with classification."""
    top = []
    for entity, degree in entity_degrees.most_common(n):
        top.append({
            "entity": entity,
            "degree": degree,
            "type": classify_entity(entity)
        })
    return top


def generate_report(results: Dict) -> str:
    """Generate text report."""
    report = []
    report.append("=" * 80)
    report.append("EDC ANALYSIS (Same Metrics as KGGen V2)")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("## SUMMARY")
    report.append(f"Total entities: {results['total_entities']:,}")
    report.append(f"Total edges: {results['total_edges']:,}")
    report.append("")

    # Top entities
    report.append("## TOP ENTITIES BY CONNECTIVITY")
    report.append("-" * 60)
    report.append(f"{'Rank':<5} {'Entity':<35} {'Degree':<10} {'Type':<20}")
    report.append("-" * 60)
    for i, ent in enumerate(results["top_entities"], 1):
        report.append(f"{i:<5} {ent['entity'][:35]:<35} {ent['degree']:<10} {ent['type']:<20}")
    report.append("")

    # Cultural hypothesis
    ch = results["cultural_hypothesis"]
    report.append("## CULTURAL STUDIES HYPOTHESIS")
    report.append("-" * 60)
    report.append(f"Literatura ↔ Cultural concepts weight: {ch['cultural_weight']}")
    report.append(f"Literatura ↔ Textual concepts weight: {ch['textual_weight']}")
    report.append(f"RATIO (Cultural/Textual): {ch['ratio']:.2f}x")
    report.append("")
    report.append("Top cultural connections:")
    for subj, obj, w in ch["cultural_connections"][:5]:
        report.append(f"  {subj} ↔ {obj}: {w}")
    report.append("")
    report.append("Top textual connections:")
    for subj, obj, w in ch["textual_connections"][:5]:
        report.append(f"  {subj} ↔ {obj}: {w}")
    report.append("")

    # Temporal evolution
    report.append("## TEMPORAL EVOLUTION OF FRAMING")
    report.append("-" * 60)
    report.append(f"{'Period':<15} {'Cultural':<12} {'Textual':<12} {'Ratio':<10}")
    report.append("-" * 60)
    for period, data in results["temporal_evolution"].items():
        ratio_str = f"{data['ratio']:.1f}x" if data['ratio'] != float('inf') else "inf"
        report.append(f"{period:<15} {data['cultural']:<12} {data['textual']:<12} {ratio_str:<10}")
    report.append("")

    # Author network
    report.append("## AUTHOR-TO-AUTHOR NETWORK")
    report.append("-" * 60)
    for conn in results["author_network"][:15]:
        preds = ", ".join(conn["predicates"][:2])
        report.append(f"{conn['source']} ↔ {conn['target']}: {conn['weight']} [{preds}]")
    report.append("")

    # Concept network
    report.append("## CONCEPT-TO-CONCEPT NETWORK")
    report.append("-" * 60)
    for conn in results["concept_network"][:10]:
        report.append(f"{conn['source']} ↔ {conn['target']}: {conn['weight']}")
    report.append("")

    # Comparison note
    report.append("=" * 80)
    report.append("COMPARE WITH KGGEN V2 RESULTS IN paper_draft.md")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    print("\n" + "=" * 60)
    print("EDC ANALYSIS: Replicating KGGen V2 Metrics")
    print("=" * 60)

    # Load KGGen curated entities for fair comparison
    curated_entities = load_kggen_curated_entities()

    # Load data
    data, metadata = load_edc_data()

    # Build graph FILTERED to curated entities only
    graph = build_aggregated_graph(data, metadata, curated_entities)
    edge_data = graph["edge_data"]
    entity_degrees = graph["entity_degrees"]

    # Run analyses
    results = {
        "total_entities": len(entity_degrees),
        "total_edges": len(edge_data),
        "top_entities": get_top_entities(entity_degrees, 20),
        "cultural_hypothesis": analyze_cultural_hypothesis(edge_data, entity_degrees),
        "temporal_evolution": analyze_temporal_evolution(edge_data),
        "author_network": analyze_author_network(edge_data, entity_degrees),
        "concept_network": analyze_concept_network(edge_data)
    }

    # Generate report
    report = generate_report(results)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = OUTPUT_DIR / "edc_analysis_like_kggen.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {report_file}")

    # Save JSON (convert sets to lists for JSON serialization)
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        return obj

    json_file = OUTPUT_DIR / "edc_analysis_like_kggen.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(convert_sets(results), f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_file}")

    # Print report
    print("\n" + report)


if __name__ == "__main__":
    main()
