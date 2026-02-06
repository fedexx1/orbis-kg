"""
Detailed comparison of EDC vs KGGen V2 extraction results.

This script provides a consistent methodology for evaluating extraction quality
by analyzing:
1. Entity composition (proper names vs concepts vs generic terms)
2. Predicate semantic categories (scholarly actions vs copulative vs generic)
3. Triplet specificity and informativeness
4. Coverage of known literary entities
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import unicodedata

# Paths
KGGEN_ROOT = Path(__file__).parent.parent
EDC_RESULTS = KGGEN_ROOT / "outputs" / "edc" / "full_20260116_161616" / "results" / "iter0" / "result_at_each_stage.json"
KGGEN_PER_ARTICLE = KGGEN_ROOT / "outputs" / "v2" / "per_article"
OUTPUT_DIR = KGGEN_ROOT / "outputs" / "comparison"

# Known literary entities (ground truth for recall)
KNOWN_AUTHORS = {
    "borges", "jorge luis borges", "cortázar", "julio cortázar", "saer", "juan josé saer",
    "puig", "manuel puig", "piglia", "ricardo piglia", "arlt", "roberto arlt",
    "sarmiento", "echeverría", "hernández", "lugones", "girondo", "gombrowicz",
    "bioy casares", "macedonio fernández", "leopoldo marechal", "lamborghini"
}

KNOWN_CRITICS = {
    "sarlo", "beatriz sarlo", "ludmer", "josefina ludmer", "jitrik", "noé jitrik",
    "viñas", "david viñas", "piglia", "gramuglio", "panesi", "amícola"
}

KNOWN_THEORISTS = {
    "barthes", "foucault", "derrida", "lacan", "freud", "benjamin", "adorno",
    "bajtín", "bajtin", "kristeva", "genette", "deleuze", "lotman", "shklovski",
    "tyniánov", "jameson"
}

KNOWN_CONCEPTS = {
    "parodia", "intertextualidad", "gauchesca", "realismo", "vanguardia",
    "postmodernismo", "criollismo", "fantástico", "testimonio", "nación",
    "identidad", "memoria", "exilio", "dictadura", "peronismo"
}

# Predicate categories for semantic analysis
SCHOLARLY_PREDICATES = {
    # Analysis actions - Spanish
    "analiza", "estudia", "examina", "investiga", "explora", "interpreta",
    "lee", "relee", "analizó", "estudió", "examinó",
    # Analysis actions - English
    "analyzes", "studies", "examines", "investigates", "explores", "interprets",
    "reads", "analyzed", "studied", "examined", "wrote about",
    # Argumentation - Spanish
    "argumenta", "sostiene", "propone", "plantea", "afirma", "sugiere",
    "defiende", "cuestiona", "critica", "rechaza", "objeta",
    # Argumentation - English
    "argues", "proposes", "suggests", "claims", "defends", "questions",
    "criticizes", "rejects", "objects",
    # Intellectual influence - Spanish
    "influye", "influyó", "influencia", "retoma", "dialoga con",
    "sigue", "se inspira en", "continúa", "desarrolla",
    # Intellectual influence - English
    "influences", "influenced", "follows", "continues", "develops",
    # Definition/Conceptualization - Spanish
    "define", "caracteriza", "conceptualiza", "teoriza", "elabora",
    "definió", "definido por",
    # Definition/Conceptualization - English
    "defines", "characterizes", "conceptualizes", "theorizes", "defined"
}

AUTHORSHIP_PREDICATES = {
    # Spanish
    "escribe", "escribió", "publica", "publicó", "autor de", "es autor de",
    "escribir", "obra de", "pertenece a", "creador de", "compone", "redacta",
    "publicado en", "publicada en", "publicado por", "publicada por",
    # English
    "wrote", "writes", "published", "publishes", "author of", "is author of",
    "written by", "is written by", "published by", "is published by",
    "published in", "wrote in", "wrote for", "belongs to", "creator of"
}

COPULATIVE_PREDICATES = {
    # Spanish
    "es", "fue", "son", "era", "ser", "siendo", "sido",
    # English
    "is", "was", "are", "were", "is a", "is an", "being", "been"
}

GENERIC_RELATION_PREDICATES = {
    # Spanish
    "tiene", "tenía", "tener", "posee", "presenta", "incluye",
    "contiene", "muestra", "exhibe", "incluyen", "tienen",
    # English
    "has", "have", "had", "possesses", "presents", "includes",
    "contains", "shows", "exhibits", "include"
}

MENTION_PREDICATES = {
    # Spanish
    "menciona", "cita", "refiere", "alude", "nombra", "señala",
    "mencionado", "citado", "referido",
    # English
    "mentions", "cites", "refers", "alludes", "names", "points",
    "mentioned", "cited", "referred", "mentioned in"
}

LOCATION_PREDICATES = {
    # Spanish
    "está en", "ubicado en", "localizado en", "situado en",
    # English
    "is in", "located in", "is from", "in", "is located in", "is part of"
}

RELATION_PREDICATES = {
    # Spanish
    "relacionado con", "vinculado con", "asociado con",
    # English
    "related to", "is related to", "associated with", "connected to"
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = unicodedata.normalize('NFC', text.lower().strip())
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def classify_entity(entity: str) -> str:
    """
    Classify entity into categories:
    - proper_name: Names of people, places, works, institutions
    - generic_concept: Abstract concepts, movements, genres
    - year: Temporal markers
    - identifier: Numeric or code-like entities
    - other: Everything else
    """
    entity_lower = normalize_text(entity)

    # Year detection
    if re.match(r'^\d{4}$', entity.strip()):
        return "year"
    if re.match(r'^(siglo|década|años)\s+', entity_lower):
        return "temporal"
    if re.match(r'^\d+s?$', entity.strip()):
        return "year"

    # Identifier detection (numbers, codes)
    if re.match(r'^[\d\.\-/]+$', entity.strip()):
        return "identifier"

    # Proper name heuristics
    # - Capitalized words (in original, not normalized)
    # - Contains proper name markers
    words = entity.split()
    if len(words) >= 1:
        # Check if most words are capitalized (suggests proper name)
        cap_words = sum(1 for w in words if w and w[0].isupper())
        if cap_words / len(words) >= 0.5 and len(entity) > 2:
            # Additional check: not a generic concept
            if entity_lower not in KNOWN_CONCEPTS:
                return "proper_name"

    # Generic concept heuristics
    if entity_lower in KNOWN_CONCEPTS:
        return "generic_concept"

    # Concept indicators
    concept_indicators = [
        "ismo", "ción", "dad", "miento", "tura", "logía", "grafía"
    ]
    for indicator in concept_indicators:
        if entity_lower.endswith(indicator):
            return "generic_concept"

    return "other"


def classify_predicate(predicate: str) -> str:
    """
    Classify predicate into semantic categories:
    - analysis: Scholarly analysis actions
    - authorship: Writing and creation
    - copulative: Being verbs
    - generic_relation: Generic possession/inclusion
    - mention: Citation and reference
    - location: Spatial relationships
    - relation: Generic associations
    - other_specific: Domain-specific but not categorized above
    """
    pred_lower = normalize_text(predicate)

    # Remove common prefixes/suffixes for matching
    pred_clean = re.sub(r'^(se\s+|no\s+)', '', pred_lower)

    # Check exact matches first for multi-word predicates
    if pred_lower in COPULATIVE_PREDICATES:
        return "copulative"
    if pred_lower in AUTHORSHIP_PREDICATES:
        return "authorship"
    if pred_lower in SCHOLARLY_PREDICATES:
        return "analysis"
    if pred_lower in GENERIC_RELATION_PREDICATES:
        return "generic_relation"
    if pred_lower in MENTION_PREDICATES:
        return "mention"
    if pred_lower in LOCATION_PREDICATES:
        return "location"
    if pred_lower in RELATION_PREDICATES:
        return "relation"

    # Check partial matches
    for word in COPULATIVE_PREDICATES:
        if pred_clean == word or pred_lower.startswith(word + " "):
            return "copulative"

    for word in AUTHORSHIP_PREDICATES:
        if word in pred_lower:
            return "authorship"

    for word in SCHOLARLY_PREDICATES:
        if word in pred_lower:
            return "analysis"

    for word in GENERIC_RELATION_PREDICATES:
        if word in pred_clean.split() or pred_clean.startswith(word):
            return "generic_relation"

    for word in MENTION_PREDICATES:
        if word in pred_lower:
            return "mention"

    for word in LOCATION_PREDICATES:
        if word in pred_lower:
            return "location"

    for word in RELATION_PREDICATES:
        if word in pred_lower:
            return "relation"

    return "other_specific"


def calculate_entity_recall(entities: Set[str], known_set: Set[str], name: str) -> Dict:
    """Calculate recall against known entities."""
    entities_lower = {normalize_text(e) for e in entities}

    found = set()
    for known in known_set:
        known_norm = normalize_text(known)
        # Check exact match or substring
        for entity in entities_lower:
            if known_norm in entity or entity in known_norm:
                found.add(known)
                break

    return {
        "category": name,
        "known_count": len(known_set),
        "found_count": len(found),
        "recall": len(found) / len(known_set) if known_set else 0,
        "found": sorted(found),
        "missing": sorted(known_set - found)
    }


def load_edc_results() -> Tuple[List[Tuple], Set[str], Set[str], Dict]:
    """Load and parse EDC results."""
    print("Loading EDC results...")
    with open(EDC_RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    triplets = []
    entities = set()
    predicates = set()
    predicate_definitions = {}

    for chunk in data:
        # Use schema_canonicalization for final triplets
        canon_triplets = chunk.get("schema_canonicalization", [])
        for triplet in canon_triplets:
            if len(triplet) >= 3:
                subj, pred, obj = triplet[0], triplet[1], triplet[2]
                triplets.append((subj, pred, obj))
                entities.add(subj)
                entities.add(obj)
                predicates.add(pred)

        # Extract predicate definitions
        schema_def = chunk.get("schema_definition", {})
        for key, value in schema_def.items():
            if key.startswith("Here are"):
                continue
            # Extract predicate name from key like "1. **vivió**"
            match = re.search(r'\*\*(.+?)\*\*', key)
            if match:
                pred_name = match.group(1)
                predicate_definitions[pred_name] = value

    return triplets, entities, predicates, predicate_definitions


def load_kggen_results() -> Tuple[List[Tuple], Set[str], Set[str]]:
    """Load and parse KGGen V2 results."""
    print("Loading KGGen V2 results...")

    triplets = []
    entities = set()
    predicates = set()

    for json_file in KGGEN_PER_ARTICLE.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add entities
        for entity in data.get("entities", []):
            entities.add(entity)

        # Add relations
        for relation in data.get("relations", []):
            if len(relation) >= 3:
                subj, pred, obj = relation[0], relation[1], relation[2]
                triplets.append((subj, pred, obj))
                entities.add(subj)
                entities.add(obj)
                predicates.add(pred)

    return triplets, entities, predicates


def analyze_system(name: str, triplets: List[Tuple], entities: Set[str], predicates: Set[str]) -> Dict:
    """Comprehensive analysis of an extraction system."""
    print(f"\nAnalyzing {name}...")

    # Entity classification
    entity_types = Counter(classify_entity(e) for e in entities)

    # Predicate classification
    predicate_types = Counter(classify_predicate(p) for p in predicates)

    # Triplet-level predicate usage
    triplet_pred_types = Counter(classify_predicate(t[1]) for t in triplets)

    # Entity recall against known entities
    author_recall = calculate_entity_recall(entities, KNOWN_AUTHORS, "Literary Authors")
    critic_recall = calculate_entity_recall(entities, KNOWN_CRITICS, "Critics")
    theorist_recall = calculate_entity_recall(entities, KNOWN_THEORISTS, "Theorists")
    concept_recall = calculate_entity_recall(entities, KNOWN_CONCEPTS, "Concepts")

    # Top predicates by frequency
    pred_freq = Counter(t[1] for t in triplets)

    # Unique subjects and objects
    subjects = set(t[0] for t in triplets)
    objects = set(t[2] for t in triplets)

    return {
        "name": name,
        "total_triplets": len(triplets),
        "unique_entities": len(entities),
        "unique_predicates": len(predicates),
        "unique_subjects": len(subjects),
        "unique_objects": len(objects),
        "entity_types": dict(entity_types),
        "predicate_types": dict(predicate_types),
        "triplet_predicate_distribution": dict(triplet_pred_types),
        "top_predicates": pred_freq.most_common(50),
        "recall": {
            "authors": author_recall,
            "critics": critic_recall,
            "theorists": theorist_recall,
            "concepts": concept_recall
        }
    }


def generate_report(edc_analysis: Dict, kggen_analysis: Dict, edc_pred_defs: Dict) -> str:
    """Generate comprehensive comparison report."""

    report = []
    report.append("=" * 80)
    report.append("EDC vs KGGen V2: DETAILED EXTRACTION QUALITY COMPARISON")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    report.append("## 1. SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"{'Metric':<30} {'EDC':>15} {'KGGen V2':>15}")
    report.append("-" * 60)
    report.append(f"{'Total triplets':<30} {edc_analysis['total_triplets']:>15,} {kggen_analysis['total_triplets']:>15,}")
    report.append(f"{'Unique entities':<30} {edc_analysis['unique_entities']:>15,} {kggen_analysis['unique_entities']:>15,}")
    report.append(f"{'Unique predicates':<30} {edc_analysis['unique_predicates']:>15,} {kggen_analysis['unique_predicates']:>15,}")
    report.append(f"{'Unique subjects':<30} {edc_analysis['unique_subjects']:>15,} {kggen_analysis['unique_subjects']:>15,}")
    report.append(f"{'Unique objects':<30} {edc_analysis['unique_objects']:>15,} {kggen_analysis['unique_objects']:>15,}")
    report.append("")

    # Entity composition
    report.append("## 2. ENTITY COMPOSITION")
    report.append("-" * 40)
    report.append("What types of entities does each system extract?")
    report.append("")
    report.append(f"{'Entity Type':<25} {'EDC':>12} {'%':>8} {'KGGen V2':>12} {'%':>8}")
    report.append("-" * 70)

    all_entity_types = set(edc_analysis['entity_types'].keys()) | set(kggen_analysis['entity_types'].keys())
    for etype in sorted(all_entity_types):
        edc_count = edc_analysis['entity_types'].get(etype, 0)
        kggen_count = kggen_analysis['entity_types'].get(etype, 0)
        edc_pct = 100 * edc_count / edc_analysis['unique_entities']
        kggen_pct = 100 * kggen_count / kggen_analysis['unique_entities']
        report.append(f"{etype:<25} {edc_count:>12,} {edc_pct:>7.1f}% {kggen_count:>12,} {kggen_pct:>7.1f}%")
    report.append("")

    report.append("INTERPRETATION:")
    report.append("- 'proper_name': People, places, works, institutions (higher = more grounded)")
    report.append("- 'generic_concept': Abstract notions, movements (higher = more conceptual)")
    report.append("- 'other': Unclassified entities (lower = cleaner extraction)")
    report.append("")

    # Predicate composition
    report.append("## 3. PREDICATE SEMANTIC COMPOSITION")
    report.append("-" * 40)
    report.append("What types of relationships does each system extract?")
    report.append("")
    report.append(f"{'Predicate Category':<25} {'EDC':>12} {'%':>8} {'KGGen V2':>12} {'%':>8}")
    report.append("-" * 70)

    all_pred_types = set(edc_analysis['triplet_predicate_distribution'].keys()) | set(kggen_analysis['triplet_predicate_distribution'].keys())
    for ptype in sorted(all_pred_types):
        edc_count = edc_analysis['triplet_predicate_distribution'].get(ptype, 0)
        kggen_count = kggen_analysis['triplet_predicate_distribution'].get(ptype, 0)
        edc_pct = 100 * edc_count / edc_analysis['total_triplets']
        kggen_pct = 100 * kggen_count / kggen_analysis['total_triplets']
        report.append(f"{ptype:<25} {edc_count:>12,} {edc_pct:>7.1f}% {kggen_count:>12,} {kggen_pct:>7.1f}%")
    report.append("")

    report.append("INTERPRETATION:")
    report.append("- 'analysis': Scholarly analysis verbs (estudia, analiza, interpreta)")
    report.append("- 'authorship': Writing/creation verbs (escribe, publica, es autor de)")
    report.append("- 'copulative': Being verbs (es, fue, son) - less informative")
    report.append("- 'generic_relation': Possession verbs (tiene, posee) - less specific")
    report.append("- 'other_specific': Domain-specific but uncategorized")
    report.append("")

    # Entity recall
    report.append("## 4. ENTITY RECALL (Against Known Literary Entities)")
    report.append("-" * 40)
    report.append("How well does each system capture known entities?")
    report.append("")

    for category in ['authors', 'critics', 'theorists', 'concepts']:
        edc_r = edc_analysis['recall'][category]
        kggen_r = kggen_analysis['recall'][category]
        report.append(f"### {edc_r['category']}")
        report.append(f"  EDC:    {edc_r['found_count']}/{edc_r['known_count']} = {edc_r['recall']:.1%}")
        report.append(f"  KGGen:  {kggen_r['found_count']}/{kggen_r['known_count']} = {kggen_r['recall']:.1%}")
        report.append(f"  EDC missing: {', '.join(edc_r['missing'][:5])}{'...' if len(edc_r['missing']) > 5 else ''}")
        report.append(f"  KGGen missing: {', '.join(kggen_r['missing'][:5])}{'...' if len(kggen_r['missing']) > 5 else ''}")
        report.append("")

    # Top predicates comparison
    report.append("## 5. TOP 30 PREDICATES")
    report.append("-" * 40)
    report.append("")
    report.append("### EDC Top Predicates:")
    for pred, count in edc_analysis['top_predicates'][:30]:
        cat = classify_predicate(pred)
        report.append(f"  {count:>6,}  [{cat:<18}] {pred}")
    report.append("")

    report.append("### KGGen V2 Top Predicates:")
    for pred, count in kggen_analysis['top_predicates'][:30]:
        cat = classify_predicate(pred)
        report.append(f"  {count:>6,}  [{cat:<18}] {pred}")
    report.append("")

    # EDC predicate definitions sample
    report.append("## 6. EDC PREDICATE DEFINITIONS (Sample)")
    report.append("-" * 40)
    report.append("EDC generates semantic definitions for predicates. Examples:")
    report.append("")
    for i, (pred, definition) in enumerate(list(edc_pred_defs.items())[:15]):
        report.append(f"  '{pred}': {definition[:100]}{'...' if len(definition) > 100 else ''}")
    report.append("")
    report.append("NOTE: KGGen V2 does not generate predicate definitions.")
    report.append("")

    # Quality assessment
    report.append("## 7. QUALITY ASSESSMENT FRAMEWORK")
    report.append("-" * 40)
    report.append("")
    report.append("### Metrics for Extraction Quality:")
    report.append("")
    report.append("1. PRECISION INDICATORS:")
    report.append("   - Lower 'copulative' % = more informative predicates")
    report.append("   - Lower 'generic_relation' % = more specific relationships")
    report.append("   - Higher 'proper_name' entity % = more grounded facts")
    report.append("")
    report.append("2. RECALL INDICATORS:")
    report.append("   - Author/Critic/Theorist recall against known lists")
    report.append("   - Concept coverage for domain terminology")
    report.append("")
    report.append("3. SCHOLARLY RELEVANCE:")
    report.append("   - Higher 'analysis' predicates = captures scholarly discourse")
    report.append("   - Higher 'authorship' predicates = captures attribution")
    report.append("")

    # Final comparison
    report.append("## 8. COMPARATIVE CONCLUSIONS")
    report.append("-" * 40)
    report.append("")

    # Calculate key metrics
    edc_copulative_pct = 100 * edc_analysis['triplet_predicate_distribution'].get('copulative', 0) / edc_analysis['total_triplets']
    kggen_copulative_pct = 100 * kggen_analysis['triplet_predicate_distribution'].get('copulative', 0) / kggen_analysis['total_triplets']

    edc_analysis_pct = 100 * edc_analysis['triplet_predicate_distribution'].get('analysis', 0) / edc_analysis['total_triplets']
    kggen_analysis_pct = 100 * kggen_analysis['triplet_predicate_distribution'].get('analysis', 0) / kggen_analysis['total_triplets']

    edc_proper_pct = 100 * edc_analysis['entity_types'].get('proper_name', 0) / edc_analysis['unique_entities']
    kggen_proper_pct = 100 * kggen_analysis['entity_types'].get('proper_name', 0) / kggen_analysis['unique_entities']

    report.append("### Predicate Specificity:")
    if edc_copulative_pct < kggen_copulative_pct:
        report.append(f"  ✓ EDC has FEWER copulative predicates ({edc_copulative_pct:.1f}% vs {kggen_copulative_pct:.1f}%)")
        report.append(f"    → EDC predicates are MORE specific")
    else:
        report.append(f"  ✓ KGGen has FEWER copulative predicates ({kggen_copulative_pct:.1f}% vs {edc_copulative_pct:.1f}%)")
        report.append(f"    → KGGen predicates are MORE specific")
    report.append("")

    report.append("### Scholarly Analysis Capture:")
    if edc_analysis_pct > kggen_analysis_pct:
        report.append(f"  ✓ EDC has MORE analysis predicates ({edc_analysis_pct:.1f}% vs {kggen_analysis_pct:.1f}%)")
        report.append(f"    → EDC better captures scholarly discourse")
    else:
        report.append(f"  ✓ KGGen has MORE analysis predicates ({kggen_analysis_pct:.1f}% vs {edc_analysis_pct:.1f}%)")
        report.append(f"    → KGGen better captures scholarly discourse")
    report.append("")

    report.append("### Entity Groundedness:")
    if edc_proper_pct > kggen_proper_pct:
        report.append(f"  ✓ EDC has MORE proper name entities ({edc_proper_pct:.1f}% vs {kggen_proper_pct:.1f}%)")
        report.append(f"    → EDC entities are MORE grounded in real-world referents")
    else:
        report.append(f"  ✓ KGGen has MORE proper name entities ({kggen_proper_pct:.1f}% vs {edc_proper_pct:.1f}%)")
        report.append(f"    → KGGen entities are MORE grounded in real-world referents")
    report.append("")

    # Author recall
    edc_author_recall = edc_analysis['recall']['authors']['recall']
    kggen_author_recall = kggen_analysis['recall']['authors']['recall']
    report.append("### Literary Author Recall:")
    if edc_author_recall > kggen_author_recall:
        report.append(f"  ✓ EDC has HIGHER author recall ({edc_author_recall:.1%} vs {kggen_author_recall:.1%})")
    else:
        report.append(f"  ✓ KGGen has HIGHER author recall ({kggen_author_recall:.1%} vs {edc_author_recall:.1%})")
    report.append("")

    report.append("### Key Differentiator: Predicate Definitions")
    report.append(f"  EDC provides {len(edc_pred_defs):,} predicate definitions")
    report.append("  KGGen provides 0 predicate definitions")
    report.append("  → EDC enables semantic interpretation of relationships")
    report.append("")

    return "\n".join(report)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    edc_triplets, edc_entities, edc_predicates, edc_pred_defs = load_edc_results()
    kggen_triplets, kggen_entities, kggen_predicates = load_kggen_results()

    print(f"\nEDC: {len(edc_triplets):,} triplets, {len(edc_entities):,} entities, {len(edc_predicates):,} predicates")
    print(f"KGGen: {len(kggen_triplets):,} triplets, {len(kggen_entities):,} entities, {len(kggen_predicates):,} predicates")

    # Analyze both systems
    edc_analysis = analyze_system("EDC", edc_triplets, edc_entities, edc_predicates)
    kggen_analysis = analyze_system("KGGen V2", kggen_triplets, kggen_entities, kggen_predicates)

    # Generate report
    report = generate_report(edc_analysis, kggen_analysis, edc_pred_defs)

    # Save report
    report_file = OUTPUT_DIR / "edc_kggen_detailed_comparison.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save raw analysis as JSON
    analysis_file = OUTPUT_DIR / "edc_kggen_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            "edc": edc_analysis,
            "kggen": kggen_analysis
        }, f, indent=2, ensure_ascii=False)
    print(f"Analysis JSON saved to: {analysis_file}")

    # Print report to console
    print("\n" + report)

    return edc_analysis, kggen_analysis


if __name__ == "__main__":
    main()
