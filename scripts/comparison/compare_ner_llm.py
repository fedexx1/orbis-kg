"""
Large-Scale Comparison: spaCy NER vs LLM (KGGen) Extraction
============================================================

Compares entity extraction across 472 Orbis Tertius articles:
1. spaCy Spanish NER (es_core_news_lg) - traditional NLP
2. LLM extraction (Gemini via KGGen) - current approach

Generates statistics for NSLP 2026 paper.
"""

import json
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import sys

try:
    import spacy
except ImportError:
    print("spaCy not installed. Run: pip install spacy")
    print("Then: python -m spacy download es_core_news_lg")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ARTICLES_DIR = PROJECT_ROOT / "data" / "raw" / "articles_orbis"
KGGEN_DIR = PROJECT_ROOT / "outputs" / "v2" / "per_article"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(s):
    """Normalize string for comparison."""
    if not s:
        return ""
    nfd = unicodedata.normalize('NFD', s.lower().strip())
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def load_articles():
    """Load all article texts."""
    articles = {}
    for folder in ARTICLES_DIR.iterdir():
        if folder.is_dir():
            for f in folder.glob("*.txt"):
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        articles[f.stem] = fp.read()
                except UnicodeDecodeError:
                    try:
                        with open(f, 'r', encoding='latin-1') as fp:
                            articles[f.stem] = fp.read()
                    except Exception as e:
                        print(f"  Warning: Could not read {f.name}: {e}")
    return articles


def load_kggen_extractions():
    """Load all KGGen extraction results."""
    extractions = {}
    for f in KGGEN_DIR.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            # Use filename without extension as key
            key = f.stem
            extractions[key] = {
                'entities': data.get('entities', []),
                'relations': data.get('relations', []),
                'year': data.get('year', ''),
                'topic': data.get('topic', 0)
            }
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")
    return extractions


def run_spacy_ner(articles, nlp):
    """Run spaCy NER on all articles."""
    results = {}
    total = len(articles)

    print(f"\nProcessing {total} articles with spaCy...")

    for i, (filename, text) in enumerate(articles.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total} articles...")

        doc = nlp(text)
        entities = {
            'PER': [],  # Persons
            'LOC': [],  # Locations
            'ORG': [],  # Organizations
            'MISC': []  # Miscellaneous
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
            else:
                entities['MISC'].append(ent.text)

        results[filename] = entities

    return results


def classify_kggen_entity(entity):
    """Heuristically classify KGGen entity as person, place, concept, or other."""
    entity_lower = entity.lower()

    # Common concept indicators
    concept_indicators = [
        'ismo', 'ción', 'dad', 'aje', 'ura', 'logía', 'grafía',
        'literatura', 'escritura', 'lectura', 'narrativa', 'poesía',
        'realismo', 'modernismo', 'vanguardia', 'crítica', 'teoría',
        'ficción', 'novela', 'cuento', 'ensayo', 'género', 'estilo',
        'lenguaje', 'discurso', 'texto', 'obra', 'autor', 'lector'
    ]

    # Place indicators
    place_indicators = [
        'argentina', 'buenos aires', 'españa', 'francia', 'méxico',
        'américa', 'europa', 'ciudad', 'país', 'región', 'universidad'
    ]

    # Check for concepts
    for indicator in concept_indicators:
        if indicator in entity_lower:
            return 'CONCEPT'

    # Check for places
    for indicator in place_indicators:
        if indicator in entity_lower:
            return 'PLACE'

    # If it looks like a name (capitalized words, 2-4 words)
    words = entity.split()
    if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
        return 'PERSON'

    return 'OTHER'


def compare_extractions(spacy_results, kggen_results):
    """Compare spaCy and KGGen extractions."""

    # Aggregate statistics
    stats = {
        'total_articles': 0,
        'spacy': {
            'total_entities': 0,
            'unique_entities': set(),
            'by_type': Counter(),
            'per_article_avg': 0
        },
        'kggen': {
            'total_entities': 0,
            'unique_entities': set(),
            'by_type': Counter(),
            'per_article_avg': 0,
            'has_relations': 0
        },
        'overlap': {
            'exact_match': 0,
            'normalized_match': 0,
            'spacy_only': set(),
            'kggen_only': set(),
            'both': set()
        }
    }

    # Per-article comparison
    article_comparisons = []

    # Find matching articles
    spacy_keys = set(spacy_results.keys())
    kggen_keys = set(kggen_results.keys())

    # Try to match by normalized filename
    matched_pairs = []
    for sk in spacy_keys:
        sk_norm = normalize(sk)
        for kk in kggen_keys:
            kk_norm = normalize(kk)
            # Check if one contains the other or significant overlap
            if sk_norm in kk_norm or kk_norm in sk_norm:
                matched_pairs.append((sk, kk))
                break
            # Try matching by article ID pattern
            if 'orbis' in sk_norm and 'orbis' in kk_norm:
                # Extract article ID
                import re
                sk_match = re.search(r'n(\d+)a(\d+)', sk_norm)
                kk_match = re.search(r'n(\d+)a(\d+)', kk_norm)
                if sk_match and kk_match and sk_match.groups() == kk_match.groups():
                    matched_pairs.append((sk, kk))
                    break

    print(f"\nMatched {len(matched_pairs)} article pairs for comparison")

    all_spacy_entities = Counter()
    all_kggen_entities = Counter()
    all_kggen_concepts = Counter()

    for spacy_key, kggen_key in matched_pairs:
        stats['total_articles'] += 1

        spacy_data = spacy_results[spacy_key]
        kggen_data = kggen_results[kggen_key]

        # Collect spaCy entities
        spacy_entities = []
        for ent_type, ents in spacy_data.items():
            for ent in ents:
                spacy_entities.append(ent)
                stats['spacy']['by_type'][ent_type] += 1
                all_spacy_entities[normalize(ent)] += 1

        stats['spacy']['total_entities'] += len(spacy_entities)
        stats['spacy']['unique_entities'].update(normalize(e) for e in spacy_entities)

        # Collect KGGen entities
        kggen_entities = kggen_data['entities']
        for ent in kggen_entities:
            ent_type = classify_kggen_entity(ent)
            stats['kggen']['by_type'][ent_type] += 1
            all_kggen_entities[normalize(ent)] += 1
            if ent_type == 'CONCEPT':
                all_kggen_concepts[normalize(ent)] += 1

        stats['kggen']['total_entities'] += len(kggen_entities)
        stats['kggen']['unique_entities'].update(normalize(e) for e in kggen_entities)

        if kggen_data.get('relations'):
            stats['kggen']['has_relations'] += 1

        # Calculate overlap for this article
        spacy_norm = {normalize(e) for e in spacy_entities}
        kggen_norm = {normalize(e) for e in kggen_entities}

        overlap = spacy_norm & kggen_norm
        spacy_only = spacy_norm - kggen_norm
        kggen_only = kggen_norm - spacy_norm

        stats['overlap']['both'].update(overlap)
        stats['overlap']['spacy_only'].update(spacy_only)
        stats['overlap']['kggen_only'].update(kggen_only)

    # Calculate averages
    if stats['total_articles'] > 0:
        stats['spacy']['per_article_avg'] = stats['spacy']['total_entities'] / stats['total_articles']
        stats['kggen']['per_article_avg'] = stats['kggen']['total_entities'] / stats['total_articles']

    # Convert sets to counts for JSON serialization
    stats['spacy']['unique_count'] = len(stats['spacy']['unique_entities'])
    stats['kggen']['unique_count'] = len(stats['kggen']['unique_entities'])
    stats['overlap']['both_count'] = len(stats['overlap']['both'])
    stats['overlap']['spacy_only_count'] = len(stats['overlap']['spacy_only'])
    stats['overlap']['kggen_only_count'] = len(stats['overlap']['kggen_only'])

    # Top entities
    stats['top_spacy'] = all_spacy_entities.most_common(50)
    stats['top_kggen'] = all_kggen_entities.most_common(50)
    stats['top_concepts'] = all_kggen_concepts.most_common(50)

    # Jaccard similarity
    all_spacy_set = set(all_spacy_entities.keys())
    all_kggen_set = set(all_kggen_entities.keys())
    intersection = all_spacy_set & all_kggen_set
    union = all_spacy_set | all_kggen_set
    stats['jaccard_similarity'] = len(intersection) / len(union) if union else 0

    return stats, all_spacy_entities, all_kggen_entities, all_kggen_concepts


def main():
    print("=" * 70)
    print("Large-Scale Comparison: spaCy NER vs LLM (KGGen)")
    print("Orbis Tertius Corpus - 472 Articles")
    print("=" * 70)

    # Load KGGen extractions
    print("\n1. Loading KGGen extractions...")
    kggen_results = load_kggen_extractions()
    print(f"   Loaded {len(kggen_results)} KGGen extractions")

    # Load articles
    print("\n2. Loading articles...")
    articles = load_articles()
    print(f"   Loaded {len(articles)} articles")

    # Load spaCy
    print("\n3. Loading spaCy model...")
    try:
        nlp = spacy.load("es_core_news_lg")
        print("   Model: es_core_news_lg")
    except OSError:
        try:
            nlp = spacy.load("es_core_news_md")
            print("   Model: es_core_news_md (fallback)")
        except OSError:
            print("   ERROR: No Spanish model found!")
            print("   Run: python -m spacy download es_core_news_lg")
            return

    # Run spaCy NER
    print("\n4. Running spaCy NER on all articles...")
    spacy_results = run_spacy_ner(articles, nlp)
    print(f"   Processed {len(spacy_results)} articles")

    # Compare
    print("\n5. Comparing extractions...")
    stats, spacy_entities, kggen_entities, kggen_concepts = compare_extractions(
        spacy_results, kggen_results
    )

    # Print results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nArticles compared: {stats['total_articles']}")

    print(f"\n{'Metric':<40} {'spaCy NER':<15} {'LLM (KGGen)':<15}")
    print("-" * 70)
    print(f"{'Total entities extracted':<40} {stats['spacy']['total_entities']:<15,} {stats['kggen']['total_entities']:<15,}")
    print(f"{'Unique entities':<40} {stats['spacy']['unique_count']:<15,} {stats['kggen']['unique_count']:<15,}")
    print(f"{'Entities per article (avg)':<40} {stats['spacy']['per_article_avg']:<15.1f} {stats['kggen']['per_article_avg']:<15.1f}")

    print(f"\n{'Entity Types (spaCy)':<40}")
    print("-" * 40)
    for ent_type, count in stats['spacy']['by_type'].most_common():
        print(f"  {ent_type:<20} {count:>10,}")

    print(f"\n{'Entity Types (KGGen - heuristic)':<40}")
    print("-" * 40)
    for ent_type, count in stats['kggen']['by_type'].most_common():
        print(f"  {ent_type:<20} {count:>10,}")

    print(f"\n{'Overlap Analysis':<40}")
    print("-" * 40)
    print(f"  {'Entities found by both:':<35} {stats['overlap']['both_count']:>10,}")
    print(f"  {'spaCy only:':<35} {stats['overlap']['spacy_only_count']:>10,}")
    print(f"  {'KGGen only:':<35} {stats['overlap']['kggen_only_count']:>10,}")
    print(f"  {'Jaccard similarity:':<35} {stats['jaccard_similarity']:>10.1%}")

    print(f"\n{'KEY FINDING: Concepts extracted by LLM':<40}")
    print("-" * 40)
    concept_count = stats['kggen']['by_type'].get('CONCEPT', 0)
    print(f"  Total concepts: {concept_count:,}")
    print(f"  Top concepts:")
    for concept, count in stats['top_concepts'][:20]:
        print(f"    {concept:<35} ({count})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'articles_compared': stats['total_articles'],
        'spacy': {
            'total_entities': stats['spacy']['total_entities'],
            'unique_entities': stats['spacy']['unique_count'],
            'per_article_avg': stats['spacy']['per_article_avg'],
            'by_type': dict(stats['spacy']['by_type'])
        },
        'kggen': {
            'total_entities': stats['kggen']['total_entities'],
            'unique_entities': stats['kggen']['unique_count'],
            'per_article_avg': stats['kggen']['per_article_avg'],
            'by_type': dict(stats['kggen']['by_type']),
            'articles_with_relations': stats['kggen']['has_relations']
        },
        'overlap': {
            'both': stats['overlap']['both_count'],
            'spacy_only': stats['overlap']['spacy_only_count'],
            'kggen_only': stats['overlap']['kggen_only_count'],
            'jaccard_similarity': stats['jaccard_similarity']
        },
        'top_spacy_entities': stats['top_spacy'],
        'top_kggen_entities': stats['top_kggen'],
        'top_concepts': stats['top_concepts']
    }

    output_file = OUTPUT_DIR / "ner_vs_llm_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {output_file}")

    # Paper-ready summary
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    print(f"""
For NSLP 2026 paper:

| Metric | spaCy NER | LLM (KGGen) |
|--------|-----------|-------------|
| Total entities | {stats['spacy']['total_entities']:,} | {stats['kggen']['total_entities']:,} |
| Unique entities | {stats['spacy']['unique_count']:,} | {stats['kggen']['unique_count']:,} |
| Per article (avg) | {stats['spacy']['per_article_avg']:.1f} | {stats['kggen']['per_article_avg']:.1f} |
| Concepts extracted | 0 | {concept_count:,} |

Overlap: {stats['overlap']['both_count']:,} entities found by both methods
Jaccard similarity: {stats['jaccard_similarity']:.1%}

Key finding: LLM extraction captures {concept_count:,} conceptual terms
(e.g., {', '.join(c[0] for c in stats['top_concepts'][:5])})
that traditional NER cannot extract.
    """)


if __name__ == '__main__':
    main()
