"""
================================================================================
KGGen V2: Apply Curated Mapping and Aggregate with Full Provenance
================================================================================

This script:
1. Loads all per-article graphs from V2 extraction
2. Loads your curated entity mapping (entity_curation.xlsx)
3. Applies fuzzy matching to map entities to canonical names
4. Aggregates all relations with complete provenance tracking
5. Outputs curated graph with 100% provenance coverage

Key difference from V1: Every edge is traceable to source articles.

Usage:
    cd src/curation/
    python apply_curation.py

Output:
    outputs/v2/aggregated/v2_raw_aggregated.json - All relations before filtering
    outputs/v2/curated/v2_curated_graph.json - Filtered to curated entities
    outputs/v2/curated/v2_provenance_index.json - Edge → articles mapping
    outputs/v2/curated/v2_metrics.json - Statistics
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG, ensure_directories


def load_curation_file(filepath: Path) -> dict:
    """
    Load curated entities from Excel file.
    
    Returns dict with:
    - entity_to_canonical: {original_name: canonical_name}
    - canonical_to_type: {canonical_name: entity_type}
    - keep_entities: set of canonical names to keep
    """
    df = pd.read_excel(filepath)
    
    entity_to_canonical = {}
    canonical_to_type = {}
    keep_entities = set()
    
    for _, row in df.iterrows():
        entity = str(row['entity']).strip()
        
        # Get canonical name (use entity itself if not specified)
        canonical = entity
        if pd.notna(row.get('canonical')) and str(row['canonical']).strip():
            canonical = str(row['canonical']).strip()
        
        # Get entity type
        etype = "unknown"
        if pd.notna(row.get('type')) and str(row['type']).strip():
            etype = str(row['type']).strip()
        
        # Check if should keep
        keep = str(row.get('keep', 'yes')).strip().lower() == 'yes'
        
        # Build mappings
        entity_to_canonical[entity] = canonical
        entity_to_canonical[entity.lower()] = canonical  # Also map lowercase
        entity_to_canonical[canonical] = canonical  # Map canonical to itself
        
        canonical_to_type[canonical] = etype
        
        if keep:
            keep_entities.add(canonical)
    
    return {
        "entity_to_canonical": entity_to_canonical,
        "canonical_to_type": canonical_to_type,
        "keep_entities": keep_entities,
    }


def fuzzy_match_entity(entity: str, curated_names: list, min_length: int = 4) -> str:
    """
    Attempt to match an entity to a curated canonical name.
    Uses substring containment for matching.
    
    Returns canonical name if matched, else None.
    """
    entity_lower = entity.lower()
    
    for curated in curated_names:
        if len(curated) < min_length:
            continue
        
        curated_lower = curated.lower()
        
        # Check if curated name is contained in entity
        if curated_lower in entity_lower:
            return curated
        
        # Check if entity is contained in curated name
        if len(entity) >= min_length and entity_lower in curated_lower:
            return curated
    
    return None


def load_per_article_graphs(graphs_dir: Path) -> list:
    """
    Load all per-article graphs with their embedded provenance.
    
    Returns list of dicts, each with:
    - filename, year, topic (provenance)
    - entities, relations (content)
    """
    graphs = []
    graph_files = sorted(graphs_dir.glob("*.json"))
    
    for gf in graph_files:
        try:
            with open(gf, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            
            # Ensure provenance fields exist
            if 'filename' not in graph:
                graph['filename'] = gf.stem
            
            graphs.append(graph)
        except Exception as e:
            print(f"   Warning: Could not load {gf.name}: {e}")
    
    return graphs


def build_entity_mapping(
    graphs: list,
    curation: dict,
    min_length: int = 4
) -> tuple:
    """
    Build comprehensive entity mapping using curated names + fuzzy matching.
    
    Returns:
    - entity_to_canonical: complete mapping for all entities
    - fuzzy_match_count: number of fuzzy matches made
    """
    # Start with curated mappings
    entity_to_canonical = curation["entity_to_canonical"].copy()
    curated_names = list(curation["keep_entities"])
    
    # Collect all unique entities from graphs
    all_entities = set()
    for graph in graphs:
        all_entities.update(graph.get('entities', []))
        for rel in graph.get('relations', []):
            if len(rel) >= 3:
                all_entities.add(rel[0])
                all_entities.add(rel[2])
    
    # Fuzzy match unmapped entities
    fuzzy_count = 0
    for entity in all_entities:
        if entity in entity_to_canonical:
            continue
        
        # Try fuzzy match
        match = fuzzy_match_entity(entity, curated_names, min_length)
        if match:
            entity_to_canonical[entity] = entity_to_canonical.get(match, match)
            fuzzy_count += 1
    
    return entity_to_canonical, fuzzy_count


def aggregate_with_provenance(
    graphs: list,
    entity_to_canonical: dict,
    keep_entities: set,
    include_uncurated: bool = False
) -> dict:
    """
    Aggregate all relations with complete provenance tracking.
    
    Returns dict with:
    - relations: list of {subject, predicate, object, provenance}
    - provenance_index: {edge_key: {articles, topics, years, predicates}}
    - entities: set of canonical entities
    - edge_types: set of predicate types
    """
    # Track unique relations and their provenance
    # Key: (canonical_subject, canonical_object) - undirected for aggregation
    edge_provenance = defaultdict(lambda: {
        "predicates": [],
        "articles": set(),
        "topics": set(),
        "years": set(),
        "directions": [],  # Track original direction
    })
    
    all_entities = set()
    all_predicates = set()
    total_relations = 0
    matched_relations = 0
    
    for graph in graphs:
        filename = graph.get('filename', 'unknown')
        year = graph.get('year', 'unknown')
        topic = graph.get('topic', -1)
        
        for rel in graph.get('relations', []):
            if len(rel) < 3:
                continue
            
            s, p, o = rel[0], rel[1], rel[2]
            total_relations += 1
            
            # Map to canonical names
            s_canon = entity_to_canonical.get(s, entity_to_canonical.get(s.lower(), s))
            o_canon = entity_to_canonical.get(o, entity_to_canonical.get(o.lower(), o))
            
            # Check if both entities are in keep list
            s_in_keep = s_canon in keep_entities
            o_in_keep = o_canon in keep_entities
            
            if include_uncurated or (s_in_keep and o_in_keep):
                # Skip self-loops
                if s_canon == o_canon:
                    continue
                
                matched_relations += 1
                
                # Create undirected edge key for aggregation
                edge_key = tuple(sorted([s_canon, o_canon]))
                
                # Store provenance
                edge_provenance[edge_key]["predicates"].append(p)
                edge_provenance[edge_key]["articles"].add(filename)
                edge_provenance[edge_key]["topics"].add(topic)
                edge_provenance[edge_key]["years"].add(year)
                edge_provenance[edge_key]["directions"].append((s_canon, o_canon))
                
                all_entities.add(s_canon)
                all_entities.add(o_canon)
                all_predicates.add(p)
    
    # Convert to serializable format
    relations = []
    provenance_index = {}
    
    for edge_key, prov in edge_provenance.items():
        # Determine primary direction (most common)
        dir_counts = Counter(prov["directions"])
        primary_dir = dir_counts.most_common(1)[0][0]
        
        relations.append({
            "subject": primary_dir[0],
            "object": primary_dir[1],
            "predicates": list(set(prov["predicates"])),
            "weight": len(prov["predicates"]),
            "articles": sorted(prov["articles"]),
            "topics": sorted(prov["topics"]),
            "years": sorted(prov["years"]),
        })
        
        # Create provenance index key
        prov_key = f"{primary_dir[0]}|{primary_dir[1]}"
        provenance_index[prov_key] = {
            "articles": sorted(prov["articles"]),
            "topics": sorted(prov["topics"]),
            "years": sorted(prov["years"]),
            "predicates": list(set(prov["predicates"])),
            "mention_count": len(prov["predicates"]),
        }
    
    return {
        "relations": relations,
        "provenance_index": provenance_index,
        "entities": all_entities,
        "edge_types": all_predicates,
        "stats": {
            "total_input_relations": total_relations,
            "matched_relations": matched_relations,
            "match_rate": matched_relations / max(1, total_relations),
        }
    }


def run_curation_pipeline(config: dict):
    """
    Main pipeline to apply curation and aggregate with provenance.
    """
    print("\n" + "=" * 60)
    print("KGGen V2: APPLY CURATION WITH FULL PROVENANCE")
    print("=" * 60)
    
    ensure_directories()
    
    # Load curation file
    print("\n📂 Loading curated entities...")
    curation = load_curation_file(config["curation_file"])
    print(f"   Curated entities to keep: {len(curation['keep_entities'])}")
    print(f"   Initial mappings: {len(curation['entity_to_canonical'])}")
    
    # Load per-article graphs
    print("\n📂 Loading per-article graphs...")
    graphs = load_per_article_graphs(config["v2_per_article_dir"])
    print(f"   Loaded {len(graphs)} article graphs")
    
    if not graphs:
        print("\n❌ ERROR: No graphs found. Run extract_v2.py first.")
        sys.exit(1)
    
    # Build comprehensive entity mapping
    print("\n🔗 Building entity mapping with fuzzy matching...")
    entity_to_canonical, fuzzy_count = build_entity_mapping(
        graphs, curation, config["fuzzy_match_min_length"]
    )
    print(f"   Total mappings: {len(entity_to_canonical)}")
    print(f"   Fuzzy matches added: {fuzzy_count}")
    
    # Aggregate with provenance
    print("\n📊 Aggregating relations with provenance...")
    result = aggregate_with_provenance(
        graphs,
        entity_to_canonical,
        curation["keep_entities"],
        include_uncurated=config["include_uncurated"]
    )
    
    print(f"   Input relations: {result['stats']['total_input_relations']:,}")
    print(f"   Matched relations: {result['stats']['matched_relations']:,}")
    print(f"   Match rate: {result['stats']['match_rate']:.1%}")
    print(f"   Unique edges: {len(result['relations'])}")
    print(f"   Unique entities: {len(result['entities'])}")
    
    # Verify 100% provenance
    edges_with_provenance = sum(
        1 for r in result['relations'] if r['articles']
    )
    provenance_rate = edges_with_provenance / max(1, len(result['relations']))
    print(f"   Provenance coverage: {provenance_rate:.1%}")
    
    # Build output graph
    curated_graph = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "source": "KGGen V2 with full provenance",
            "articles_processed": len(graphs),
            "curation_file": str(config["curation_file"]),
        },
        "entities": sorted(result['entities']),
        "entity_types": {
            e: curation["canonical_to_type"].get(e, "unknown")
            for e in result['entities']
        },
        "relations": result['relations'],
        "edge_types": sorted(result['edge_types']),
        "statistics": {
            "total_entities": len(result['entities']),
            "total_edges": len(result['relations']),
            "total_edge_types": len(result['edge_types']),
            "provenance_coverage": provenance_rate,
            "fuzzy_matches": fuzzy_count,
            **result['stats'],
        }
    }
    
    # Save outputs
    print("\n💾 Saving outputs...")
    
    # Curated graph
    curated_path = Path(config["v2_curated_dir"]) / "v2_curated_graph.json"
    with open(curated_path, 'w', encoding='utf-8') as f:
        json.dump(curated_graph, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Curated graph: {curated_path}")
    
    # Provenance index
    prov_path = Path(config["v2_curated_dir"]) / "v2_provenance_index.json"
    with open(prov_path, 'w', encoding='utf-8') as f:
        json.dump(result['provenance_index'], f, indent=2, ensure_ascii=False)
    print(f"   ✓ Provenance index: {prov_path}")
    
    # Entity mapping (for reference)
    mapping_path = Path(config["v2_curated_dir"]) / "v2_entity_mapping.json"
    # Only save mappings for entities that appear in graphs
    relevant_mappings = {
        k: v for k, v in entity_to_canonical.items()
        if v in curation["keep_entities"]
    }
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(relevant_mappings, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Entity mapping: {mapping_path}")
    
    # Metrics summary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "articles_processed": len(graphs),
        "curated_entities": len(curation["keep_entities"]),
        "fuzzy_matches": fuzzy_count,
        "total_entity_mappings": len(entity_to_canonical),
        "unique_entities_in_graph": len(result['entities']),
        "unique_edges": len(result['relations']),
        "unique_predicates": len(result['edge_types']),
        "provenance_coverage": provenance_rate,
        "input_relations": result['stats']['total_input_relations'],
        "matched_relations": result['stats']['matched_relations'],
    }
    metrics_path = Path(config["v2_curated_dir"]) / "v2_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics: {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CURATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Final Statistics:")
    print(f"   Curated entities: {len(result['entities'])}")
    print(f"   Aggregated edges: {len(result['relations'])}")
    print(f"   Provenance coverage: {provenance_rate:.1%}")
    print(f"\n📁 Outputs in: {config['v2_curated_dir']}")
    
    return curated_graph, result['provenance_index']


if __name__ == "__main__":
    run_curation_pipeline(CONFIG)
