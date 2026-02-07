# Comparing LLM-Based Knowledge Graph Extraction Approaches on Literary Studies in Spanish

**A Case Study on Orbis Tertius (1996-2024)**

This repository contains the code, data, and outputs for the paper submitted to NSLP 2026 Workshop at LREC 2026.

## Key Results

| Metric | KGGen | EDC |
|--------|-------|-----|
| Articles analyzed | 472 | 472 |
| Curated entities | 106 | 104 |
| Unique edges | 2,093 | 2,412 |
| Cultural/Textual ratio | 2.52x | 2.17x |
| Provenance coverage | 100% | - |

### Main Findings

1. **Cultural Framing Dominance**: Both methods show that *literatura* connects 2.2-2.5x more strongly to cultural concepts (*cultura*, *historia*, *arte*) than to textual concepts (*texto*, *escritura*), with statistical significance (p < .001).

2. **Method Convergence**: Despite fundamental architectural differences, KGGen and EDC converge on key findings, strengthening confidence that patterns reflect the corpus rather than extraction artifacts.

3. **Entity Composition Differences**: KGGen captures more proper names (40.7% vs 18.7%), while EDC captures more abstract concepts (42.8%) and preserves Spanish predicates.

## Repository Structure

```
publication/
├── README.md                 # This file
├── REPRODUCIBILITY.md        # Step-by-step reproduction guide
├── requirements.txt          # Python dependencies
│
├── curation/
│   └── entity_curation.xlsx  # Manual entity curation (106 entities)
│
├── data/
│   ├── README.md             # Corpus documentation
│   └── topic_assignments.csv # BERTopic cluster assignments (472 articles)
│
├── outputs/
│   ├── validation_report.json
│   ├── graphs/
│   │   ├── v2_curated_graph.json      # Main knowledge graph
│   │   ├── v2_entity_mapping.json     # Entity canonicalization
│   │   ├── v2_provenance_index.json   # Edge provenance
│   │   └── v2_metrics.json            # Extraction metrics
│   ├── comparison/
│   │   ├── edc_vs_kggen.json          # Raw extraction comparison
│   │   ├── edc_analysis_like_kggen.json # EDC with matched entities
│   │   └── ner_vs_llm_comparison.json # spaCy NER vs KGGen
│   ├── tables/
│   │   ├── summary_statistics.json
│   │   ├── table_4_2_entity_types.csv
│   │   ├── table_4_2_top_entities.csv
│   │   ├── table_4_3_concept_network.csv
│   │   ├── table_4_3_cultural_hypothesis.csv
│   │   ├── table_4_4_author_network.csv
│   │   └── table_appendix_c_edc_comparison.csv
│   └── visualizations/
│       ├── v2_visualization.html      # Interactive KGGen graph
│       └── edc_visualization.html     # Interactive EDC graph
│
├── scripts/
│   ├── validate_outputs.py            # Validate all outputs match paper
│   ├── generate_paper_tables.py       # Generate CSV tables from graphs
│   ├── analysis/
│   │   ├── cultural_hypothesis.py     # Cultural vs textual framing
│   │   └── chi_squared_test.py        # Statistical significance tests
│   ├── comparison/
│   │   ├── compare_ner_llm.py         # spaCy NER vs KGGen comparison
│   │   ├── compare_edc_kggen.py       # EDC vs KGGen comparison
│   │   └── analyze_edc_like_kggen.py  # EDC with matched entity set
│   └── export/
│       └── export_csv.py              # Export tables to CSV
│
└── src/
    ├── config.py                      # Configuration settings
    ├── extraction/
    │   ├── extract_v2.py              # KGGen extraction pipeline
    │   └── edc/
    │       ├── preprocess_orbis.py    # Preprocess articles for EDC
    │       └── run_edc.py             # EDC extraction pipeline
    ├── curation/
    │   ├── build_curation.py          # Generate curation recommendations
    │   └── apply_curation.py          # Apply entity canonicalization
    └── visualization/
        ├── create_v2_viz.py           # Generate KGGen visualization
        └── create_edc_viz.py          # Generate EDC visualization
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Outputs

Verify that all outputs match the claims in the paper:

```bash
python scripts/validate_outputs.py
```

### 3. Explore the Graph

Open `outputs/visualizations/v2_visualization.html` in a browser for interactive exploration.

### 4. Generate Paper Tables

Regenerate CSV tables from the graph data:

```bash
python scripts/generate_paper_tables.py
```

## Reproduction

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full reproduction instructions.

**Note**: Raw article texts are not included due to copyright. See `data/README.md` for acquisition instructions from the open-access journal.

## Output Files

### Primary Graph Outputs

| File | Description |
|------|-------------|
| `outputs/graphs/v2_curated_graph.json` | Main knowledge graph (106 entities, 2,093 edges) |
| `outputs/graphs/v2_entity_mapping.json` | Raw to canonical entity mappings |
| `outputs/graphs/v2_provenance_index.json` | Edge-to-article provenance |
| `outputs/graphs/v2_metrics.json` | Extraction statistics |

### Paper Tables (Section 4)

| File | Paper Reference |
|------|-----------------|
| `table_4_2_entity_types.csv` | Table 3: Entity type distribution |
| `table_4_2_top_entities.csv` | Table 4: Entity composition |
| `table_4_3_cultural_hypothesis.csv` | Table 5: Cultural vs textual framing |
| `table_4_4_author_network.csv` | Section 4.4: Author networks |
| `table_appendix_c_edc_comparison.csv` | EDC comparison data |

## License

- **Code**: MIT License
- **Data outputs**: CC-BY 4.0 (curated graph, tables, visualizations)
- **Raw articles**: Copyright Orbis Tertius / UNLP (not included)

