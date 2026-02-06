# Reproducibility Guide

This document provides step-by-step instructions to reproduce the results in the NSLP 2026 paper.

## Environment Setup

### Requirements

- Python 3.10+
- 8GB RAM minimum
- Internet connection (for LLM API calls during extraction)

### Installation

```bash
# Clone repository (update URL before publication)
git clone https://github.com/YOUR-USERNAME/orbis-kg.git
cd orbis-kg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### API Keys

For running extraction (optional - outputs are provided):

```bash
# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Reproducing Results

### Option A: Verify Provided Outputs (Recommended)

The repository includes all outputs. To verify they match paper claims:

```bash
# Validate all statistics
python scripts/validate_outputs.py

# Regenerate paper tables from outputs
python scripts/generate_paper_tables.py
```

Expected output:
```
[1] Validating Curated Graph...
  Entities: 106 (expected 106) [PASS]
  Edges: 2093 (expected 2093) [PASS]
...
ALL VALIDATIONS PASSED
```

### Option B: Full Pipeline Reproduction

**Note**: Requires raw article texts (see Data Acquisition below).

```bash
# 1. Extract knowledge graphs from articles
python src/extraction/extract_v2.py

# 2. Build curation recommendations
python src/curation/build_curation.py

# 3. Apply curation to create final graph
python src/curation/apply_curation.py

# 4. Create visualizations
python src/visualization/create_v2_viz.py

# 5. Generate analysis tables
python scripts/generate_paper_tables.py
```

## Data Acquisition

### Corpus

The raw article texts from *Orbis Tertius* are not included due to copyright.

To obtain them:
1. Visit: https://www.orbistertius.unlp.edu.ar/
2. Download articles from volumes 1-27 (1996-2024)
3. Convert PDFs to text format
4. Place in `data/raw/articles/` with structure:
   ```
   data/raw/articles/
   ├── vol01/
   │   ├── article1.txt
   │   └── ...
   └── vol27/
   ```

### Metadata

Topic assignments are provided:
- `data/topic_assignments.csv` - BERTopic cluster assignments for 472 articles

## Output Files

### Primary Outputs

| File | Description | Size |
|------|-------------|------|
| `outputs/graphs/v2_curated_graph.json` | Main knowledge graph | ~1 MB |
| `outputs/graphs/v2_entity_mapping.json` | Entity canonicalization | ~600 KB |
| `outputs/graphs/v2_provenance_index.json` | Edge provenance | ~800 KB |

### Analysis Outputs

| File | Paper Section |
|------|---------------|
| `outputs/comparison/ner_vs_llm_comparison.json` | Section 3.1 (NER baseline) |
| `outputs/tables/table_4_2_top_entities.csv` | Section 4.2 (Entity Composition) |
| `outputs/tables/table_4_4_author_network.csv` | Section 4.4 (Author Networks) |
| `outputs/tables/table_4_3_cultural_hypothesis.csv` | Section 4.3 (Cultural vs. Textual Framing) |
| `outputs/comparison/edc_analysis_like_kggen.json` | Appendix C |

### Visualizations

- `outputs/visualizations/v2_visualization.html` - Interactive graph explorer
- `outputs/visualizations/edc_visualization.html` - EDC comparison view

## Verification Checksums

To verify file integrity:

```bash
# Generate checksums
sha256sum outputs/graphs/*.json

# Expected (example):
# abc123... v2_curated_graph.json
# def456... v2_entity_mapping.json
```
