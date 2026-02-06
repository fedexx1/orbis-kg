# Data Documentation

## Corpus Overview

**Source**: Orbis Tertius - Revista de Teoria y Critica Literaria
**Publisher**: Universidad Nacional de La Plata, Argentina
**URL**: https://www.orbistertius.unlp.edu.ar/
**Language**: Spanish
**Domain**: Argentine and Latin American literary criticism

## Files in this Directory

### `topic_assignments.csv`

BERTopic cluster assignments for 472 articles.

| Column | Description |
|--------|-------------|
| `filename` | Article filename |
| `topic` | Assigned topic cluster (0-8) |
| `topic_label` | Human-readable topic label |
| `year` | Publication year |

**Topic Clusters**:
- Topic 0: General literary criticism
- Topic 1: Borges studies
- Topic 2: Poetry and poetics
- Topic 3: Argentine narrative
- Topic 4: Cultural studies
- Topic 5: Theory and methodology
- Topic 6: Latin American literature
- Topic 7: 19th century literature
- Topic 8: Contemporary fiction

### `corpus_metadata.json` (if present)

Article inventory with publication metadata.

## Raw Article Texts (Not Included)

Due to copyright restrictions, raw article texts are not included in this repository.

### How to Obtain

1. **Official Source**: Download from https://www.orbistertius.unlp.edu.ar/
2. **Format**: PDF articles, convert to plain text (.txt)
3. **Structure**: Organize by volume/year

### Expected Directory Structure

```
data/raw/articles/
├── vol01_1996/
│   ├── article1.txt
│   └── ...
├── vol02_1997/
│   └── ...
└── vol27_2024/
    └── ...
```

### Text Preprocessing

Articles should be:
- Plain UTF-8 text
- One article per file
- Minimal formatting (headers, footnotes removed)
- Spanish language preserved (no translation)

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Total articles | 472 |
| Date range | 1996-2024 |
| Volumes | 27 |
| Average length | ~5,000 words |
| Language | Spanish |

## Citation

When using this corpus, please cite:

```
Orbis Tertius: Revista de Teoria y Critica Literaria
Universidad Nacional de La Plata
ISSN: 1851-7811
https://www.orbistertius.unlp.edu.ar/
```

## License

- **Topic assignments**: CC-BY 4.0 (derived metadata)
- **Raw articles**: Copyright Orbis Tertius / UNLP (not redistributable)
