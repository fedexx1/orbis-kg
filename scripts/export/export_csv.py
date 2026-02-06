"""
Export Topic and Temporal Analysis to CSV for publication use.
Creates Excel-ready CSV files from the JSON analysis data.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
analysis_dir = Path("outputs/v2/analysis")
output_dir = analysis_dir / "csv"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("EXPORTING ANALYSIS DATA TO CSV")
print("=" * 60)

# =============================================================================
# TOPIC ANALYSIS CSVs
# =============================================================================

print("\n📊 Loading topic analysis data...")
with open(analysis_dir / "topic_analysis_report.json", 'r', encoding='utf-8') as f:
    topic_data = json.load(f)

# 1. Topic Overview
print("   Creating topic_overview.csv...")
topic_overview = []
for tid, sg in topic_data['subgraphs'].items():
    stats = sg['statistics']
    topic_overview.append({
        'topic_id': tid,
        'label': sg['label'],
        'unique_entities': stats['unique_entities'],
        'unique_edges': stats['unique_edges'],
        'total_relations': stats['total_relations'],
        'avg_degree': stats['avg_degree'],
        'density': stats['density'],
        'top_entities': ', '.join(stats['top_entities'][:5]),
    })
pd.DataFrame(topic_overview).to_csv(output_dir / "topic_overview.csv", index=False, encoding='utf-8-sig')

# 2. Topic Entities (all entities per topic)
print("   Creating topic_entities.csv...")
topic_entities = []
for tid, sg in topic_data['subgraphs'].items():
    for node in sg['nodes']:
        topic_entities.append({
            'topic_id': tid,
            'topic_label': sg['label'],
            'entity': node['id'],
            'type': node['type'],
            'degree': node['degree'],
            'weighted_degree': node['weighted'],
            'articles': node.get('articles', 0),
        })
pd.DataFrame(topic_entities).to_csv(output_dir / "topic_entities.csv", index=False, encoding='utf-8-sig')

# 3. Jaccard Similarity
print("   Creating topic_jaccard_similarity.csv...")
jaccard_data = []
for pair, value in topic_data['cross_topic_metrics']['jaccard_similarity'].items():
    # Handle negative topic IDs (e.g., '-1-0' or '0-1')
    # Split from the right, taking last element as t2
    parts = pair.rsplit('-', 1)
    if len(parts) == 2:
        t1, t2 = parts[0], parts[1]
    else:
        t1, t2 = pair, ''
    jaccard_data.append({
        'topic_1': t1,
        'topic_2': t2,
        'jaccard_index': value,
    })
pd.DataFrame(jaccard_data).to_csv(output_dir / "topic_jaccard_similarity.csv", index=False, encoding='utf-8-sig')

# 4. Entity-Topic Count (how many topics each entity appears in)
print("   Creating entity_topic_count.csv...")
entity_topic_count = []
for entity, count in topic_data['cross_topic_metrics']['entity_topic_count'].items():
    entity_topic_count.append({
        'entity': entity,
        'topic_count': count,
    })
pd.DataFrame(entity_topic_count).sort_values('topic_count', ascending=False).to_csv(
    output_dir / "entity_topic_count.csv", index=False, encoding='utf-8-sig')

# 5. Type Distribution by Topic
print("   Creating topic_type_distribution.csv...")
type_dist = []
for tid, prefs in topic_data['cross_topic_metrics']['type_preferences'].items():
    for entity_type, proportion in prefs.items():
        type_dist.append({
            'topic_id': tid,
            'entity_type': entity_type,
            'proportion': proportion,
        })
pd.DataFrame(type_dist).to_csv(output_dir / "topic_type_distribution.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# TEMPORAL ANALYSIS CSVs
# =============================================================================

print("\n📊 Loading temporal analysis data...")
with open(analysis_dir / "temporal_analysis_report.json", 'r', encoding='utf-8') as f:
    temporal_data = json.load(f)

# 1. Yearly Statistics
print("   Creating yearly_statistics.csv...")
pd.DataFrame(temporal_data['yearly_stats']).to_csv(
    output_dir / "yearly_statistics.csv", index=False, encoding='utf-8-sig')

# 2. Entity Timelines (full timeline for each entity)
print("   Creating entity_timelines.csv...")
years = temporal_data['years']
timelines = []
for entity, data in temporal_data['entity_timelines'].items():
    row = {
        'entity': entity,
        'type': data['type'],
        'total': data['total'],
        'first_year': data['first_year'],
        'last_year': data['last_year'],
        'peak_year': data['peak_year'],
        'peak_value': data.get('peak_value', ''),
        'active_years': data.get('active_years', ''),
        'trend_slope': data.get('trend_slope', ''),
    }
    # Add yearly columns
    for i, year in enumerate(years):
        row[f'y_{year}'] = data['timeline'][i]
    timelines.append(row)
pd.DataFrame(timelines).to_csv(output_dir / "entity_timelines.csv", index=False, encoding='utf-8-sig')

# 3. Period Comparison
print("   Creating period_comparison.csv...")
period_data = []
for period_name, pg in temporal_data['period_graphs'].items():
    period_data.append({
        'period': period_name,
        'unique_entities': pg['unique_entities'],
        'total_relations': pg['total_relations'],
        'top_entities': ', '.join(pg.get('top_entities', [])[:5]),
    })
pd.DataFrame(period_data).to_csv(output_dir / "period_comparison.csv", index=False, encoding='utf-8-sig')

# 4. Period Entities (top entities per period)
print("   Creating period_entities.csv...")
period_entities = []
for period_name, pg in temporal_data['period_graphs'].items():
    for node in pg['nodes'][:50]:  # Top 50 per period
        period_entities.append({
            'period': period_name,
            'entity': node['id'],
            'type': node['type'],
            'degree': node['degree'],
        })
pd.DataFrame(period_entities).to_csv(output_dir / "period_entities.csv", index=False, encoding='utf-8-sig')

# 5. Rising Entities
print("   Creating rising_entities.csv...")
pd.DataFrame(temporal_data['rising']).to_csv(
    output_dir / "rising_entities.csv", index=False, encoding='utf-8-sig')

# 6. Falling Entities
print("   Creating falling_entities.csv...")
pd.DataFrame(temporal_data['falling']).to_csv(
    output_dir / "falling_entities.csv", index=False, encoding='utf-8-sig')

# 7. Stable High Entities
print("   Creating stable_high_entities.csv...")
pd.DataFrame(temporal_data['stable_high']).to_csv(
    output_dir / "stable_high_entities.csv", index=False, encoding='utf-8-sig')

# 8. Type by Period
print("   Creating type_by_period.csv...")
type_period = []
for period, types in temporal_data['type_by_period'].items():
    for entity_type, count in types.items():
        type_period.append({
            'period': period,
            'entity_type': entity_type,
            'count': count,
        })
pd.DataFrame(type_period).to_csv(output_dir / "type_by_period.csv", index=False, encoding='utf-8-sig')

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("CSV EXPORT COMPLETE")
print("=" * 60)
print(f"\n📁 Output directory: {output_dir.absolute()}")
print("\n📄 Topic Analysis CSVs:")
print("   - topic_overview.csv")
print("   - topic_entities.csv")
print("   - topic_jaccard_similarity.csv")
print("   - entity_topic_count.csv")
print("   - topic_type_distribution.csv")
print("\n📄 Temporal Analysis CSVs:")
print("   - yearly_statistics.csv")
print("   - entity_timelines.csv")
print("   - period_comparison.csv")
print("   - period_entities.csv")
print("   - rising_entities.csv")
print("   - falling_entities.csv")
print("   - stable_high_entities.csv")
print("   - type_by_period.csv")
