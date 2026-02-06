"""
================================================================================
EDC: Visualization with Predicate Definitions
================================================================================

Creates interactive HTML visualization from EDC extraction results.
Similar to KGGen V2 visualization but includes EDC predicate definitions.

Usage:
    python src/visualization/create_edc_viz.py

Output:
    outputs/visualizations/edc/edc_visualization.html
================================================================================
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Set

# Paths
KGGEN_ROOT = Path(__file__).parent.parent.parent
EDC_RESULTS = KGGEN_ROOT / "outputs" / "edc" / "full_20260116_161616" / "results" / "iter0" / "result_at_each_stage.json"
EDC_METADATA = KGGEN_ROOT / "edc" / "datasets" / "orbis_metadata.json"
OUTPUT_DIR = KGGEN_ROOT / "outputs" / "visualizations" / "edc"

# Entity type classification based on heuristics
def classify_entity(entity: str) -> str:
    """Classify entity into a type category."""
    entity_lower = entity.lower().strip()

    # Year detection
    if re.match(r'^\d{4}$', entity.strip()):
        return "Temporal: Year"
    if re.match(r'^(siglo|década|años)\s+', entity_lower):
        return "Temporal: Period"

    # Known literary authors (sample list)
    literary_authors = {
        "borges", "cortázar", "saer", "puig", "piglia", "arlt", "sarmiento",
        "echeverría", "hernández", "lugones", "girondo", "gombrowicz",
        "marechal", "macedonio", "lamborghini", "bioy casares", "walsh",
        "ocampo", "storni", "alfonsina", "quiroga", "güiraldes"
    }

    # Known critics/theorists
    critics_theorists = {
        "sarlo", "ludmer", "jitrik", "viñas", "gramuglio", "panesi", "amícola",
        "barthes", "foucault", "derrida", "lacan", "freud", "benjamin", "adorno",
        "bajtín", "bajtin", "kristeva", "genette", "deleuze", "lotman", "shklovski",
        "tyniánov", "jameson", "sklodowska", "hutcheon"
    }

    # Check against known entities
    for author in literary_authors:
        if author in entity_lower:
            return "Person: Author"

    for critic in critics_theorists:
        if critic in entity_lower:
            return "Person: Critic/Theorist"

    # Place indicators
    place_indicators = ["argentina", "buenos aires", "francia", "españa", "méxico",
                       "uruguay", "chile", "europa", "latinoamérica", "parís", "londres"]
    for place in place_indicators:
        if place in entity_lower:
            return "Place: Location"

    # Literary concept indicators
    concept_suffixes = ["ismo", "ción", "dad", "miento", "ura", "logía"]
    for suffix in concept_suffixes:
        if entity_lower.endswith(suffix):
            return "Concept: Literary"

    # Work indicators (titles often have capitals)
    if entity and entity[0].isupper() and len(entity) > 3:
        words = entity.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            return "Work: Text"

    return "Other"


def load_edc_results() -> Tuple[List[Dict], Dict[str, str], Dict[int, Dict]]:
    """Load EDC results and metadata."""
    print("Loading EDC results...")

    # Load triplets
    with open(EDC_RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load metadata for provenance
    metadata = {}
    if EDC_METADATA.exists():
        with open(EDC_METADATA, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            for chunk in meta.get("chunks", []):
                metadata[chunk["line_index"]] = chunk

    # Extract predicate definitions
    predicate_defs = {}
    for chunk in data:
        schema_def = chunk.get("schema_definition", {})
        for key, value in schema_def.items():
            if key.startswith("Here are"):
                continue
            match = re.search(r'\*\*(.+?)\*\*', key)
            if match:
                pred_name = match.group(1)
                if pred_name not in predicate_defs:
                    predicate_defs[pred_name] = value

    return data, predicate_defs, metadata


def build_graph(data: List[Dict], metadata: Dict[int, Dict], min_degree: int = 100) -> Tuple[List, List, Dict]:
    """
    Build aggregated graph from EDC triplets.

    Uses min_degree filtering to match KGGen V2 visualization approach.
    Only entities with degree >= min_degree are included.
    """
    print(f"Building graph (min_degree={min_degree})...")

    # Aggregate triplets
    edge_data = defaultdict(lambda: {
        "weight": 0,
        "predicates": [],
        "articles": set(),
        "years": set(),
        "chunks": []
    })

    entity_degrees = Counter()
    all_entities = set()

    for chunk in data:
        chunk_idx = chunk.get("index", 0)
        chunk_meta = metadata.get(chunk_idx, {})
        article = chunk_meta.get("filename", f"chunk_{chunk_idx}")
        year = chunk_meta.get("year", "unknown")

        triplets = chunk.get("schema_canonicalization", [])
        for triplet in triplets:
            if len(triplet) >= 3:
                subj, pred, obj = triplet[0], triplet[1], triplet[2]

                # Skip very short entities (likely noise)
                if len(subj) < 2 or len(obj) < 2:
                    continue

                all_entities.add(subj)
                all_entities.add(obj)

                # Use tuple for edge aggregation
                edge_key = (subj, obj)

                edge_data[edge_key]["weight"] += 1
                if pred not in edge_data[edge_key]["predicates"]:
                    edge_data[edge_key]["predicates"].append(pred)
                edge_data[edge_key]["articles"].add(article)
                edge_data[edge_key]["years"].add(year)
                edge_data[edge_key]["chunks"].append(chunk_idx)

                entity_degrees[subj] += 1
                entity_degrees[obj] += 1

    print(f"  Raw entities: {len(all_entities)}")
    print(f"  Raw edges: {len(edge_data)}")

    # Filter entities by min_degree (same approach as KGGen V2)
    high_degree_entities = set(e for e, deg in entity_degrees.items() if deg >= min_degree)
    print(f"  Entities with degree >= {min_degree}: {len(high_degree_entities)}")

    # Build filtered edges (only between high-degree entities)
    filtered_edges = []
    for (subj, obj), edata in edge_data.items():
        if subj in high_degree_entities and obj in high_degree_entities:
            filtered_edges.append({
                "source": subj,
                "target": obj,
                "weight": edata["weight"],
                "predicates": edata["predicates"][:5],  # Top 5 predicates
                "articles": list(edata["articles"])[:5],  # Top 5 articles
                "years": sorted(edata["years"])[:5],
            })

    # Get entities that appear in edges
    connected_entities = set()
    for edge in filtered_edges:
        connected_entities.add(edge["source"])
        connected_entities.add(edge["target"])

    # Build nodes
    nodes = []
    entity_types = {}
    for entity in connected_entities:
        etype = classify_entity(entity)
        entity_types[entity] = etype
        nodes.append({
            "id": entity,
            "type": etype,
            "degree": entity_degrees[entity]
        })

    # Sort nodes by degree for consistent display
    nodes.sort(key=lambda x: x["degree"], reverse=True)

    print(f"  Filtered nodes: {len(nodes)}")
    print(f"  Filtered edges: {len(filtered_edges)}")
    if nodes:
        print(f"  Avg edges per node: {len(filtered_edges) * 2 / len(nodes):.1f}")

    return nodes, filtered_edges, entity_types


def create_visualization(nodes: List, links: List, entity_types: Dict, predicate_defs: Dict):
    """Create D3.js visualization HTML."""
    print("Creating visualization...")

    # Get unique types for filters
    all_types = sorted(set(entity_types.values()))

    # Type colors
    type_colors = {
        "Person: Author": "#e74c3c",
        "Person: Critic/Theorist": "#9b59b6",
        "Place: Location": "#f1c40f",
        "Work: Text": "#3498db",
        "Concept: Literary": "#2ecc71",
        "Temporal: Year": "#e67e22",
        "Temporal: Period": "#d35400",
        "Other": "#7f8c8d",
    }

    # Sample predicate definitions for display
    pred_defs_sample = dict(list(predicate_defs.items())[:100])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDC Knowledge Graph - Orbis Tertius</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            overflow: hidden;
        }}
        #container {{ display: flex; height: 100vh; }}
        #sidebar {{
            width: 350px;
            background: #12121a;
            padding: 16px;
            overflow-y: auto;
            border-right: 1px solid #2a2a3a;
        }}
        #graph {{ flex: 1; position: relative; }}
        h1 {{
            font-size: 1.3rem;
            margin-bottom: 8px;
            color: #fff;
        }}
        .subtitle {{
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 16px;
        }}
        .badge {{
            display: inline-block;
            background: #e74c3c;
            color: #fff;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: bold;
            margin-left: 8px;
        }}
        .badge-edc {{
            background: linear-gradient(135deg, #e74c3c, #9b59b6);
        }}
        .controls {{ margin-bottom: 16px; }}
        .controls input, .controls select {{
            width: 100%;
            padding: 8px 12px;
            margin-bottom: 8px;
            border: 1px solid #3a3a4a;
            border-radius: 6px;
            background: #1a1a24;
            color: #e0e0e0;
            font-size: 0.9rem;
        }}
        .controls input:focus, .controls select:focus {{
            outline: none;
            border-color: #e74c3c;
        }}
        .entity-list {{
            max-height: calc(100vh - 500px);
            overflow-y: auto;
        }}
        .entity-item {{
            padding: 8px 10px;
            margin: 4px 0;
            background: #1a1a24;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .entity-item:hover {{ background: #252532; }}
        .entity-item.selected {{
            background: #e74c3c;
            color: #fff;
        }}
        .entity-name {{
            font-size: 0.85rem;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .entity-meta {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .entity-type {{
            font-size: 0.7rem;
            padding: 2px 6px;
            background: #2a2a3a;
            border-radius: 4px;
        }}
        .entity-degree {{
            font-size: 0.75rem;
            color: #888;
            min-width: 30px;
            text-align: right;
        }}
        #nodeInfo {{
            margin-top: 16px;
            padding: 12px;
            background: #1a1a24;
            border-radius: 8px;
            font-size: 0.85rem;
            min-height: 120px;
        }}
        .node-title {{
            font-weight: bold;
            font-size: 1rem;
            margin-bottom: 4px;
        }}
        .node-type {{
            color: #888;
            font-size: 0.8rem;
            margin-bottom: 8px;
        }}
        .relation-item {{
            padding: 6px 0;
            border-bottom: 1px solid #2a2a3a;
        }}
        .relation-item:last-child {{ border-bottom: none; }}
        .predicate-def {{
            font-size: 0.75rem;
            color: #9b59b6;
            font-style: italic;
            margin-top: 4px;
            padding: 4px 8px;
            background: #1f1f2a;
            border-radius: 4px;
        }}
        .article-list {{
            font-size: 0.75rem;
            color: #e74c3c;
            margin-top: 4px;
        }}
        .dimmed {{ opacity: 0.15; }}
        .highlighted {{ stroke: #ffcc00 !important; stroke-width: 3px !important; }}
        .stats {{
            margin-top: 16px;
            padding: 12px;
            background: #1a1a24;
            border-radius: 8px;
            font-size: 0.8rem;
        }}
        .stats div {{ margin: 4px 0; }}
        .stats span {{ color: #e74c3c; font-weight: bold; }}
        .legend {{
            margin-top: 16px;
            padding: 12px;
            background: #1a1a24;
            border-radius: 8px;
        }}
        .legend-title {{
            font-size: 0.85rem;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            padding: 3px 0;
            cursor: pointer;
        }}
        .legend-item:hover {{ color: #fff; }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .feature-badge {{
            display: inline-block;
            background: #9b59b6;
            color: #fff;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 0.65rem;
            margin-left: 4px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>EDC Knowledge Graph <span class="badge badge-edc">EDC</span></h1>
            <div class="subtitle">
                {len(nodes)} entities | {len(links)} edges | Spanish predicates
                <span class="feature-badge">+Definitions</span>
            </div>

            <div class="controls">
                <input type="text" id="search" placeholder="Buscar entidades...">
                <select id="typeFilter">
                    <option value="">Todos los tipos</option>
                    {"".join(f'<option value="{t}">{t}</option>' for t in all_types)}
                </select>
            </div>

            <div class="entity-list" id="entityList"></div>

            <div id="nodeInfo">
                <div style="color:#888">Click en un nodo para ver detalles y definiciones de predicados</div>
            </div>

            <div class="stats">
                <div>Total triplets: <span>106,883</span></div>
                <div>Shown entities: <span>{len(nodes)}</span></div>
                <div>Shown edges: <span>{len(links)}</span></div>
                <div>Predicate defs: <span>{len(predicate_defs):,}</span></div>
                <div>Generated: <span>{datetime.now().strftime('%Y-%m-%d')}</span></div>
            </div>

            <div class="legend">
                <div class="legend-title">Entity Types</div>
                {"".join(f'''<div class="legend-item" onclick="filterByType('{t}')">
                    <div class="legend-color" style="background:{type_colors.get(t, '#888')}"></div>
                    <span>{t}</span>
                </div>''' for t in all_types)}
            </div>
        </div>
        <div id="graph"></div>
    </div>

    <script>
        const nodes = {json.dumps(nodes, ensure_ascii=False)};
        const links = {json.dumps(links, ensure_ascii=False)};
        const typeColors = {json.dumps(type_colors)};
        const predicateDefs = {json.dumps(pred_defs_sample, ensure_ascii=False)};

        const width = document.getElementById('graph').clientWidth;
        const height = document.getElementById('graph').clientHeight;

        const svg = d3.select("#graph").append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, width, height]);

        // Arrow markers
        svg.append("defs").selectAll("marker")
            .data(["arrow", "arrow-highlight"])
            .join("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("fill", d => d === "arrow-highlight" ? "#e74c3c" : "#4a5568")
            .attr("d", "M0,-5L10,0L0,5");

        const g = svg.append("g");

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", e => g.attr("transform", e.transform));
        svg.call(zoom);

        // Scales
        const degrees = nodes.map(n => n.degree);
        const sizeScale = d3.scaleSqrt().domain([0, d3.max(degrees)]).range([4, 30]);
        const weights = links.map(l => l.weight);
        const weightScale = d3.scaleLinear().domain([1, d3.max(weights) || 1]).range([1, 5]);

        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(d => sizeScale(d.degree) + 10));

        // Links
        const link = g.append("g").selectAll("line").data(links).join("line")
            .attr("stroke", "#4a5568")
            .attr("stroke-width", d => weightScale(d.weight))
            .attr("stroke-opacity", 0.6)
            .attr("marker-end", "url(#arrow)");

        // Edge labels
        const edgeLabels = g.append("g").selectAll("text").data(links).join("text")
            .attr("class", "edge-label")
            .attr("font-size", "7px")
            .attr("fill", "#aaa")
            .attr("text-anchor", "middle")
            .attr("opacity", 0)
            .text(d => d.predicates[0] || "");

        // Nodes
        const nodeMap = new Map(nodes.map(n => [n.id, n]));
        const node = g.append("g").selectAll("circle").data(nodes).join("circle")
            .attr("r", d => sizeScale(d.degree))
            .attr("fill", d => typeColors[d.type] || "#888")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", (e, d) => {{ e.stopPropagation(); selectNode(d.id); }});

        // Labels
        const labels = g.append("g").selectAll("text").data(nodes).join("text")
            .attr("class", "node-label")
            .attr("dy", d => -sizeScale(d.degree) - 4)
            .attr("opacity", 0)
            .attr("fill", "white")
            .attr("font-size", "10px")
            .attr("text-anchor", "middle")
            .text(d => d.id.length > 25 ? d.id.substring(0, 23) + "..." : d.id);

        // Tooltips
        node.append("title").text(d => `${{d.id}}\\n${{d.type}}\\n${{d.degree}} conexiones`);

        // Tick
        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            labels.attr("x", d => d.x).attr("y", d => d.y);
            edgeLabels.attr("x", d => (d.source.x + d.target.x)/2)
                      .attr("y", d => (d.source.y + d.target.y)/2);
        }});

        // Drag functions
        function dragstarted(e) {{
            if (!e.active) simulation.alphaTarget(0.3).restart();
            e.subject.fx = e.subject.x;
            e.subject.fy = e.subject.y;
        }}
        function dragged(e) {{ e.subject.fx = e.x; e.subject.fy = e.y; }}
        function dragended(e) {{
            if (!e.active) simulation.alphaTarget(0);
            e.subject.fx = null;
            e.subject.fy = null;
        }}

        // Entity list
        const entityList = document.getElementById('entityList');
        const sortedNodes = [...nodes].sort((a,b) => b.degree - a.degree);
        entityList.innerHTML = sortedNodes.map(e => `
            <div class="entity-item" data-id="${{e.id.replace(/"/g, '&quot;')}}">
                <span class="entity-name">${{e.id}}</span>
                <span class="entity-meta">
                    <span class="entity-type" style="border-left: 3px solid ${{typeColors[e.type] || '#888'}}">${{e.type.split(':')[0]}}</span>
                    <span class="entity-degree">${{e.degree}}</span>
                </span>
            </div>
        `).join('');

        entityList.querySelectorAll('.entity-item').forEach(item => {{
            item.addEventListener('click', () => selectNode(item.getAttribute('data-id')));
        }});

        // Selection with predicate definitions
        let selectedNode = null;
        function selectNode(id) {{
            selectedNode = id;
            const nodeData = nodeMap.get(id);

            const connectedIds = new Set([id]);
            links.forEach(l => {{
                if (l.source.id === id) connectedIds.add(l.target.id);
                if (l.target.id === id) connectedIds.add(l.source.id);
            }});

            entityList.querySelectorAll('.entity-item').forEach(item => {{
                item.classList.toggle('selected', item.getAttribute('data-id') === id);
            }});

            node.classed('highlighted', d => d.id === id)
                .classed('dimmed', d => !connectedIds.has(d.id));

            const isConnected = l => l.source.id === id || l.target.id === id;
            link.classed('dimmed', l => !isConnected(l))
                .attr('stroke', l => isConnected(l) ? '#e74c3c' : '#4a5568')
                .attr('stroke-width', l => isConnected(l) ? weightScale(l.weight) + 2 : weightScale(l.weight));

            edgeLabels.attr('opacity', l => isConnected(l) ? 1 : 0);
            labels.attr('opacity', d => connectedIds.has(d.id) ? 1 : 0)
                  .attr('fill', d => d.id === id ? '#ffcc00' : 'white');

            // Show info with predicate definitions
            const connections = links.filter(l => l.source.id === id || l.target.id === id)
                                     .sort((a,b) => b.weight - a.weight);

            document.getElementById('nodeInfo').innerHTML = `
                <div class="node-title">${{id}}</div>
                <div class="node-type">${{nodeData?.type || 'unknown'}} | ${{connections.length}} conexiones</div>
                <div class="relation-list">
                    ${{connections.slice(0, 6).map(l => {{
                        const other = l.source.id === id ? l.target.id : l.source.id;
                        const mainPred = l.predicates[0] || '';
                        const predDef = predicateDefs[mainPred] || '';
                        const articles = l.articles.slice(0, 2).join(", ");
                        return `<div class="relation-item">
                            <b>${{other}}</b> <span style="color:#888">(${{l.weight}}x)</span>
                            <div style="font-size:0.75rem;color:#e74c3c">[${{mainPred}}]</div>
                            ${{predDef ? `<div class="predicate-def">"${{predDef.substring(0, 100)}}${{predDef.length > 100 ? '...' : ''}}"</div>` : ''}}
                            <div class="article-list">${{articles}}${{l.articles.length > 2 ? '...' : ''}}</div>
                        </div>`;
                    }}).join('')}}
                    ${{connections.length > 6 ? `<div style="color:#666;padding-top:4px">...y ${{connections.length - 6}} más</div>` : ''}}
                </div>
            `;
        }}

        // Search
        document.getElementById('search').addEventListener('input', e => {{
            const term = e.target.value.toLowerCase();
            if (!term) {{
                node.classed('dimmed', false);
                link.classed('dimmed', false);
                entityList.querySelectorAll('.entity-item').forEach(i => i.style.display = '');
            }} else {{
                node.classed('dimmed', d => !d.id.toLowerCase().includes(term));
                link.classed('dimmed', true);
                entityList.querySelectorAll('.entity-item').forEach(i => {{
                    i.style.display = i.querySelector('.entity-name').textContent.toLowerCase().includes(term) ? '' : 'none';
                }});
            }}
        }});

        // Type filter
        const typeFilter = document.getElementById('typeFilter');

        function filterByType(type) {{
            typeFilter.value = type;
            applyFilters();
        }}

        typeFilter.addEventListener('change', applyFilters);

        function applyFilters() {{
            const type = typeFilter.value;

            let visibleNodes = new Set(nodes.map(n => n.id));

            if (type) {{
                visibleNodes = new Set(nodes.filter(n => n.type === type).map(n => n.id));
            }}

            node.classed('dimmed', d => !visibleNodes.has(d.id));
            link.classed('dimmed', l => !visibleNodes.has(l.source.id) || !visibleNodes.has(l.target.id));

            if (type) {{
                labels.attr('opacity', d => visibleNodes.has(d.id) ? 1 : 0);
            }} else {{
                labels.attr('opacity', 0);
            }}
        }}

        // Clear selection
        svg.on("click", e => {{
            if (e.target === svg.node()) {{
                selectedNode = null;
                node.classed('highlighted', false).classed('dimmed', false);
                link.classed('dimmed', false)
                    .attr('stroke', '#4a5568')
                    .attr('stroke-width', d => weightScale(d.weight));
                labels.attr('opacity', 0);
                edgeLabels.attr('opacity', 0);
                entityList.querySelectorAll('.entity-item').forEach(i => i.classList.remove('selected'));
                document.getElementById('nodeInfo').innerHTML = '<div style="color:#888">Click en un nodo para ver detalles y definiciones de predicados</div>';
                typeFilter.value = '';
            }}
        }});
    </script>
</body>
</html>"""

    return html_content


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("EDC: CREATING VISUALIZATION")
    print("=" * 60)

    # Load data
    data, predicate_defs, metadata = load_edc_results()
    print(f"  Loaded {len(data)} chunks")
    print(f"  Loaded {len(predicate_defs)} predicate definitions")
    print(f"  Loaded {len(metadata)} metadata entries")

    # Build graph with min_degree=100 to match KGGen V2 approach
    nodes, links, entity_types = build_graph(data, metadata, min_degree=100)

    # Create visualization
    html_content = create_visualization(nodes, links, entity_types, predicate_defs)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "edc_visualization.html"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nVisualization saved: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print("\nOpen in browser to explore the EDC knowledge graph.")

    return output_path


if __name__ == "__main__":
    main()
