"""
================================================================================
KGGen V2: Visualization with Full Provenance
================================================================================

Creates interactive HTML visualization from V2 curated graph.
Same features as V1 but with 100% provenance coverage.

Usage:
    cd src/visualization/
    python create_v2_viz.py

Output:
    outputs/visualizations/v2/v2_visualization.html
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG, ensure_directories


def create_visualization(config: dict):
    """Create interactive D3.js visualization from V2 curated graph."""
    
    print("\n" + "=" * 60)
    print("KGGen V2: CREATING VISUALIZATION")
    print("=" * 60)
    
    ensure_directories()
    
    # Load curated graph
    print("\n📂 Loading V2 curated graph...")
    graph_path = Path(config["v2_curated_dir"]) / "v2_curated_graph.json"
    
    if not graph_path.exists():
        print(f"❌ ERROR: Graph not found at {graph_path}")
        print("   Run apply_curation.py first.")
        sys.exit(1)
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    print(f"   Entities: {len(graph['entities'])}")
    print(f"   Edges: {len(graph['relations'])}")
    
    # Build nodes
    entity_types = graph.get('entity_types', {})
    entity_degrees = {}
    for rel in graph['relations']:
        entity_degrees[rel['subject']] = entity_degrees.get(rel['subject'], 0) + rel['weight']
        entity_degrees[rel['object']] = entity_degrees.get(rel['object'], 0) + rel['weight']
    
    nodes = []
    for entity in graph['entities']:
        nodes.append({
            "id": entity,
            "type": entity_types.get(entity, "unknown"),
            "degree": entity_degrees.get(entity, 0),
        })
    
    # Build links
    links = []
    for rel in graph['relations']:
        links.append({
            "source": rel['subject'],
            "target": rel['object'],
            "weight": rel['weight'],
            "predicates": rel['predicates'],
            "articles": rel['articles'],
            "topics": rel['topics'],
            "years": rel['years'],
        })
    
    # Get unique types and topics for filters
    all_types = sorted(set(entity_types.values()))
    all_topics = sorted(set(t for rel in graph['relations'] for t in rel['topics']))
    
    print(f"   Entity types: {len(all_types)}")
    print(f"   Topics: {len(all_topics)}")
    
    # Type colors - NEW format: "Category: Subtype"
    type_colors = {
        # Person types
        "Person: Author (literary)": "#e74c3c",
        "Person: Author (critic)": "#3498db",
        "Person: Author (theory)": "#9b59b6",
        "Person: Author (other)": "#f39c12",
        "Person: Character": "#8e44ad",
        "Person: Historical": "#e67e22",
        
        # Concept types
        "Concept: Literary": "#2ecc71",
        "Concept: Genre": "#27ae60",
        "Concept: Cultural": "#1abc9c",
        
        # Place types
        "Place: City": "#f1c40f",
        "Place: Country": "#f39c12",
        "Place: Region": "#e67e22",
        
        # Organization types
        "Org: Institution": "#2980b9",
        "Org: Journal": "#c0392b",
        "Org: Publisher": "#d35400",
        
        # Thing types
        "Thing: Literary object": "#34495e",
        "Thing: General": "#7f8c8d",
        "Thing: Role": "#95a5a6",
        
        "unknown": "#95a5a6",
    }
    
    # Create HTML
    print("\n🎨 Generating visualization...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orbis Tertius Knowledge Graph V2</title>
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
            width: 320px; 
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
            background: #2ecc71;
            color: #000;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: bold;
            margin-left: 8px;
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
            border-color: #6c63ff;
        }}
        .entity-list {{ 
            max-height: calc(100vh - 400px); 
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
            background: #6c63ff; 
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
            min-height: 100px;
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
        .article-list {{
            font-size: 0.75rem;
            color: #6c63ff;
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
        .stats span {{ color: #6c63ff; font-weight: bold; }}
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
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>Orbis Tertius KG <span class="badge">V2</span></h1>
            <div class="subtitle">
                {len(nodes)} entities • {len(links)} edges • 100% provenance
            </div>
            
            <div class="controls">
                <input type="text" id="search" placeholder="🔍 Search entities...">
                <select id="typeFilter">
                    <option value="">All entity types</option>
                    {"".join(f'<option value="{t}">{t}</option>' for t in all_types)}
                </select>
                <select id="topicFilter">
                    <option value="">All topics</option>
                    {"".join(f'<option value="{t}">Topic {t}</option>' for t in all_topics)}
                </select>
            </div>
            
            <div class="entity-list" id="entityList"></div>
            
            <div id="nodeInfo">Click a node to see details and provenance</div>
            
            <div class="stats">
                <div>Entities: <span>{len(nodes)}</span></div>
                <div>Edges: <span>{len(links)}</span></div>
                <div>Provenance: <span>100%</span></div>
                <div>Generated: <span>{datetime.now().strftime('%Y-%m-%d')}</span></div>
            </div>
            
            <div class="legend">
                <div class="legend-title">Entity Types</div>
                {"".join(f'''<div class="legend-item" onclick="filterByType('{t}')">
                    <div class="legend-color" style="background:{type_colors.get(t, '#888')}"></div>
                    <span>{t}</span>
                </div>''' for t in all_types[:12])}
            </div>
        </div>
        <div id="graph"></div>
    </div>
    
    <script>
        const nodes = {json.dumps(nodes)};
        const links = {json.dumps(links)};
        const typeColors = {json.dumps(type_colors)};
        
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
            .attr("fill", d => d === "arrow-highlight" ? "#6c63ff" : "#4a5568")
            .attr("d", "M0,-5L10,0L0,5");
        
        const g = svg.append("g");
        
        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", e => g.attr("transform", e.transform));
        svg.call(zoom);
        
        // Scales
        const degrees = nodes.map(n => n.degree);
        const sizeScale = d3.scaleSqrt().domain([0, d3.max(degrees)]).range([4, 25]);
        const weights = links.map(l => l.weight);
        const weightScale = d3.scaleLinear().domain([1, d3.max(weights) || 1]).range([1, 4]);
        
        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-350))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(d => sizeScale(d.degree) + 8));
        
        // Links
        const link = g.append("g").selectAll("line").data(links).join("line")
            .attr("stroke", "#4a5568")
            .attr("stroke-width", d => weightScale(d.weight))
            .attr("stroke-opacity", 0.6)
            .attr("marker-end", "url(#arrow)");
        
        // Edge labels
        const edgeLabels = g.append("g").selectAll("text").data(links).join("text")
            .attr("class", "edge-label")
            .attr("font-size", "6px")
            .attr("fill", "#888")
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
            .text(d => d.id.length > 20 ? d.id.substring(0, 18) + "..." : d.id);
        
        // Tooltips
        node.append("title").text(d => `${{d.id}}\\n${{d.type}}\\n${{d.degree}} connections`);
        
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
                    <span class="entity-type" style="border-left: 3px solid ${{typeColors[e.type] || '#888'}}">${{e.type.includes(':') ? e.type.split(':')[0] : e.type}}</span>
                    <span class="entity-degree">${{e.degree}}</span>
                </span>
            </div>
        `).join('');
        
        entityList.querySelectorAll('.entity-item').forEach(item => {{
            item.addEventListener('click', () => selectNode(item.getAttribute('data-id')));
        }});
        
        // Selection
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
                .attr('stroke', l => isConnected(l) ? '#6c63ff' : '#4a5568')
                .attr('stroke-width', l => isConnected(l) ? weightScale(l.weight) + 1.5 : weightScale(l.weight));
            
            edgeLabels.attr('opacity', l => isConnected(l) ? 1 : 0);
            labels.attr('opacity', d => connectedIds.has(d.id) ? 1 : 0)
                  .attr('fill', d => d.id === id ? '#ffcc00' : 'white');
            
            // Show info with article provenance
            const connections = links.filter(l => l.source.id === id || l.target.id === id)
                                     .sort((a,b) => b.weight - a.weight);
            
            document.getElementById('nodeInfo').innerHTML = `
                <div class="node-title">${{id}}</div>
                <div class="node-type">${{nodeData?.type || 'unknown'}} • ${{connections.length}} connections</div>
                <div class="relation-list">
                    ${{connections.slice(0, 8).map(l => {{
                        const other = l.source.id === id ? l.target.id : l.source.id;
                        const preds = l.predicates.slice(0, 2).join(", ");
                        const articles = l.articles.slice(0, 3).join(", ");
                        return `<div class="relation-item">
                            <b>${{other}}</b> <span style="color:#888">(${{l.weight}} mentions)</span>
                            <div style="font-size:0.75rem;color:#aaa">[${{preds}}]</div>
                            <div class="article-list">📄 ${{articles}}${{l.articles.length > 3 ? '...' : ''}}</div>
                        </div>`;
                    }}).join('')}}
                    ${{connections.length > 8 ? `<div style="color:#666;padding-top:4px">...and ${{connections.length - 8}} more</div>` : ''}}
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
        
        // Filters
        const typeFilter = document.getElementById('typeFilter');
        const topicFilter = document.getElementById('topicFilter');
        
        function filterByType(type) {{
            typeFilter.value = type;
            applyFilters();
        }}
        
        typeFilter.addEventListener('change', applyFilters);
        topicFilter.addEventListener('change', applyFilters);
        
        function applyFilters() {{
            const type = typeFilter.value;
            const topic = topicFilter.value;
            
            let visibleNodes = new Set(nodes.map(n => n.id));
            
            if (type) {{
                visibleNodes = new Set(nodes.filter(n => n.type === type).map(n => n.id));
            }}
            
            if (topic) {{
                const topicNum = parseInt(topic);
                const topicLinks = links.filter(l => l.topics.includes(topicNum));
                const topicNodes = new Set();
                topicLinks.forEach(l => {{ topicNodes.add(l.source.id); topicNodes.add(l.target.id); }});
                visibleNodes = new Set([...visibleNodes].filter(n => topicNodes.has(n)));
            }}
            
            node.classed('dimmed', d => !visibleNodes.has(d.id));
            link.classed('dimmed', l => !visibleNodes.has(l.source.id) || !visibleNodes.has(l.target.id));
            
            // Show labels for filtered entities
            if (type || topic) {{
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
                document.getElementById('nodeInfo').innerHTML = 'Click a node to see details and provenance';
                typeFilter.value = '';
                topicFilter.value = '';
            }}
        }});
    </script>
</body>
</html>"""
    
    # Save
    output_dir = Path(config["v2_viz_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v2_visualization.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Visualization saved: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n   Open in browser to explore the knowledge graph.")
    
    return output_path


if __name__ == "__main__":
    create_visualization(CONFIG)
