import json
from collections import defaultdict
from pathlib import Path

def analyze_cultural_hypothesis():
    print("=== Cultural Studies Hypothesis Analysis (Degree >= 50 Graph) ===")
    graph_path = Path("outputs/v2/degree50/degree50_graph.json")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    edges = graph["edges"]
    
    cultural_concepts = ['cultura', 'historia', 'sociedad', 'arte', 'política', 'memoria', 'identidad', 'nación']
    textual_concepts = ['texto', 'escritura', 'lectura', 'lenguaje', 'palabra', 'narrador', 'narración']
    
    # 1a. Curated Graph Check
    print("\n--- CURATED GRAPH (Degree >= 100) CHECK ---")
    curated_path = Path("outputs/v2/curated/v2_curated_graph.json")
    with open(curated_path, "r", encoding="utf-8") as f:
        g_curated = json.load(f)
    
    c_weight_curated = 0
    t_weight_curated = 0
    
    for rel in g_curated["relations"]:
        subj = rel["subject"].lower()
        obj = rel["object"].lower()
        weight = rel["weight"]
        
        is_lit = "literatura" in subj or "literatura" in obj
        if not is_lit: continue
        
        other = obj if "literatura" in subj else subj
        
        if any(c in other for c in cultural_concepts):
            c_weight_curated += weight
        elif any(t in other for t in textual_concepts):
            t_weight_curated += weight
            
    print(f"Curated Cultural Weight: {c_weight_curated}")
    print(f"Curated Textual Weight: {t_weight_curated}")
    if t_weight_curated > 0:
        print(f"Curated Ratio: {c_weight_curated/t_weight_curated:.2f}x")

    # 1b. Degree 50 Overall Ratio
    print("\n--- DEGREE 50 GRAPH CHECK ---")
    cultural_stats = defaultdict(lambda: {"weight": 0, "count": 0})
    textual_stats = defaultdict(lambda: {"weight": 0, "count": 0})
    
    total_cultural_weight = 0
    total_textual_weight = 0
    
    for edge in edges:
        subj = edge["subject"].lower()
        obj = edge["object"].lower()
        weight = edge["weight"]
        
        # Check for 'literatura' framing
        is_lit = "literatura" in subj or "literatura" in obj
        if not is_lit:
            continue
            
        other = obj if "literatura" in subj else subj
        
        # Fuzzy match
        framing = None
        for c in cultural_concepts:
            if c == other or (len(other) > 3 and c in other):
                framing = "CULTURAL"
                cultural_stats[c]["weight"] += weight
                cultural_stats[c]["count"] += 1
                total_cultural_weight += weight
                break
        
        if not framing:
            for t in textual_concepts:
                if t == other or (len(other) > 3 and t in other):
                    framing = "TEXTUAL"
                    textual_stats[t]["weight"] += weight
                    textual_stats[t]["count"] += 1
                    total_textual_weight += weight
                    break

    print(f"\nTotal Cultural Weight: {total_cultural_weight}")
    print(f"Total Textual Weight: {total_textual_weight}")
    ratio = total_cultural_weight / total_textual_weight if total_textual_weight > 0 else 0
    print(f"Overall RATIO: {ratio:.2f}x")

    # 2. Temporal Evolution
    print("\n=== Temporal Evolution of Framing ===")
    temporal_cultural = defaultdict(int)
    temporal_textual = defaultdict(int)
    
    for edge in edges:
        subj = edge["subject"].lower()
        obj = edge["object"].lower()
        weight = edge["weight"]
        years = edge.get("years", [])
        
        is_lit = "literatura" in subj or "literatura" in obj
        if not is_lit:
            continue
            
        other = obj if "literatura" in subj else subj
        
        framing = None
        # Re-check framing for this specific edge
        for c in cultural_concepts:
            if c == other or (len(other) > 3 and c in other):
                framing = "CULTURAL"
                break
        if not framing:
            for t in textual_concepts:
                if t == other or (len(other) > 3 and t in other):
                    framing = "TEXTUAL"
                    break
            
        if framing:
            for year_str in years:
                try:
                    year = int(year_str)
                    if year < 2000: decade = "1996-1999"
                    elif year < 2010: decade = "2000-2009"
                    elif year < 2020: decade = "2010-2019"
                    else: decade = "2020-2024"
                    
                    if framing == "CULTURAL":
                        temporal_cultural[decade] += weight
                    else:
                        temporal_textual[decade] += weight
                except ValueError:
                    continue

    print(f"{ 'Period':<15} | { 'Cultural':<10} | { 'Textual':<10} | { 'Ratio':<10}")
    print("-" * 55)
    for decade in sorted(set(list(temporal_cultural.keys()) + list(temporal_textual.keys()))):
        c_w = temporal_cultural[decade]
        t_w = temporal_textual[decade]
        d_ratio = c_w / t_w if t_w > 0 else 0
        print(f"{decade:<15} | {c_w:<10} | {t_w:<10} | {d_ratio:.1f}x")

if __name__ == "__main__":
    analyze_cultural_hypothesis()
