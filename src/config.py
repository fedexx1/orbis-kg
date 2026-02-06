"""
Shared configuration for KGGen V2 workflow.
All paths are relative to project root (KGGEN/).
"""

from pathlib import Path

# Project root (adjust if running from different directory)
PROJECT_ROOT = Path(__file__).parent.parent

CONFIG = {
    # ==========================================================================
    # INPUT PATHS
    # ==========================================================================
    "articles_dir": PROJECT_ROOT / "data/raw/articles_orbis/",
    "topic_assignments": PROJECT_ROOT / "data/processed/topic_assignments_orbis.csv",
    "curation_file": PROJECT_ROOT / "outputs/v2/curation/v2_curation_recommendation_degree=100.xlsx",
    
    # V1 outputs (for comparison)
    "v1_aggregated_graph": PROJECT_ROOT / "outputs/graphs/aggregated/stage2_graph_20260106_202745.json",
    "v1_metrics": PROJECT_ROOT / "outputs/graphs/aggregated/stage2_metrics_20260106_202745.json",
    "v1_per_article_dir": PROJECT_ROOT / "outputs/graphs/per_article/graphs/",
    
    # ==========================================================================
    # V2 OUTPUT PATHS
    # ==========================================================================
    "v2_per_article_dir": PROJECT_ROOT / "outputs/v2/per_article/",
    "v2_aggregated_dir": PROJECT_ROOT / "outputs/v2/aggregated/",
    "v2_curated_dir": PROJECT_ROOT / "outputs/v2/curated/",
    "v2_viz_dir": PROJECT_ROOT / "outputs/visualizations/v2/",
    "v2_logs_dir": PROJECT_ROOT / "outputs/v2/logs/",
    
    # ==========================================================================
    # LLM SETTINGS
    # ==========================================================================
    # Using Gemini 2.0 Flash for cost efficiency (~$1-2 for full corpus)
    # Alternative: "openai/gpt-4o-mini", "anthropic/claude-3-haiku-20240307"
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0.0,
    "chunk_size": 5000,
    
    # Context hint for extraction
    "context": (
        "Academic articles on literary studies and literary criticism from Argentina, "
        "written in Spanish. The corpus includes scholarly analysis of literature, "
        "critical theory, cultural studies, and references to authors like Borges, "
        "Cortázar, Puig, Saer, and Piglia. Extract entities (authors, works, concepts, "
        "places, institutions) and their relationships."
    ),
    
    # ==========================================================================
    # CURATION SETTINGS
    # ==========================================================================
    "min_entity_degree": 100,        # Threshold for entity curation
    "fuzzy_match_min_length": 4,     # Minimum chars for substring matching
    "include_uncurated": False,      # Include edges with uncurated entities?
    
    # ==========================================================================
    # PROCESSING SETTINGS
    # ==========================================================================
    "batch_size": 50,                # Articles per batch (for progress reporting)
    "request_delay": 0.1,            # Seconds between API requests
    "max_retries": 3,                # Retries on API failure
    "retry_delay": 5.0,              # Seconds to wait before retry
    
    # ==========================================================================
    # DEBUG/TEST SETTINGS
    # ==========================================================================
    "test_mode": False,              # If True, process only 10 articles
    "test_sample_size": 10,          # Number of articles in test mode
    "verbose": True,                 # Print detailed progress
    "save_intermediate": True,       # Save intermediate results
}


def get_config():
    """Return configuration dictionary."""
    return CONFIG.copy()


def ensure_directories():
    """Create output directories if they don't exist."""
    dirs = [
        CONFIG["v2_per_article_dir"],
        CONFIG["v2_aggregated_dir"],
        CONFIG["v2_curated_dir"],
        CONFIG["v2_viz_dir"],
        CONFIG["v2_logs_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories ready")


def validate_inputs():
    """Check that required input files exist."""
    checks = [
        ("Articles directory", CONFIG["articles_dir"]),
        ("Topic assignments", CONFIG["topic_assignments"]),
        ("Curation file", CONFIG["curation_file"]),
    ]
    
    all_ok = True
    for name, path in checks:
        if Path(path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    print("KGGen V2 Configuration")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {CONFIG['model']}")
    print(f"Test mode: {CONFIG['test_mode']}")
    print()
    validate_inputs()
    ensure_directories()
