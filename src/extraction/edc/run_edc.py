"""
Run EDC framework on Orbis Tertius corpus.

This script orchestrates the EDC (Extract-Define-Canonicalize) pipeline
with Mistral API and multilingual embeddings for Spanish literary criticism.

FEATURES:
- Checkpointing: Saves progress after each phase (OIE, SD, SC)
- Resume capability: Automatically resumes from last checkpoint on restart
- UTF-8 encoding: Proper handling for Spanish characters

REQUIREMENTS:
- EDC framework (https://github.com/clear-nus/edc)
- MISTRAL_API_KEY environment variable
- Preprocessed input files (run preprocess_orbis.py first)

Usage:
    python src/extraction/edc/run_edc.py --mode pilot
    python src/extraction/edc/run_edc.py --mode full --edc-path /path/to/edc
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pickle

from dotenv import load_dotenv

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # publication/

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# EDC Configuration
EDC_CONFIG = {
    # LLM settings - Mistral API
    "oie_llm": "mistral-api/mistral-small-latest",
    "sd_llm": "mistral-api/mistral-small-latest",
    "sc_llm": "mistral-api/mistral-small-latest",
    "ee_llm": "mistral-api/mistral-small-latest",

    # Embedding settings - Local multilingual model
    "sc_embedder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sr_embedder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",

    # Schema settings - Self-canonicalization mode (no predefined schema)
    "target_schema_path": None,
    "enrich_schema": True,

    # Refinement
    "refinement_iterations": 0,

    # Schema retriever (not used without adapter)
    "sr_adapter_path": None,

    # Logging
    "loglevel": logging.INFO,
}


class CheckpointManager:
    """Manages checkpoints for EDC pipeline phases."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = output_dir / "run_state.json"

    def save_state(self, phase: str, completed: bool = True):
        """Save current run state."""
        state = self.load_state()
        state["phases"][phase] = {
            "completed": completed,
            "timestamp": datetime.now().isoformat()
        }
        state["last_phase"] = phase
        state["last_update"] = datetime.now().isoformat()

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Dict[str, Any]:
        """Load run state or return default."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"phases": {}, "last_phase": None, "last_update": None}

    def save_checkpoint(self, phase: str, data: Dict[str, Any]):
        """Save checkpoint data for a phase."""
        checkpoint_file = self.checkpoint_dir / f"{phase}_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        self.save_state(phase, completed=True)

    def load_checkpoint(self, phase: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for a phase if it exists."""
        checkpoint_file = self.checkpoint_dir / f"{phase}_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

    def is_phase_completed(self, phase: str) -> bool:
        """Check if a specific phase is completed."""
        state = self.load_state()
        return state["phases"].get(phase, {}).get("completed", False)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    log_file = output_dir / f"edc_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def validate_environment(edc_root: Path) -> bool:
    """Check all required components are available."""
    errors = []

    if not os.environ.get("MISTRAL_API_KEY"):
        errors.append("MISTRAL_API_KEY not set in environment")

    if not edc_root.exists():
        errors.append(f"EDC framework not found at: {edc_root}")

    input_dir = PROJECT_ROOT / "data" / "edc_input"
    if not input_dir.exists():
        errors.append("No preprocessed input. Run preprocess_orbis.py first.")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return False

    return True


def setup_edc_patches(edc_root: Path, logger):
    """Setup EDC framework with Mistral API patches."""
    sys.path.insert(0, str(edc_root))

    from edc.edc_framework import EDC
    from edc.utils import llm_utils

    # Create Mistral utilities
    def is_model_mistral_api(model_name: str) -> bool:
        return model_name.startswith("mistral-api/")

    def mistral_chat_completion(model: str, messages: list, temperature: float = 0, max_tokens: int = 512):
        from mistralai import Mistral

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        model_name = model.replace("mistral-api/", "")

        response = client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # Patch OpenAI completion to handle Mistral
    original_openai_completion = llm_utils.openai_chat_completion

    def patched_completion(model, system_prompt, history, temperature=0, max_tokens=512):
        if is_model_mistral_api(model):
            messages = history
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + history
            return mistral_chat_completion(model, messages, temperature, max_tokens)
        return original_openai_completion(model, system_prompt, history, temperature, max_tokens)

    llm_utils.openai_chat_completion = patched_completion

    original_is_openai = llm_utils.is_model_openai

    def patched_is_openai(model_name):
        if is_model_mistral_api(model_name):
            return True
        return original_is_openai(model_name)

    llm_utils.is_model_openai = patched_is_openai

    logger.info("EDC framework patched for Mistral API")
    return EDC


def run_edc_pipeline(
    edc_path: str,
    mode: str = "pilot",
    output_dir: Optional[Path] = None,
    resume: bool = True
):
    """
    Run the EDC pipeline on Orbis corpus with checkpointing.

    Args:
        edc_path: Path to EDC framework installation
        mode: "pilot" (subset) or "full" (all articles)
        output_dir: Output directory
        resume: Whether to resume from checkpoint if available
    """
    edc_root = Path(edc_path)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / "outputs" / "edc" / f"{mode}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting EDC pipeline: mode={mode}")
    logger.info(f"EDC framework path: {edc_root}")
    logger.info(f"Output directory: {output_dir}")

    if not validate_environment(edc_root):
        logger.error("Environment validation failed. Aborting.")
        return None

    # Determine input file
    input_dir = PROJECT_ROOT / "data" / "edc_input"
    suffix = f"_{mode}" if mode == "pilot" else ""
    input_file = input_dir / f"orbis_articles{suffix}.txt"
    metadata_file = input_dir / f"orbis_metadata{suffix}.json"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Run preprocess_orbis.py first.")
        return None

    logger.info(f"Input file: {input_file}")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(output_dir)

    # Import and patch EDC
    try:
        EDC = setup_edc_patches(edc_root, logger)

        # Setup prompt template paths
        config = EDC_CONFIG.copy()
        config["oie_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "oie_template.txt")
        config["sd_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "sd_template.txt")
        config["sc_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "sc_template.txt")
        config["oie_refine_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "oie_r_template.txt")
        config["ee_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "ee_template.txt")
        config["em_prompt_template_file_path"] = str(edc_root / "prompt_templates" / "em_template.txt")

        # Use default few-shot examples
        config["oie_few_shot_example_file_path"] = str(edc_root / "few_shot_examples" / "example" / "oie_few_shot_examples.txt")
        config["sd_few_shot_example_file_path"] = str(edc_root / "few_shot_examples" / "example" / "sd_few_shot_examples.txt")
        config["oie_refine_few_shot_example_file_path"] = str(edc_root / "few_shot_examples" / "example" / "oie_few_shot_refine_examples.txt")
        config["ee_few_shot_example_file_path"] = str(edc_root / "few_shot_examples" / "example" / "ee_few_shot_examples.txt")

        edc = EDC(**config)
        logger.info("EDC framework initialized")
    except Exception as e:
        logger.error(f"Failed to initialize EDC: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Load input texts
    with open(input_file, 'r', encoding='utf-8') as f:
        input_texts = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Loaded {len(input_texts)} text chunks")

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== PHASE 1: OIE (Open Information Extraction) ==========
    if not checkpoint_mgr.is_phase_completed("oie"):
        logger.info("=" * 60)
        logger.info("PHASE 1: Open Information Extraction (OIE)")
        logger.info("=" * 60)
        try:
            oie_triplets_list, entity_hint_list, relation_hint_list = edc.oie(
                input_texts,
                free_model=False,
                previous_extracted_triplets_list=None,
            )
            checkpoint_mgr.save_checkpoint("oie", {
                "oie_triplets_list": oie_triplets_list,
                "entity_hint_list": entity_hint_list,
                "relation_hint_list": relation_hint_list,
            })
            logger.info(f"OIE completed: {len(oie_triplets_list)} chunks processed")
        except Exception as e:
            logger.error(f"OIE failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    else:
        logger.info("Loading OIE results from checkpoint...")
        oie_data = checkpoint_mgr.load_checkpoint("oie")
        oie_triplets_list = oie_data["oie_triplets_list"]
        entity_hint_list = oie_data["entity_hint_list"]
        relation_hint_list = oie_data["relation_hint_list"]

    # ========== PHASE 2: Schema Definition (SD) ==========
    if not checkpoint_mgr.is_phase_completed("sd"):
        logger.info("=" * 60)
        logger.info("PHASE 2: Schema Definition (SD)")
        logger.info("=" * 60)
        try:
            sd_dict_list = edc.schema_definition(
                input_texts,
                oie_triplets_list,
                free_model=False,
            )
            checkpoint_mgr.save_checkpoint("sd", {"sd_dict_list": sd_dict_list})
            logger.info(f"SD completed: {len(sd_dict_list)} chunks processed")
        except Exception as e:
            logger.error(f"SD failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    else:
        logger.info("Loading SD results from checkpoint...")
        sd_data = checkpoint_mgr.load_checkpoint("sd")
        sd_dict_list = sd_data["sd_dict_list"]

    # ========== PHASE 3: Schema Canonicalization (SC) ==========
    if not checkpoint_mgr.is_phase_completed("sc"):
        logger.info("=" * 60)
        logger.info("PHASE 3: Schema Canonicalization (SC)")
        logger.info("=" * 60)
        try:
            canon_triplets_list, canon_candidate_dict_list = edc.schema_canonicalization(
                input_texts,
                oie_triplets_list,
                sd_dict_list,
                free_model=False,
            )
            checkpoint_mgr.save_checkpoint("sc", {
                "canon_triplets_list": canon_triplets_list,
                "canon_candidate_dict_list": canon_candidate_dict_list,
            })
            logger.info(f"SC completed: {len(canon_triplets_list)} chunks processed")
        except Exception as e:
            logger.error(f"SC failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    else:
        logger.info("Loading SC results from checkpoint...")
        sc_data = checkpoint_mgr.load_checkpoint("sc")
        canon_triplets_list = sc_data["canon_triplets_list"]
        canon_candidate_dict_list = sc_data["canon_candidate_dict_list"]

    # ========== SAVE FINAL RESULTS ==========
    logger.info("=" * 60)
    logger.info("Saving final results...")
    logger.info("=" * 60)

    non_null_triplets_list = [
        [triple for triple in triplets if triple is not None]
        for triplets in canon_triplets_list
    ]

    # Save detailed results
    json_results_list = []
    for idx in range(len(oie_triplets_list)):
        result_json = {
            "index": idx,
            "input_text": input_texts[idx][:500] + "..." if len(input_texts[idx]) > 500 else input_texts[idx],
            "entity_hint": entity_hint_list[idx],
            "relation_hint": relation_hint_list[idx],
            "oie": oie_triplets_list[idx],
            "schema_definition": sd_dict_list[idx],
            "canonicalization_candidates": str(canon_candidate_dict_list[idx]),
            "schema_canonicalization": canon_triplets_list[idx],
        }
        json_results_list.append(result_json)

    with open(results_dir / "result_at_each_stage.json", "w", encoding="utf-8") as f:
        json.dump(json_results_list, f, indent=4, ensure_ascii=False)

    with open(results_dir / "canon_kg.txt", "w", encoding="utf-8") as f:
        for idx, canon_triplets in enumerate(non_null_triplets_list):
            f.write(str(canon_triplets))
            if idx != len(non_null_triplets_list) - 1:
                f.write("\n")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "mode": mode,
        "input_file": str(input_file),
        "total_chunks": len(input_texts),
        "total_triplets": sum(len(t) for t in non_null_triplets_list),
        "output_dir": str(output_dir),
        "status": "completed"
    }

    with open(output_dir / "run_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_dir}")
    logger.info(f"Total triplets extracted: {summary['total_triplets']}")
    logger.info("EDC pipeline completed successfully!")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDC on Orbis corpus")
    parser.add_argument("--edc-path", type=str, required=True,
                       help="Path to EDC framework installation")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                       help="Processing mode")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Custom output directory")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh, don't resume from checkpoint")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_edc_pipeline(
        edc_path=args.edc_path,
        mode=args.mode,
        output_dir=output_dir,
        resume=not args.no_resume
    )

    if result:
        print(f"\nEDC PIPELINE COMPLETED")
        print(f"Output directory: {result}")
    else:
        print("\nPipeline failed. Check logs for details.")
        sys.exit(1)
