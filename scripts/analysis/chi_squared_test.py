"""
Chi-squared tests for the cultural vs. textual framing hypothesis.

Computes chi-squared goodness-of-fit tests for both KGGen and EDC methods,
testing the null hypothesis that cultural and textual framing weights are equal.

Results correspond to Table 5 (tab:framing) in the NSLP 2026 paper.

Output: prints results and optionally writes to outputs/tables/chi_squared_results.json
"""

import csv
import json
from pathlib import Path

from scipy.stats import chisquare

PUB_ROOT = Path(__file__).parent.parent.parent
KGGEN_HYPOTHESIS_FILE = PUB_ROOT / "outputs" / "tables" / "table_4_3_cultural_hypothesis.csv"
EDC_COMPARISON_FILE = PUB_ROOT / "outputs" / "comparison" / "edc_analysis_like_kggen.json"
OUTPUT_FILE = PUB_ROOT / "outputs" / "tables" / "chi_squared_results.json"


def load_kggen_weights():
    """Load cultural and textual weights from KGGen hypothesis CSV."""
    cultural_total = 0
    textual_total = 0

    with open(KGGEN_HYPOTHESIS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        current_frame = None
        for row in reader:
            if len(row) < 3:
                continue
            frame, connection, weight = row[0], row[1], row[2]

            if frame == "CULTURAL FRAME":
                current_frame = "cultural"
                continue
            elif frame == "TEXTUAL FRAME":
                current_frame = "textual"
                continue
            elif frame.startswith("RATIO"):
                continue

            if connection == "SUBTOTAL":
                if current_frame == "cultural":
                    cultural_total = int(weight)
                elif current_frame == "textual":
                    textual_total = int(weight)
            elif connection == "":
                continue

    return cultural_total, textual_total


def load_edc_weights():
    """Load cultural and textual weights from EDC comparison JSON."""
    with open(EDC_COMPARISON_FILE, "r", encoding="utf-8") as f:
        edc_data = json.load(f)

    hypothesis = edc_data["cultural_hypothesis"]
    return hypothesis["cultural_weight"], hypothesis["textual_weight"]


def run_chi_squared(cultural, textual, method_name):
    """Run chi-squared goodness-of-fit test against equal-weight null hypothesis."""
    observed = [cultural, textual]
    total = cultural + textual
    expected = [total / 2, total / 2]

    chi2, p_value = chisquare(observed, f_exp=expected)
    ratio = cultural / textual if textual > 0 else float("inf")

    print(f"\n{'=' * 50}")
    print(f"  {method_name}")
    print(f"{'=' * 50}")
    print(f"  Cultural weight:  {cultural}")
    print(f"  Textual weight:   {textual}")
    print(f"  Ratio:            {ratio:.2f}x")
    print(f"  Chi-squared:      {chi2:.1f}")
    print(f"  p-value:          {p_value:.2e}")
    print(f"  Significant:      {'Yes (p < .001)' if p_value < 0.001 else 'No'}")

    return {
        "method": method_name,
        "cultural_weight": cultural,
        "textual_weight": textual,
        "ratio": round(ratio, 2),
        "chi_squared": round(chi2, 1),
        "p_value": p_value,
        "significant": p_value < 0.001,
    }


def main():
    print("Cultural vs. Textual Framing: Chi-Squared Tests")
    print("(Table 5 in NSLP 2026 paper)")

    # Load data
    kggen_cultural, kggen_textual = load_kggen_weights()
    edc_cultural, edc_textual = load_edc_weights()

    # Run tests
    kggen_result = run_chi_squared(kggen_cultural, kggen_textual, "KGGen")
    edc_result = run_chi_squared(edc_cultural, edc_textual, "EDC")

    # Verify against paper claims
    print(f"\n{'=' * 50}")
    print("  Verification against paper (Table 5)")
    print(f"{'=' * 50}")

    checks = [
        ("KGGen cultural", kggen_result["cultural_weight"], 101),
        ("KGGen textual", kggen_result["textual_weight"], 40),
        ("KGGen ratio", kggen_result["ratio"], 2.52),
        ("KGGen chi-squared", kggen_result["chi_squared"], 26.4),
        ("EDC cultural", edc_result["cultural_weight"], 126),
        ("EDC textual", edc_result["textual_weight"], 58),
        ("EDC ratio", edc_result["ratio"], 2.17),
        ("EDC chi-squared", edc_result["chi_squared"], 25.1),
    ]

    all_pass = True
    for label, computed, expected in checks:
        match = computed == expected
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  {label}: {computed} (expected {expected}) [{status}]")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED - Script output matches paper.")
    else:
        print("  SOME CHECKS FAILED - Review discrepancies above.")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {"kggen": kggen_result, "edc": edc_result, "all_checks_passed": all_pass}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
