"""Operational SOC simulation from binary threshold-ablation results.

This converts model rates into analyst-facing quantities: alerts/day, false
alerts/day, true attacks detected, and missed attacks under an assumed traffic
volume and malicious base rate.

Run:
    .venv/bin/python src/soc_simulation.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import data as D

RESULTS = D.REPO_ROOT / "results"
THRESHOLD_CSV = RESULTS / "threshold_ablation.csv"
CSV_PATH = RESULTS / "soc_simulation.csv"
REPORT_PATH = RESULTS / "soc_simulation.md"


@dataclass(frozen=True)
class SocScenario:
    """Daily traffic scenario for operational IDS evaluation."""

    daily_flows: int = 1_000_000
    malicious_rate: float = 0.005


def simulate_row(
    normal_recall: float,
    attack_recall: float,
    scenario: SocScenario = SocScenario(),
) -> dict[str, float]:
    """Convert recall rates to daily SOC workload numbers."""
    benign = scenario.daily_flows * (1.0 - scenario.malicious_rate)
    malicious = scenario.daily_flows * scenario.malicious_rate
    false_positive_rate = 1.0 - normal_recall
    false_alerts = benign * false_positive_rate
    true_alerts = malicious * attack_recall
    missed_attacks = malicious * (1.0 - attack_recall)
    total_alerts = false_alerts + true_alerts
    precision = true_alerts / total_alerts if total_alerts else 0.0
    return {
        "false_alerts_per_day": float(false_alerts),
        "true_alerts_per_day": float(true_alerts),
        "missed_attacks_per_day": float(missed_attacks),
        "total_alerts_per_day": float(total_alerts),
        "operational_precision": float(precision),
    }


def run(
    threshold_csv: str | None = None,
    scenario: SocScenario = SocScenario(),
) -> pd.DataFrame:
    """Run the SOC simulation from threshold-ablation results."""
    path = THRESHOLD_CSV if threshold_csv is None else D.REPO_ROOT / threshold_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `.venv/bin/python src/threshold_ablation.py` first."
        )

    thresholds = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, r in thresholds.iterrows():
        sim = simulate_row(
            normal_recall=float(r["normal_recall"]),
            attack_recall=float(r["attack_recall"]),
            scenario=scenario,
        )
        rows.append(
            {
                "model": r["model"],
                "threshold_name": r["threshold_name"],
                "threshold": r["threshold"],
                "macro_f1": r["macro_f1"],
                "normal_recall": r["normal_recall"],
                "attack_recall": r["attack_recall"],
                **sim,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(render_markdown(out, scenario), encoding="utf-8")
    return out


def render_markdown(rows: pd.DataFrame, scenario: SocScenario) -> str:
    lines = [
        "# Operational SOC Simulation",
        "",
        f"Scenario: {scenario.daily_flows:,} daily flows, "
        f"{scenario.malicious_rate * 100:.2f}% malicious.",
        "",
        "| Model | Threshold rule | Macro-F1 | Attack recall | False alerts/day | True alerts/day | Missed attacks/day | Total alerts/day | Operational precision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['model']} | {r['threshold_name']} | {r['macro_f1']:.4f} | "
            f"{r['attack_recall']:.4f} | {r['false_alerts_per_day']:.0f} | "
            f"{r['true_alerts_per_day']:.0f} | {r['missed_attacks_per_day']:.0f} | "
            f"{r['total_alerts_per_day']:.0f} | {r['operational_precision']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This table often changes which model looks best. A threshold that improves "
        "macro-F1 can still produce an impossible analyst workload if benign false "
        "alerts are too high.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    rows = run()
    print(rows.to_string(index=False))
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
