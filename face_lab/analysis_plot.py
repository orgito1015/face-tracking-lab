"""
Session 2 — 3-panel time-series plot from rc_log.csv.

Usage:
    python analysis_plot.py --input rc_log.csv --output outputs/tracking_analysis.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot yaw, face X position, and face area from rc_log.csv"
    )
    parser.add_argument("--input", default="rc_log.csv",
                        help="Path to the CSV log file (default: rc_log.csv)")
    parser.add_argument("--output", default=None,
                        help="Extra output PNG path (optional)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"time", "yaw", "cx", "area"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    for col in ("time", "yaw", "cx", "area"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "yaw", "cx", "area"])

    if df.empty:
        raise ValueError("No valid rows found in the input CSV.")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["time"], df["yaw"], color="#01696F")
    axes[0].set_title("Yaw Command vs Time")
    axes[0].set_ylabel("Yaw")

    axes[1].plot(df["time"], df["cx"], color="#A84B2F")
    axes[1].set_title("Face X Position vs Time")
    axes[1].set_ylabel("X (px)")

    axes[2].plot(df["time"], df["area"], color="#848456")
    axes[2].set_title("Face Area vs Time")
    axes[2].set_ylabel("Area (px²)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()

    # Always save to outputs/tracking_analysis.png
    Path("outputs").mkdir(exist_ok=True)
    default_out = Path("outputs") / "tracking_analysis.png"
    fig.savefig(default_out, dpi=150)
    print(f"Saved: {default_out}")

    # Also save to --output if explicitly provided
    if args.output is not None:
        extra = Path(args.output)
        extra.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra, dpi=150)
        print(f"Saved: {extra}")

    plt.show()


if __name__ == "__main__":
    main()
