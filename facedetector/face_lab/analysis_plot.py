import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot yaw, face X, and face area from rc_log.csv")
    parser.add_argument("--input", default="rc_log.csv")
    parser.add_argument("--output", default="tracking_analysis.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"t", "yaw", "cx", "area"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["yaw"] = pd.to_numeric(df["yaw"], errors="coerce")
    df["cx"] = pd.to_numeric(df["cx"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["t", "yaw", "cx", "area"])

    if df.empty:
        raise ValueError("Input log has no valid rows to plot")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df["t"], df["yaw"], color="#1b9e77", linewidth=1.3)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Yaw")
    axes[0].set_title("Yaw vs Time")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["t"], df["cx"], color="#d95f02", linewidth=1.3)
    axes[1].axhline(320, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Face X")
    axes[1].set_title("Face Center X vs Time")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["t"], df["area"], color="#7570b3", linewidth=1.3)
    axes[2].axhspan(8000, 14000, color="#a6cee3", alpha=0.25)
    axes[2].set_ylabel("Face Area")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Face Area vs Time")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
