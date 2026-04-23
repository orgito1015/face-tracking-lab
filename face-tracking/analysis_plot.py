import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = os.path.join("outputs", "face_tracking_log.csv")
DEFAULT_OUTPUT = os.path.join("outputs", "tracking_analysis.png")


def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_png = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    df = pd.read_csv(input_csv)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    df.plot(x="time", y="yaw_error", ax=axes[0], color="#01696F", title="Yaw Error vs Time")
    df.plot(x="time", y="cx", ax=axes[1], color="#A84B2F", title="Face X Position vs Time")
    df.plot(x="time", y="area", ax=axes[2], color="#848456", title="Face Area vs Time")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()