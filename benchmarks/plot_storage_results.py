
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "storage"
CHARTS_DIR = RESULTS_DIR / "charts"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Load benchmark results
summary = pd.read_csv(
    RESULTS_DIR / "benchmark_summary.csv"
)

# Display labels
labels = {
    "csv_plain": "CSV Plain",
    "csv_gzip": "CSV GZIP",
    "pickle_plain": "Pickle Plain",
    "pickle_gzip": "Pickle GZIP",
    "parquet_plain": "Parquet Plain",
    "parquet_snappy": "Parquet Snappy",
}

order = list(labels.keys())


# Chart function
def create_chart(data, title, ylabel, filename):

    data = data.set_index("configuration").reindex(order)

    names = [labels[name] for name in order]
    values = data["median_seconds"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, values)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Storage Configuration")

    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()

    output_path = CHARTS_DIR / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print("Saved:", output_path)


# Chart 1: File size
file_sizes = (
    summary[["configuration", "file_size_mb"]]
    .drop_duplicates("configuration")
    .set_index("configuration")
    .reindex(order)
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    [labels[name] for name in order],
    file_sizes["file_size_mb"],
)

ax.set_title("Storage Format File Size Comparison")
ax.set_xlabel("Storage Configuration")
ax.set_ylabel("File Size (MB)")
ax.tick_params(axis="x", rotation=30)

for bar, value in zip(bars, file_sizes["file_size_mb"]):
    ax.annotate(
        f"{value:.2f}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
        ),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

ax.set_ylim(
    0,
    file_sizes["file_size_mb"].max() * 1.15,
)

fig.tight_layout()

fig.savefig(
    CHARTS_DIR / "01_file_size_comparison.png",
    dpi=300,
)

plt.close(fig)

print("Saved: 01_file_size_comparison.png")


# Chart 2: Write time
create_chart(
    summary[summary["operation"] == "write"],
    "Storage Format Write Time Comparison",
    "Median Write Time (seconds)",
    "02_write_time_comparison.png",
)


# Chart 3: Full read time
create_chart(
    summary[summary["operation"] == "full_read"],
    "Storage Format Full Read Time Comparison",
    "Median Full Read Time (seconds)",
    "03_full_read_comparison.png",
)


# Chart 4: Single-column read time
create_chart(
    summary[summary["operation"] == "column_read"],
    "Storage Format Revenue Column Read Comparison",
    "Median Column Read Time (seconds)",
    "04_column_read_comparison.png",
)

print("\nAll four charts generated successfully!")
