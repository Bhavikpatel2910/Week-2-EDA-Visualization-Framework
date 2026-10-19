"""Plot generation and Markdown report output."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .eda import (
    build_quality_notes,
    categorical_summary,
    dataset_overview,
    iqr_outlier_flags,
    numeric_summary,
    profile_columns,
    correlation_matrix,
)

sns.set_theme(style="whitegrid", context="notebook")


def _safe_filename(value: str) -> str:
    result = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    result = "_".join(part for part in result.split("_") if part)
    return result[:70] or "column"


def create_figures(df: pd.DataFrame, figures_dir: Path, max_categories: int = 15) -> list[str]:
    """Create general-purpose plots and return relative file paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    for col in numeric_cols[:12]:
        clean = df[col].dropna()
        if clean.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(clean, bins="auto", edgecolor="black", alpha=0.8)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(str(col))
        ax.set_ylabel("Count")
        fig.tight_layout()
        name = f"{_safe_filename(col)}_histogram.png"
        fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.boxplot(clean, vert=False)
        ax.set_title(f"Box plot of {col}")
        ax.set_xlabel(str(col))
        fig.tight_layout()
        name = f"{_safe_filename(col)}_boxplot.png"
        fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    for col in categorical_cols[:8]:
        counts = df[col].astype("string").fillna("<MISSING>").value_counts().head(max_categories)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, max(4, min(8, 0.32 * len(counts)))))
        ax.barh(counts.index.astype(str), counts.values)
        ax.invert_yaxis()
        ax.set_title(f"Top categories in {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(str(col))
        fig.tight_layout()
        name = f"{_safe_filename(col)}_categories.png"
        fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        if not corr.empty:
            fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(corr.columns)), max(5, 0.65 * len(corr.columns))))
            image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)), labels=corr.index)
            if len(corr.columns) <= 10:
                for i in range(len(corr.index)):
                    for j in range(len(corr.columns)):
                        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, label="Pearson correlation")
            ax.set_title("Correlation heat map (numeric columns)")
            fig.tight_layout()
            name = "numeric_correlation_heatmap.png"
            fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(name)

        # A simple exploratory scatter plot for the first two numeric columns.
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        pair = df[[x_col, y_col]].dropna()
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(pair[x_col], pair[y_col], alpha=0.65)
            ax.set_title(f"{y_col} vs {x_col}")
            fig.tight_layout()
            name = "numeric_scatter_plot.png"
            fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(name)

    return saved


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    figure_names: list[str],
) -> Path:
    """Write CSV summaries and an EDA Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = dataset_overview(df)
    profile = profile_columns(df)
    numeric = numeric_summary(df)
    categorical = categorical_summary(df)
    outliers = iqr_outlier_flags(df)
    corr = correlation_matrix(df)
    notes = build_quality_notes(df, profile, outliers)

    profile.to_csv(output_dir / "data_profile.csv", index=False)
    numeric.to_csv(output_dir / "numeric_summary.csv", index=False)
    categorical.to_csv(output_dir / "categorical_summary.csv", index=False)
    outliers.to_csv(output_dir / "outlier_flags.csv", index=False)
    if not corr.empty:
        corr.to_csv(output_dir / "correlation_matrix.csv")

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## 1. Purpose and scope",
        "",
        "This report was generated by a reusable EDA framework. Interpretations are exploratory and should be validated with domain knowledge.",
        "",
        "## 2. Dataset overview",
        "",
        f"- **Rows:** {overview['rows']:,}",
        f"- **Columns:** {overview['columns']:,}",
        f"- **Exact duplicate rows:** {overview['duplicate_rows']:,}",
        f"- **Duplicate column names:** {overview['duplicate_column_names']:,}",
        f"- **Approximate DataFrame memory:** {overview['memory_mb']} MB",
        "",
        "## 3. Data quality observations",
        "",
    ]
    lines.extend([f"- {note}" for note in notes])
    lines += [
        "",
        "## 4. Column profile",
        "",
        "| Column | Type | Non-null | Missing | Missing % | Unique values |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in profile.iterrows():
        lines.append(
            f"| {row['column']} | {row['dtype']} | {row['non_null_count']} | "
            f"{row['missing_count']} | {row['missing_percent']}% | {row['unique_values']} |"
        )

    lines += ["", "## 5. Numeric summary", ""]
    if numeric.empty:
        lines.append("No numeric columns were detected.")
    else:
        lines.append(numeric.to_markdown(index=False))

    lines += ["", "## 6. Categorical summary", ""]
    if categorical.empty:
        lines.append("No non-numeric columns were detected.")
    else:
        lines.append(categorical.head(50).to_markdown(index=False))
        if len(categorical) > 50:
            lines.append("")
            lines.append(f"Showing the first 50 category-frequency rows out of {len(categorical)}.")

    lines += [
        "",
        "## 7. Visualizations",
        "",
        "Charts are saved in the `figures/` directory. The following plots were generated:",
        "",
    ]
    if figure_names:
        lines.extend([f"- `../figures/{name}`" for name in figure_names])
    else:
        lines.append("No plots could be generated from the available columns.")

    lines += [
        "",
        "## 8. Interpretation and limitations",
        "",
        "- IQR outlier flags are prompts for investigation, not automatic evidence of data errors.",
        "- Missing values should be handled according to their meaning and cause; do not replace them with zero by default.",
        "- Correlation is not proof of causation.",
        "- Automated summaries do not know the domain context, sampling design, or measurement process.",
        "- For predictive modelling, split data appropriately and fit preprocessing steps only on training data to avoid leakage.",
        "",
        "## 9. Recommended next steps",
        "",
        "1. Review `data_profile.csv` and investigate fields with substantial missingness or unexpected types.",
        "2. Verify duplicate rows and potential outliers against source records and domain rules.",
        "3. Select bivariate and multivariate comparisons that answer the project's specific question.",
        "4. Add captions explaining the meaning of each important chart.",
        "5. Document any data cleaning, assumptions, exclusions, and remaining limitations.",
        "",
        "## 10. Reproducibility",
        "",
        "Run `python run_eda.py --input data/your_dataset.csv --output reports/my_eda` to regenerate the report and tables.",
        "",
    ]
    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
