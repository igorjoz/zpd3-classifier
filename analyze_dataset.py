import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

DEFAULT_INPUTS = [Path("project-3-at-2026-04-25-19-49-76eb6179-mini.json")]
NO_VALUE = "__none__"
NO_FLAG = "no_flag"
MULTIPLE_SAME_CLASS = "multiple_objects_same_class"
CLASS_COLORS = {
    "bowl": "#2563eb",
    "mug": "#0891b2",
    "pot": "#7c3aed",
    "flat_plate": "#16a34a",
    "saucepan": "#ea580c",
    "wine_glass": "#db2777",
    "soup_plate": "#ca8a04",
}
STATUS_COLORS = {
    "classifiable": "#16a34a",
    "multiple_different_classes": "#f97316",
    "unreadable": "#dc2626",
}
METRIC_COLORS = {
    "image_status": "#0f766e",
    "dish_class_with_missing": "#7c3aed",
    "image_flags": "#ea580c",
    "dish_class_classifiable_only": "#2563eb",
}


def normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, list):
            return ";".join(str(item).strip() for item in text if str(item).strip())
        if text is not None:
            return str(text).strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return ";".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def image_stem(uri):
    name = uri.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def parse_object_ids(value):
    return ";".join(re.findall(r"[A-Za-z]+\d+", value or ""))


def load_annotations(paths):
    rows = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            object_id = normalize_value(item.get("object_id"))
            rows.append(
                {
                    "source_file": path.name,
                    "task_id": int(item["id"]),
                    "image": normalize_value(item.get("image")),
                    "image_name": image_stem(normalize_value(item.get("image"))),
                    "annotator": int(item["annotator"]),
                    "annotation_id": int(item["annotation_id"]),
                    "created_at": normalize_value(item.get("created_at")),
                    "lead_time_seconds": float(item.get("lead_time") or 0.0),
                    "image_status": normalize_value(item.get("image_status")),
                    "dish_class": normalize_value(item.get("dish_class")),
                    "image_flags": normalize_value(item.get("image_flags")),
                    "object_id_raw": object_id,
                    "object_ids": parse_object_ids(object_id),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(["annotation_id", "task_id", "annotator"])
    df["image_flags_binary"] = df["image_flags"].where(df["image_flags"] != "", NO_FLAG)
    df["dish_class_with_none"] = df["dish_class"].where(df["dish_class"] != "", NO_VALUE)
    return df


def majority(values):
    values = [value for value in values if value != ""]
    if not values:
        return "", 0, False, ""
    counts = Counter(values)
    top_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top_count)
    distribution = ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))
    return winners[0], top_count, len(winners) > 1, distribution


def merge_to_image_level(df):
    rows = []
    for task_id, group in df.groupby("task_id", sort=True):
        status, status_votes, status_tie, status_counts = majority(group["image_status"])
        class_votes = group.loc[
            (group["image_status"] == "classifiable") & (group["dish_class"] != ""),
            "dish_class",
        ]
        dish_class, class_votes_count, class_tie, class_counts = majority(class_votes)
        final_flag, flag_votes, flag_tie, flag_counts = majority(group["image_flags_binary"])
        object_ids = sorted({oid for ids in group["object_ids"] for oid in ids.split(";") if oid})

        status_values = set(group["image_status"])
        class_values = set(class_votes)
        flag_values = set(group["image_flags_binary"])
        review_reasons = []
        if len(status_values) > 1:
            review_reasons.append("status_disagreement")
        if len(class_values) > 1:
            review_reasons.append("class_disagreement")
        if len(flag_values) > 1:
            review_reasons.append("flag_disagreement")
        if status != "classifiable":
            review_reasons.append(f"status_{status}")
        if class_tie:
            review_reasons.append("class_tie")
        if status == "classifiable" and not dish_class:
            review_reasons.append("missing_class")

        rows.append(
            {
                "task_id": task_id,
                "image": group["image"].iloc[0],
                "image_name": group["image_name"].iloc[0],
                "annotator_count": group["annotator"].nunique(),
                "annotators": ";".join(map(str, sorted(group["annotator"].unique()))),
                "final_status": status,
                "status_vote_count": status_votes,
                "status_tie": status_tie,
                "status_vote_distribution": status_counts,
                "final_dish_class": dish_class,
                "class_vote_count": class_votes_count,
                "class_tie": class_tie,
                "class_vote_distribution": class_counts,
                "final_image_flag": "" if final_flag == NO_FLAG else final_flag,
                "flag_vote_count": flag_votes,
                "flag_tie": flag_tie,
                "flag_vote_distribution": flag_counts,
                "object_ids": ";".join(object_ids),
                "primary_object_id": object_ids[0] if object_ids else f"task_{task_id}",
                "has_status_disagreement": len(status_values) > 1,
                "has_class_disagreement": len(class_values) > 1,
                "has_flag_disagreement": len(flag_values) > 1,
                "needs_review": bool(review_reasons),
                "review_reasons": ";".join(review_reasons),
                "mean_lead_time_seconds": group["lead_time_seconds"].mean(),
                "max_lead_time_seconds": group["lead_time_seconds"].max(),
            }
        )
    return pd.DataFrame(rows)


def reliability_tables(df):
    metrics = {
        "image_status": "image_status",
        "dish_class_with_missing": "dish_class_with_none",
        "image_flags": "image_flags_binary",
    }
    annotators = sorted(df["annotator"].unique())
    pairwise_rows = []
    fleiss_rows = []

    for metric_name, column in metrics.items():
        pivot = df.pivot_table(index="task_id", columns="annotator", values=column, aggfunc="last")
        full = pivot.dropna(subset=annotators)
        values = full[annotators].astype(str).values.tolist()
        table, _ = aggregate_raters(values)
        fleiss_rows.append(
            {
                "field": metric_name,
                "overlap_items": len(values),
                "fleiss_kappa": float(fleiss_kappa(table, method="fleiss")),
            }
        )

        for idx, left in enumerate(annotators):
            for right in annotators[idx + 1 :]:
                pair = pivot.dropna(subset=[left, right])
                pairwise_rows.append(
                    {
                        "field": metric_name,
                        "annotator_pair": f"{left}-{right}",
                        "overlap_items": len(pair),
                        "cohen_kappa": float(cohen_kappa_score(pair[left].astype(str), pair[right].astype(str))),
                    }
                )

    status_pivot = df.pivot_table(index="task_id", columns="annotator", values="image_status", aggfunc="last")
    classifiable_ids = status_pivot.dropna(subset=annotators).query(
        " and ".join(f"`{annotator}` == 'classifiable'" for annotator in annotators)
    ).index
    class_df = df[df["task_id"].isin(classifiable_ids) & (df["dish_class"] != "")]
    class_pivot = class_df.pivot_table(index="task_id", columns="annotator", values="dish_class", aggfunc="last")
    class_full = class_pivot.dropna(subset=annotators)
    values = class_full[annotators].astype(str).values.tolist()
    table, _ = aggregate_raters(values)
    fleiss_rows.append(
        {
            "field": "dish_class_classifiable_only",
            "overlap_items": len(values),
            "fleiss_kappa": float(fleiss_kappa(table, method="fleiss")),
        }
    )
    for idx, left in enumerate(annotators):
        for right in annotators[idx + 1 :]:
            pair = class_pivot.dropna(subset=[left, right])
            pairwise_rows.append(
                {
                    "field": "dish_class_classifiable_only",
                    "annotator_pair": f"{left}-{right}",
                    "overlap_items": len(pair),
                    "cohen_kappa": float(cohen_kappa_score(pair[left].astype(str), pair[right].astype(str))),
                }
            )

    return pd.DataFrame(pairwise_rows), pd.DataFrame(fleiss_rows)


def disagreement_examples(df, limit=30):
    rows = []
    for task_id, group in df.groupby("task_id", sort=True):
        status_values = set(group["image_status"])
        class_values = set(group.loc[group["dish_class"] != "", "dish_class"])
        flag_values = set(group["image_flags_binary"])
        if len(status_values) <= 1 and len(class_values) <= 1 and len(flag_values) <= 1:
            continue
        rows.append(
            {
                "task_id": task_id,
                "image": group["image"].iloc[0],
                "statuses_by_annotator": "; ".join(
                    f"A{row.annotator}:{row.image_status}" for row in group.itertuples()
                ),
                "classes_by_annotator": "; ".join(
                    f"A{row.annotator}:{row.dish_class or NO_VALUE}" for row in group.itertuples()
                ),
                "flags_by_annotator": "; ".join(
                    f"A{row.annotator}:{row.image_flags_binary}" for row in group.itertuples()
                ),
                "lead_times_seconds": "; ".join(
                    f"A{row.annotator}:{row.lead_time_seconds:.1f}" for row in group.itertuples()
                ),
            }
        )
    return pd.DataFrame(rows).head(limit)


def object_group_stats(image_df):
    exploded = image_df[image_df["object_ids"] != ""].copy()
    exploded["object_id"] = exploded["object_ids"].str.split(";")
    exploded = exploded.explode("object_id")
    stats = (
        exploded.groupby("object_id")
        .agg(
            image_count=("task_id", "count"),
            classes=("final_dish_class", lambda values: ";".join(sorted(set(values) - {""}))),
            class_count=("final_dish_class", lambda values: len(set(values) - {""})),
            review_count=("needs_review", "sum"),
            mean_lead_time_seconds=("mean_lead_time_seconds", "mean"),
        )
        .reset_index()
    )
    return stats.sort_values(["image_count", "object_id"], ascending=[False, True])


def multiple_same_class_stats(df, image_df):
    rows = [
        {
            "level": "final_images",
            "group": "all",
            "total": len(image_df),
            "multiple_same_class_count": int((image_df["final_image_flag"] == MULTIPLE_SAME_CLASS).sum()),
        },
        {
            "level": "annotation_rows",
            "group": "all",
            "total": len(df),
            "multiple_same_class_count": int((df["image_flags_binary"] == MULTIPLE_SAME_CLASS).sum()),
        },
    ]

    for class_name, group in image_df[image_df["final_dish_class"] != ""].groupby("final_dish_class"):
        rows.append(
            {
                "level": "final_images_by_class",
                "group": class_name,
                "total": len(group),
                "multiple_same_class_count": int((group["final_image_flag"] == MULTIPLE_SAME_CLASS).sum()),
            }
        )

    for annotator, group in df.groupby("annotator"):
        rows.append(
            {
                "level": "annotation_rows_by_annotator",
                "group": f"A{annotator}",
                "total": len(group),
                "multiple_same_class_count": int((group["image_flags_binary"] == MULTIPLE_SAME_CLASS).sum()),
            }
        )

    stats = pd.DataFrame(rows)
    stats["multiple_same_class_percent"] = 100 * stats["multiple_same_class_count"] / stats["total"]
    return stats


def save_tables(
    output_dir,
    df,
    image_df,
    pairwise_df,
    fleiss_df,
    examples_df,
    object_df,
):
    df.to_csv(output_dir / "annotation_rows_normalized.csv", index=False)
    image_df.to_csv(output_dir / "final_image_dataset.csv", index=False)
    pairwise_df.to_csv(output_dir / "reliability_pairwise_cohen_kappa.csv", index=False)
    fleiss_df.to_csv(output_dir / "reliability_fleiss_kappa.csv", index=False)
    examples_df.to_csv(output_dir / "disagreement_examples.csv", index=False)
    object_df.to_csv(output_dir / "object_group_stats.csv", index=False)

    annotator_workload = df.groupby("annotator").agg(
        annotation_rows=("task_id", "count"),
        unique_tasks=("task_id", "nunique"),
        median_lead_time_seconds=("lead_time_seconds", "median"),
        mean_lead_time_seconds=("lead_time_seconds", "mean"),
    )
    annotator_workload.to_csv(output_dir / "annotator_workload.csv")

    multiple_same_class_stats(df, image_df).to_csv(output_dir / "multiple_same_class_stats.csv", index=False)


def pretty_label(value):
    return str(value).replace("_", " ").title()


def setup_plots():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def style_axis(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=24 if subtitle else 12)
    if subtitle:
        ax.text(
            0,
            1.005,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#64748b",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, figures_dir, name):
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax, values, fmt="{:.0f}"):
    values = list(values)
    offset = max(values) * 0.012 if values else 0.1
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + offset,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )


def write_figure_index(figures_dir, figure_index):
    lines = ["# Figure Index", ""]
    lines.extend(f"- `{filename}`: {description}" for filename, description in figure_index)
    (figures_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(
    matrix,
    figures_dir,
    name,
    title,
    fmt=".2f",
    xlabel=None,
    ylabel=None,
):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    vmax = max(1.0, float(matrix.max().max()))
    image = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), [pretty_label(col) for col in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), [pretty_label(row) for row in matrix.index])
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=12)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=12)
    style_axis(ax, title)
    ax.grid(False)
    for row_idx, row in enumerate(matrix.index):
        for col_idx, col in enumerate(matrix.columns):
            value = matrix.loc[row, col]
            text = "-" if pd.isna(value) else format(float(value), fmt)
            color = "white" if pd.notna(value) and float(value) > vmax * 0.55 else "#0f172a"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir, name)


def write_plots(
    output_dir,
    df,
    image_df,
    pairwise_df,
    fleiss_df,
    object_df,
):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plots()
    figure_index = []

    class_counts = image_df.loc[
        (image_df["final_status"] == "classifiable") & (image_df["final_dish_class"] != ""),
        "final_dish_class",
    ].value_counts()
    status_counts = image_df["final_status"].value_counts()

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    values = class_counts.sort_values()
    ax.barh(
        [pretty_label(label) for label in values.index],
        values.values,
        color=[CLASS_COLORS.get(label, "#64748b") for label in values.index],
    )
    add_bar_labels(ax, values.values)
    style_axis(ax, "Final Class Counts", "After majority vote per image")
    ax.set_xlabel("Images")
    save_figure(fig, figures_dir, "01_final_class_counts")
    figure_index.append(("01_final_class_counts.png", "Final class counts."))

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    values = class_counts.sort_values(ascending=False)
    ax.pie(
        values.values,
        labels=[pretty_label(label) for label in values.index],
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        colors=[CLASS_COLORS.get(label, "#64748b") for label in values.index],
        labeldistance=1.08,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"color": "#334155", "fontsize": 12},
    )
    ax.set_title("Final Class Share", loc="left", fontsize=13, fontweight="bold", pad=18)
    ax.text(
        0,
        1.0,
        "After majority vote per image",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    ax.axis("equal")
    save_figure(fig, figures_dir, "01_final_class_share_pie")
    figure_index.append(("01_final_class_share_pie.png", "Final class share as a pie chart."))

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    values = status_counts.sort_values()
    ax.barh(
        [pretty_label(label) for label in values.index],
        values.values,
        color=[STATUS_COLORS.get(label, "#64748b") for label in values.index],
    )
    add_bar_labels(ax, values.values)
    style_axis(ax, "Final Image Status")
    ax.set_xlabel("Images")
    save_figure(fig, figures_dir, "01_final_image_status")
    figure_index.append(("01_final_image_status.png", "Final image-status counts."))

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    values = status_counts.sort_values(ascending=False)
    total = values.sum()
    wedges, _ = ax.pie(
        values.values,
        labels=None,
        startangle=90,
        counterclock=False,
        colors=[STATUS_COLORS.get(label, "#64748b") for label in values.index],
        wedgeprops={"width": 0.55, "linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Final Image Status Share", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.text(
        0,
        0.06,
        f"{100 * values.iloc[0] / total:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        -0.12,
        pretty_label(values.index[0]),
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    legend_labels = [
        f"{pretty_label(label)}: {int(count)} ({100 * count / total:.1f}%)"
        for label, count in values.items()
    ]
    ax.legend(
        wedges,
        legend_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=10,
    )
    ax.axis("equal")
    save_figure(fig, figures_dir, "01_final_image_status_pie")
    figure_index.append(("01_final_image_status_pie.png", "Final image-status share as a pie chart."))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fleiss_plot = fleiss_df.sort_values("fleiss_kappa")
    ax.barh(
        [pretty_label(label) for label in fleiss_plot["field"]],
        fleiss_plot["fleiss_kappa"],
        color=[METRIC_COLORS.get(label, "#64748b") for label in fleiss_plot["field"]],
    )
    for patch, row in zip(ax.patches, fleiss_plot.itertuples()):
        ax.text(row.fleiss_kappa + 0.015, patch.get_y() + patch.get_height() / 2, f"{row.fleiss_kappa:.3f}", va="center")
    style_axis(ax, "Three-Observer Reliability")
    ax.set_xlabel("Fleiss' kappa")
    ax.set_xlim(0, 1.05)
    save_figure(fig, figures_dir, "03_fleiss_kappa")
    figure_index.append(("03_fleiss_kappa.png", "Fleiss kappa on the subset annotated by every observer."))

    pairwise_matrix = pairwise_df.pivot(index="field", columns="annotator_pair", values="cohen_kappa")
    write_heatmap(pairwise_matrix, figures_dir, "04_pairwise_cohen_kappa", "Pairwise Cohen's Kappa", ".3f")
    figure_index.append(("04_pairwise_cohen_kappa.png", "Cohen kappa for each annotator pair."))

    total_images = len(image_df)
    review_images = int(image_df["needs_review"].sum())
    clean_images = total_images - review_images
    review_reasons = (
        image_df.loc[image_df["review_reasons"] != "", "review_reasons"]
        .str.split(";")
        .explode()
        .value_counts()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [0.9, 1.35]})
    ax = axes[0]
    outcome = pd.Series({"Clean": clean_images, "Needs review": review_images})
    colors = ["#16a34a", "#ef4444"]
    bars = ax.barh(outcome.index, outcome.values, color=colors)
    for bar, count in zip(bars, outcome.values):
        ax.text(
            bar.get_width() + total_images * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({100 * count / total_images:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color="#334155",
        )
    style_axis(ax, "Review Outcome", "Unique image count")
    ax.set_xlabel("Images")
    ax.set_xlim(0, total_images * 1.16)

    ax = axes[1]
    values = review_reasons.sort_values()
    ax.barh([pretty_label(label) for label in values.index], values.values, color="#f97316")
    for patch, (reason, count) in zip(ax.patches, values.items()):
        ax.text(
            patch.get_width() + max(values.values) * 0.018,
            patch.get_y() + patch.get_height() / 2,
            f"{int(count)} ({100 * count / total_images:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color="#334155",
        )
    style_axis(ax, "Review Reasons", "Reasons can overlap, so they do not sum to Needs review")
    ax.set_xlabel("Images")
    ax.set_xlim(0, max(values.values) * 1.32)
    save_figure(fig, figures_dir, "05_disagreements_and_review")
    figure_index.append(("05_disagreements_and_review.png", "Unique review outcome and overlapping review reasons."))

    workload = (
        df.groupby("annotator")
        .agg(
            annotation_rows=("task_id", "count"),
            unique_tasks=("task_id", "nunique"),
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x_labels = [f"A{annotator}" for annotator in workload.index]
    bars = ax.bar(x_labels, workload["annotation_rows"], color="#2563eb", width=0.55)
    for bar, row in zip(bars, workload.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + workload["annotation_rows"].max() * 0.015,
            f"{int(row.annotation_rows)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
        )
    style_axis(ax, "Annotator Workload")
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Annotation rows")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    save_figure(fig, figures_dir, "06_annotator_workload")
    figure_index.append(("06_annotator_workload.png", "Annotation rows submitted by each annotator."))

    lead_times = df["lead_time_seconds"].dropna()
    lead_times = lead_times[lead_times <= lead_times.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(lead_times, bins=30, color="#0f766e", edgecolor="white", linewidth=0.6)
    median_time = lead_times.median()
    p90_time = lead_times.quantile(0.90)
    ax.axvline(median_time, color="#f97316", linewidth=2, label=f"median {median_time:.1f}s")
    ax.axvline(p90_time, color="#dc2626", linewidth=2, linestyle="--", label=f"90th pct {p90_time:.1f}s")
    style_axis(ax, "Annotation Time Distribution", "Clipped at 99th percentile for readability")
    ax.set_xlabel("Lead time [seconds]")
    ax.set_ylabel("Annotation rows")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    save_figure(fig, figures_dir, "07_annotation_time_distribution")
    figure_index.append(("07_annotation_time_distribution.png", "Distribution of annotation lead times."))

    status_by_annotator = pd.crosstab(df["annotator"], df["image_status"], normalize="index")
    status_by_annotator = status_by_annotator.reindex(columns=STATUS_COLORS.keys(), fill_value=0)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bottom = pd.Series(0.0, index=status_by_annotator.index)
    for status in status_by_annotator.columns:
        values = status_by_annotator[status]
        bars = ax.bar(
            [f"A{idx}" for idx in status_by_annotator.index],
            values,
            bottom=bottom,
            label=pretty_label(status),
            color=STATUS_COLORS[status],
        )
        for bar, value, base in zip(bars, values, bottom):
            if value < 0.035:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        bottom += values
    style_axis(ax, "Status Choices by Annotator")
    ax.set_ylabel("Percent of annotator rows")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save_figure(fig, figures_dir, "08_status_by_annotator")
    figure_index.append(("08_status_by_annotator.png", "Status-label proportions by annotator."))

    top_objects = object_df.head(15).sort_values("image_count")
    fig, ax = plt.subplots(figsize=(10, 6))
    clean = top_objects["image_count"] - top_objects["review_count"]
    ax.barh(top_objects["object_id"], clean, color="#60a5fa", label="No review flag")
    ax.barh(top_objects["object_id"], top_objects["review_count"], left=clean, color="#ef4444", label="Needs review")
    for row in top_objects.itertuples():
        ax.text(row.image_count + 0.3, row.object_id, f"{int(row.image_count)}", va="center", fontsize=9)
    style_axis(ax, "Most Repeated Object IDs")
    ax.set_xlabel("Images")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, figures_dir, "09_top_object_groups")
    figure_index.append(("09_top_object_groups.png", "Repeated object groups and their review burden."))

    final_lookup = image_df.set_index("task_id")["final_dish_class"]
    comparable = df[(df["image_status"] == "classifiable") & (df["dish_class"] != "")].copy()
    comparable["final_dish_class"] = comparable["task_id"].map(final_lookup)
    comparable = comparable[comparable["final_dish_class"] != ""]
    class_order = class_counts.index.tolist()
    confusion = pd.crosstab(comparable["dish_class"], comparable["final_dish_class"]).reindex(
        index=class_order,
        columns=class_order,
        fill_value=0,
    )
    write_heatmap(
        confusion,
        figures_dir,
        "10_annotator_vs_final_class",
        "Annotator Labels vs Final Class",
        ".0f",
        xlabel="Final class after majority vote",
        ylabel="Single annotator label",
    )
    figure_index.append(("10_annotator_vs_final_class.png", "Class-label dependencies and confusions."))

    final_flag_count = int((image_df["final_image_flag"] == MULTIPLE_SAME_CLASS).sum())
    raw_flag_count = int((df["image_flags_binary"] == MULTIPLE_SAME_CLASS).sum())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [0.8, 1.4]})

    ax = axes[0]
    totals = pd.Series(
        {
            "No final flag": len(image_df) - final_flag_count,
            "Multiple same class": final_flag_count,
        }
    )
    bars = ax.barh(totals.index, totals.values, color=["#94a3b8", "#ea580c"])
    for bar, count in zip(bars, totals.values):
        ax.text(
            bar.get_width() + len(image_df) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({100 * count / len(image_df):.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color="#334155",
        )
    style_axis(ax, "Multiple Same-Class Objects", "Final image-level flag")
    ax.set_xlabel("Images")
    ax.set_xlim(0, len(image_df) * 1.18)

    ax = axes[1]
    by_class = (
        image_df[image_df["final_dish_class"] != ""]
        .groupby("final_dish_class")
        .agg(
            total=("task_id", "count"),
            flagged=("final_image_flag", lambda values: int((values == MULTIPLE_SAME_CLASS).sum())),
        )
    )
    by_class["rate"] = by_class["flagged"] / by_class["total"]
    by_class = by_class.sort_values("rate")
    ax.barh(
        [pretty_label(label) for label in by_class.index],
        by_class["rate"],
        color=[CLASS_COLORS.get(label, "#64748b") for label in by_class.index],
    )
    for patch, row in zip(ax.patches, by_class.itertuples()):
        ax.text(
            patch.get_width() + 0.008,
            patch.get_y() + patch.get_height() / 2,
            f"{row.rate:.1%} ({int(row.flagged)}/{int(row.total)})",
            va="center",
            fontsize=9,
            color="#334155",
        )
    style_axis(
        ax,
        "Rate by Final Class",
        f"Raw annotator usage: {raw_flag_count}/{len(df)} rows ({100 * raw_flag_count / len(df):.1f}%)",
    )
    ax.set_xlabel("Images with final multiple-same-class flag")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, max(0.1, by_class["rate"].max() * 1.28))
    save_figure(fig, figures_dir, "11_multiple_same_class_objects")
    figure_index.append(
        (
            "11_multiple_same_class_objects.png",
            "Frequency of the multiple-objects-same-class flag overall and by final class.",
        )
    )

    write_figure_index(figures_dir, figure_index)


def main():
    parser = argparse.ArgumentParser(description="Create dataset tables and plots.")
    parser.add_argument("--input", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.input)
    images = merge_to_image_level(annotations)
    pairwise, fleiss = reliability_tables(annotations)
    examples = disagreement_examples(annotations)
    objects = object_group_stats(images)

    save_tables(args.output_dir, annotations, images, pairwise, fleiss, examples, objects)
    write_plots(args.output_dir, annotations, images, pairwise, fleiss, objects)
    print(f"Wrote tables and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
