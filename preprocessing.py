import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("project-3-at-2026-04-25-19-49-76eb6179-mini-matched.json")
DEFAULT_OUTPUT_DIR = Path("data_splits")
DEFAULT_SEED = 42
CLASSES = [
    "mug",
    "flat_plate",
    "soup_plate",
    "bowl",
    "pot",
    "wine_glass",
    "saucepan",
]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
OBJECT_ID_PATTERN = re.compile(r"[A-Za-z]+\d+")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def majority_vote(values):
    counts = Counter(value for value in values if value)
    if not counts:
        return None, {}, True
    largest = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == largest)
    return winners[0], dict(sorted(counts.items())), len(winners) > 1


def extract_object_ids(rows):
    object_ids = set()
    for row in rows:
        value = str(row.get("object_id") or "")
        object_ids.update(OBJECT_ID_PATTERN.findall(value))
    return sorted(object_ids)


def aggregate_annotations(rows):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["id"])].append(row)

    accepted = []
    rejected = []
    for task_id in sorted(grouped_rows):
        task_rows = grouped_rows[task_id]
        images = {str(row["image"]) for row in task_rows}
        if len(images) != 1:
            raise ValueError(f"Task {task_id} references more than one local image.")

        status, status_distribution, status_tie = majority_vote(
            row.get("image_status") for row in task_rows
        )
        if status_tie:
            rejected.append(
                {
                    "task_id": task_id,
                    "image": next(iter(images)),
                    "reason": "status_tie",
                    "status_distribution": status_distribution,
                }
            )
            continue
        if status != "classifiable":
            rejected.append(
                {
                    "task_id": task_id,
                    "image": next(iter(images)),
                    "reason": status or "missing_status",
                    "status_distribution": status_distribution,
                }
            )
            continue

        eligible_labels = [
            row.get("dish_class")
            for row in task_rows
            if row.get("image_status") == "classifiable"
            and row.get("dish_class") in LABEL_TO_INDEX
        ]
        label, label_distribution, label_tie = majority_vote(eligible_labels)
        if label is None or label_tie:
            rejected.append(
                {
                    "task_id": task_id,
                    "image": next(iter(images)),
                    "reason": "class_missing_or_tie",
                    "status_distribution": status_distribution,
                    "class_distribution": label_distribution,
                }
            )
            continue

        accepted.append(
            {
                "task_id": task_id,
                "image": next(iter(images)),
                "label": label,
                "label_index": LABEL_TO_INDEX[label],
                "object_ids": extract_object_ids(task_rows),
                "annotator_count": len(task_rows),
                "status_vote_distribution": status_distribution,
                "class_vote_distribution": label_distribution,
            }
        )
    return accepted, rejected


def add_leakage_groups(samples):
    union_find = UnionFind()
    for sample in samples:
        object_ids = sample["object_ids"]
        if not object_ids:
            continue
        for object_id in object_ids[1:]:
            union_find.union(object_ids[0], object_id)

    for sample in samples:
        if sample["object_ids"]:
            sample["leakage_group"] = union_find.find(sample["object_ids"][0])
        else:
            sample["leakage_group"] = f"task_{sample['task_id']}"


def split_score(split_counts, split_totals, class_totals, total_samples):
    score = 0.0
    for split, ratio in SPLIT_RATIOS.items():
        target_total = total_samples * ratio
        score += 0.3 * ((split_totals[split] - target_total) ** 2) / total_samples
        for label in CLASSES:
            target = class_totals[label] * ratio
            score += ((split_counts[split][label] - target) ** 2) / max(
                class_totals[label], 1
            )
    return score


def assign_splits(samples, seed, attempts=300):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["leakage_group"]].append(sample)
    groups = list(grouped.items())
    class_totals = Counter(sample["label"] for sample in samples)
    total_samples = len(samples)
    best_assignment = None
    best_score = float("inf")

    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        candidate_groups = list(groups)
        rng.shuffle(candidate_groups)
        candidate_groups.sort(key=lambda item: len(item[1]), reverse=True)
        split_counts = {split: Counter() for split in SPLIT_RATIOS}
        split_totals = Counter()
        assignment = {}

        for group_id, group_samples in candidate_groups:
            group_counts = Counter(sample["label"] for sample in group_samples)
            group_size = len(group_samples)
            candidates = []
            split_order = list(SPLIT_RATIOS)
            rng.shuffle(split_order)
            for split in split_order:
                split_counts[split].update(group_counts)
                split_totals[split] += group_size
                candidate_score = split_score(
                    split_counts, split_totals, class_totals, total_samples
                )
                split_counts[split].subtract(group_counts)
                split_totals[split] -= group_size
                candidates.append((candidate_score, split))

            _, chosen_split = min(candidates, key=lambda item: item[0])
            assignment[group_id] = chosen_split
            split_counts[chosen_split].update(group_counts)
            split_totals[chosen_split] += group_size

        candidate_score = split_score(
            split_counts, split_totals, class_totals, total_samples
        )
        if candidate_score < best_score:
            best_score = candidate_score
            best_assignment = assignment

    for sample in samples:
        sample["split"] = best_assignment[sample["leakage_group"]]


def create_summary(samples, rejected, input_rows, seed):
    summary = {
        "seed": seed,
        "split_ratios_requested": SPLIT_RATIOS,
        "annotation_rows_input": len(input_rows),
        "accepted_unique_images": len(samples),
        "rejected_unique_images": len(rejected),
        "classes": CLASSES,
        "splits": {},
        "rejected_reasons": dict(Counter(item["reason"] for item in rejected)),
    }
    groups_by_split = {}
    for split in SPLIT_RATIOS:
        subset = [sample for sample in samples if sample["split"] == split]
        groups_by_split[split] = {sample["leakage_group"] for sample in subset}
        summary["splits"][split] = {
            "images": len(subset),
            "ratio": len(subset) / len(samples),
            "leakage_groups": len(groups_by_split[split]),
            "class_counts": dict(
                sorted(Counter(sample["label"] for sample in subset).items())
            ),
        }

    overlaps = {}
    split_names = list(SPLIT_RATIOS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}_{right}"] = sorted(
                groups_by_split[left] & groups_by_split[right]
            )
    summary["leakage_group_overlaps"] = overlaps
    summary["leakage_check_passed"] = not any(overlaps.values())
    return summary


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path, samples):
    fields = [
        "task_id",
        "image",
        "label",
        "label_index",
        "object_ids",
        "leakage_group",
        "annotator_count",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {field: sample[field] for field in fields}
            row["object_ids"] = ";".join(row["object_ids"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate annotation votes and create leakage-safe dataset splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    annotation_rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(annotation_rows, list):
        raise ValueError("The annotation input JSON must contain a list.")

    samples, rejected = aggregate_annotations(annotation_rows)
    add_leakage_groups(samples)
    assign_splits(samples, args.seed)
    summary = create_summary(samples, rejected, annotation_rows, args.seed)
    if not summary["leakage_check_passed"]:
        raise RuntimeError("Detected an object group shared by more than one split.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dataset_clean.json", samples)
    write_json(args.output_dir / "rejected_images.json", rejected)
    write_json(args.output_dir / "split_summary.json", summary)
    write_csv(args.output_dir / "dataset_clean.csv", samples)
    for split in SPLIT_RATIOS:
        subset = [sample for sample in samples if sample["split"] == split]
        write_json(args.output_dir / f"{split}.json", subset)

    print(f"Accepted images: {len(samples)}")
    print(f"Rejected images: {len(rejected)}")
    for split in SPLIT_RATIOS:
        details = summary["splits"][split]
        print(
            f"{split}: {details['images']} images "
            f"({details['ratio']:.1%}), {details['class_counts']}"
        )
    print(f"Leakage check passed: {summary['leakage_check_passed']}")
    print(f"Wrote processed data to {args.output_dir}")


if __name__ == "__main__":
    main()
