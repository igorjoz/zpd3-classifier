import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# CHANGED: Extends the original baseline training script with group-aware cross-validation.
CLASSES = [
    "mug",
    "flat_plate",
    "soup_plate",
    "bowl",
    "pot",
    "wine_glass",
    "saucepan",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DishDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(record["label_index"])


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def validate_records(records):
    missing = [record["image"] for record in records if not Path(record["image"]).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Missing {len(missing)} images, for example: {preview}")


def prepare_resized_cache(records_by_split, cache_dir, shortest_side):
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_sources = sorted(
        {Path(record["image"]) for records in records_by_split.values() for record in records}
    )
    cached_paths = {}
    for index, source in enumerate(all_sources, start=1):
        target = cache_dir / source.name
        cached_paths[source.as_posix()] = target
        if target.exists():
            continue
        with Image.open(source) as image:
            image = image.convert("RGB")
            scale = shortest_side / min(image.size)
            resized_size = tuple(round(dimension * scale) for dimension in image.size)
            image.resize(resized_size, Image.Resampling.BILINEAR).save(
                target, quality=95, optimize=True
            )
        if index % 250 == 0:
            print(f"Prepared resized cache for {index}/{len(all_sources)} images.")

    runtime_records = {}
    for split, records in records_by_split.items():
        runtime_records[split] = []
        for record in records:
            runtime_record = dict(record)
            runtime_record["image"] = str(cached_paths[Path(record["image"]).as_posix()])
            runtime_records[split].append(runtime_record)
    return runtime_records


def create_loaders(records_by_split, batch_size, workers, seed, image_cache_dir, cache_shortest_side):
    runtime_records = prepare_resized_cache(
        records_by_split, image_cache_dir, cache_shortest_side
    )
    train_transform, evaluation_transform = create_transforms()
    generators = {}
    loaders = {}
    for name, records, transform, shuffle in [
        ("train", runtime_records["train"], train_transform, True),
        ("validation", runtime_records["validation"], evaluation_transform, False),
        ("test", runtime_records["test"], evaluation_transform, False),
    ]:
        generators[name] = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(
            DishDataset(records, transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            generator=generators[name],
        )
    return loaders, runtime_records


# CHANGED: New helper functions for cross-validation data preparation.
def load_dataset_records(splits_dir):
    train_records = read_json(splits_dir / "train.json")
    validation_records = read_json(splits_dir / "validation.json")
    test_records = read_json(splits_dir / "test.json")
    for records in [train_records, validation_records, test_records]:
        validate_records(records)
    return train_records + validation_records, test_records


def create_cross_validation_splits(records, folds, seed):
    labels = np.asarray([record["label_index"] for record in records], dtype=int)
    groups = np.asarray([record["leakage_group"] for record in records], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = []
    for fold_index, (train_idx, validation_idx) in enumerate(
        splitter.split(records, labels, groups), start=1
    ):
        splits.append(
            {
                "fold": fold_index,
                "train": [records[i] for i in train_idx],
                "validation": [records[i] for i in validation_idx],
            }
        )
    return splits


def create_model():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, len(CLASSES))
    return model


def configure_trainable_layers(model, stage, unfreeze_blocks):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if stage == "fine_tune":
        for block in list(model.features.children())[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def keep_frozen_batch_norm_fixed(model, stage, unfreeze_blocks):
    if stage == "classifier":
        model.features.eval()
    else:
        for block in list(model.features.children())[:-unfreeze_blocks]:
            block.eval()


def create_optimizer(model, stage, learning_rate, feature_learning_rate, weight_decay):
    parameter_groups = [{"params": model.classifier.parameters(), "lr": learning_rate}]
    if stage == "fine_tune":
        feature_parameters = [
            parameter
            for parameter in model.features.parameters()
            if parameter.requires_grad
        ]
        parameter_groups.append({"params": feature_parameters, "lr": feature_learning_rate})
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def compute_metrics(targets, probabilities):
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    one_hot_targets = np.eye(len(CLASSES))[targets]
    try:
        auc = roc_auc_score(
            one_hot_targets, probabilities, average="macro", multi_class="ovr"
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "roc_auc_ovr_macro": float(auc),
    }, predictions


def run_epoch(model, loader, criterion, device, stage, unfreeze_blocks, optimizer=None):
    is_training = optimizer is not None
    if is_training:
        model.train()
        keep_frozen_batch_norm_fixed(model, stage, unfreeze_blocks)
    else:
        model.eval()

    loss_total = 0.0
    targets = []
    probabilities = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        loss_total += float(loss.item()) * images.size(0)
        targets.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

    metrics, predictions = compute_metrics(targets, probabilities)
    metrics["loss"] = loss_total / len(loader.dataset)
    return metrics, np.asarray(targets), np.asarray(probabilities), predictions


def annotate_matrix(ax, matrix):
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "#111827",
            )


def save_confusion_plot(path, epoch_title, train_targets, train_predictions, val_targets, val_predictions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, title, targets, predictions in [
        (axes[0], "Train", train_targets, train_predictions),
        (axes[1], "Validation", val_targets, val_predictions),
    ]:
        matrix = confusion_matrix(targets, predictions, labels=range(len(CLASSES)))
        image = ax.imshow(matrix, cmap="Blues")
        annotate_matrix(ax, matrix)
        ax.set_title(f"{title} confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(epoch_title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_roc_curves(ax, targets, probabilities, title):
    one_hot_targets = np.eye(len(CLASSES))[targets]
    for class_index, label in enumerate(CLASSES):
        if len(np.unique(one_hot_targets[:, class_index])) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            one_hot_targets[:, class_index], probabilities[:, class_index]
        )
        class_auc = np.trapezoid(true_positive_rate, false_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, label=f"{label} ({class_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)


def save_roc_plot(path, epoch_title, train_targets, train_probabilities, val_targets, val_probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    draw_roc_curves(axes[0], train_targets, train_probabilities, "Train ROC one-vs-rest")
    draw_roc_curves(
        axes[1], val_targets, val_probabilities, "Validation ROC one-vs-rest"
    )
    fig.suptitle(epoch_title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_plots(output_dir, prefix, display_name, targets, probabilities, predictions):
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASSES)))
    image = ax.imshow(matrix, cmap="Blues")
    annotate_matrix(ax, matrix)
    ax.set_title(f"{display_name} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{prefix}_confusion_matrix.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    draw_roc_curves(
        ax, targets, probabilities, f"{display_name} ROC one-vs-rest"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_diagnostics(path, metrics, targets, predictions):
    write_json(
        path,
        {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=range(len(CLASSES))
            ).tolist(),
            "class_order": CLASSES,
            "classification_report": classification_report(
                targets,
                predictions,
                labels=range(len(CLASSES)),
                target_names=CLASSES,
                output_dict=True,
                zero_division=0,
            ),
        },
    )


def save_history_csv(path, history):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(path, history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    metric_specs = [
        ("loss", "Cross-entropy loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1-score"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc_ovr_macro", "ROC-AUC macro OvR"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], marker="o", label="train")
        ax.plot(
            epochs,
            [row[f"validation_{metric}"] for row in history],
            marker="o",
            label="validation",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()

    lr_ax = axes.flat[-1]
    lr_ax.plot(
        epochs, [row["classifier_learning_rate"] for row in history], marker="o", label="classifier"
    )
    lr_ax.plot(
        epochs, [row["feature_learning_rate"] for row in history], marker="o", label="features"
    )
    lr_ax.set_title("Learning rate")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_yscale("log")
    lr_ax.grid(alpha=0.25)
    lr_ax.legend()
    fig.suptitle("MobileNetV3-Large training and validation metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def checkpoint_sort_key(item):
    return (-item["validation_macro_f1"], item["validation_loss"], item["epoch"])


def update_checkpoints(checkpoint_dir, model, epoch, stage, train_metrics, validation_metrics, config, top):
    name = (
        f"mobilenet_v3_large_epoch_{epoch:02d}_"
        f"val_f1_{validation_metrics['macro_f1']:.4f}.pt"
    )
    checkpoint_path = checkpoint_dir / name
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_name": "MobileNetV3-Large",
            "classes": CLASSES,
            "model_state_dict": model.state_dict(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "config": config,
        },
        checkpoint_path,
    )
    top.append(
        {
            "epoch": epoch,
            "path": str(checkpoint_path),
            "stage": stage,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
        }
    )
    top.sort(key=checkpoint_sort_key)
    while len(top) > 3:
        discarded = top.pop()
        Path(discarded["path"]).unlink(missing_ok=True)
    return top


def format_metric(value):
    return f"{value:.4f}" if value is not None else "-"


# CHANGED: New reporting function for cross-validation results.
def write_crossvalidation_report(path, config, fold_summaries):
    lines = [
        "# MobileNetV3-Large - raport cross-validation",
        "",
        "## Konfiguracja eksperymentu",
        "",
        f"- Liczba foldów: `{config['folds']}`",
        f"- Epoki: `{config['epochs']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Warmup epochs: `{config['warmup_epochs']}`",
        f"- Unfreeze blocks: `{config['unfreeze_blocks']}`",
        f"- Learning rate classifier: `{config['learning_rate']}`",
        f"- Learning rate feature: `{config['feature_learning_rate']}`",
        f"- Weight decay: `{config['weight_decay']}`",
        "",
        "## Wyniki foldów",
        "",
        "| Fold | Validation macro F1 | Validation loss | Best epoch |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for summary in fold_summaries:
        lines.append(
            f"| {summary['fold']} | {summary['best_macro_f1']:.4f} | {summary['best_loss']:.4f} | {summary['best_epoch']} |"
        )
    mean_f1 = np.mean([summary["best_macro_f1"] for summary in fold_summaries])
    std_f1 = np.std([summary["best_macro_f1"] for summary in fold_summaries])
    lines.extend(
        [
            "",
            "## Agregacja wyników",
            "",
            f"- Średnie validation macro F1: `{mean_f1:.4f}`",
            f"- Odchylenie standardowe validation macro F1: `{std_f1:.4f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# CHANGED: New utility functions for per-fold outputs.
def build_fold_output_dirs(root_dir, fold):
    fold_dir = root_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    per_epoch_dir = fold_dir / "per_epoch"
    per_epoch_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir, per_epoch_dir


def summarize_fold(history, top_checkpoints):
    best = top_checkpoints[0]
    best_history = next(row for row in history if row["epoch"] == best["epoch"])
    return {
        "fold": best["epoch"],
        "best_epoch": best["epoch"],
        "best_macro_f1": best_history["validation_macro_f1"],
        "best_loss": best_history["validation_loss"],
        "best_checkpoint": best["path"],
    }


def main():
    # CHANGED: Parser default directories and cross-validation arguments added.
    parser = argparse.ArgumentParser(
        description="Train and validate a MobileNetV3-Large model using group-aware cross-validation."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data_splits"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("crossvalidation_outputs") / "mobilenet_v3_large",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("MODELS_IMPROVED"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feature-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-cache-dir", type=Path, default=Path("resized_image_cache"))
    parser.add_argument("--cache-shortest-side", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 10:
        raise ValueError("This assignment requires at least 10 training epochs.")
    if args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be smaller than epochs.")
    if args.folds < 2:
        raise ValueError("folds must be at least 2.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "MobileNetV3-Large",
        "pretrained_weights": "MobileNet_V3_Large_Weights.DEFAULT (ImageNet)",
        "classes": CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "unfreeze_blocks": args.unfreeze_blocks,
        "learning_rate": args.learning_rate,
        "feature_learning_rate": args.feature_learning_rate,
        "weight_decay": args.weight_decay,
        "input_size": [224, 224],
        "resized_image_cache_shortest_side": args.cache_shortest_side,
        "folds": args.folds,
        "checkpoint_selection": "validation_macro_f1_desc_then_validation_loss_asc",
    }
    write_json(args.output_dir / "config.json", config)

    crossval_records, test_records = load_dataset_records(args.splits_dir)
    fold_splits = create_cross_validation_splits(crossval_records, args.folds, args.seed)
    split_summary = read_json(args.splits_dir / "split_summary.json")
    print(
        "Total records used for cross-validation:",
        len(crossval_records),
        "test records:",
        len(test_records),
    )
    print(
        "Class distributions in cross-validation dataset:",
        dict(Counter(record["label"] for record in crossval_records)),
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fold_summaries = []
    for fold_details in fold_splits:
        fold = fold_details["fold"]
        print(f"Starting fold {fold}/{args.folds}")
        fold_dir, per_epoch_dir = build_fold_output_dirs(args.output_dir, fold)
        fold_model_dir = args.models_dir / f"fold_{fold:02d}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        records_by_split = {
            "train": fold_details["train"],
            "validation": fold_details["validation"],
            "test": test_records,
        }
        loaders, _ = create_loaders(
            records_by_split,
            args.batch_size,
            args.workers,
            args.seed,
            args.image_cache_dir,
            args.cache_shortest_side,
        )

        model = create_model().to(device)
        criterion = nn.CrossEntropyLoss()
        history = []
        top_checkpoints = []
        stage = "classifier"
        configure_trainable_layers(model, stage, args.unfreeze_blocks)
        optimizer = create_optimizer(
            model, stage, args.learning_rate, args.feature_learning_rate, args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.35, patience=2, min_lr=1e-6
        )

        for epoch in range(1, args.epochs + 1):
            if epoch == args.warmup_epochs + 1:
                stage = "fine_tune"
                configure_trainable_layers(model, stage, args.unfreeze_blocks)
                optimizer = create_optimizer(
                    model,
                    stage,
                    args.learning_rate * 0.35,
                    args.feature_learning_rate,
                    args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.35, patience=2, min_lr=1e-6
                )
                print(f"Epoch {epoch}: unfreezing final {args.unfreeze_blocks} feature blocks.")

            started = time.perf_counter()
            train_metrics, train_targets, train_probabilities, train_predictions = run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                stage,
                args.unfreeze_blocks,
                optimizer=optimizer,
            )
            validation_metrics, val_targets, val_probabilities, val_predictions = run_epoch(
                model,
                loaders["validation"],
                criterion,
                device,
                stage,
                args.unfreeze_blocks,
            )
            scheduler.step(validation_metrics["macro_f1"])
            elapsed = time.perf_counter() - started
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            row = {
                "epoch": epoch,
                "stage": stage,
                "duration_seconds": round(elapsed, 2),
                "classifier_learning_rate": learning_rates[0],
                "feature_learning_rate": learning_rates[1] if len(learning_rates) > 1 else 0.0,
            }
            for prefix, metrics in [
                ("train", train_metrics),
                ("validation", validation_metrics),
            ]:
                for metric, value in metrics.items():
                    row[f"{prefix}_{metric}"] = value
            history.append(row)
            top_checkpoints = update_checkpoints(
                fold_model_dir,
                model,
                epoch,
                stage,
                train_metrics,
                validation_metrics,
                config,
                top_checkpoints,
            )
            save_confusion_plot(
                per_epoch_dir / f"epoch_{epoch:02d}_confusion_matrix.png",
                f"Fold {fold} - Epoch {epoch} - {stage}",
                train_targets,
                train_predictions,
                val_targets,
                val_predictions,
            )
            save_roc_plot(
                per_epoch_dir / f"epoch_{epoch:02d}_roc.png",
                f"Fold {fold} - Epoch {epoch} - {stage}",
                train_targets,
                train_probabilities,
                val_targets,
                val_probabilities,
            )
            print(
                f"Fold {fold} epoch {epoch:02d}/{args.epochs} [{stage}] "
                f"train loss={train_metrics['loss']:.4f} f1={train_metrics['macro_f1']:.4f} | "
                f"val loss={validation_metrics['loss']:.4f} "
                f"f1={validation_metrics['macro_f1']:.4f} "
                f"auc={validation_metrics['roc_auc_ovr_macro']:.4f} | {elapsed:.1f}s"
            )

        save_history_csv(fold_dir / "history.csv", history)
        write_json(fold_dir / "history.json", history)
        write_json(fold_dir / "selected_checkpoints.json", top_checkpoints)
        save_history_plot(fold_dir / "training_curves.png", history)

        best_checkpoint = torch.load(top_checkpoints[0]["path"], map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        selected_validation_metrics, selected_val_targets, selected_val_probabilities, selected_val_predictions = run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            best_checkpoint["stage"],
            args.unfreeze_blocks,
        )
        save_evaluation_diagnostics(
            fold_dir / "selected_validation_diagnostics.json",
            selected_validation_metrics,
            selected_val_targets,
            selected_val_predictions,
        )
        save_evaluation_plots(
            fold_dir,
            "selected_validation",
            "Validation",
            selected_val_targets,
            selected_val_probabilities,
            selected_val_predictions,
        )

        fold_summary = {
            "fold": fold,
            "best_epoch": best_checkpoint["epoch"],
            "best_macro_f1": selected_validation_metrics["macro_f1"],
            "best_loss": selected_validation_metrics["loss"],
            "best_checkpoint": top_checkpoints[0]["path"],
        }
        fold_summaries.append(fold_summary)

    write_json(args.output_dir / "fold_summaries.json", fold_summaries)

    best_overall = max(fold_summaries, key=lambda summary: summary["best_macro_f1"])
    best_checkpoint = torch.load(best_overall["best_checkpoint"], map_location=device)
    write_crossvalidation_report(
        args.output_dir / "REPORT.md",
        config,
        fold_summaries,
    )

    print(f"Best overall fold: {best_overall['fold']} with validation macro F1 = {best_overall['best_macro_f1']:.4f}")
    print(f"Wrote cross-validation artefakty do {args.output_dir}")


if __name__ == "__main__":
    main()
