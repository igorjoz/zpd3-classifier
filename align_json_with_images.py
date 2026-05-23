import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse


DEFAULT_INPUT = Path("project-3-at-2026-04-25-19-49-76eb6179-mini.json")
DEFAULT_IMAGES_DIR = Path("images")
DEFAULT_OUTPUT = Path("project-3-at-2026-04-25-19-49-76eb6179-mini-matched.json")
DEFAULT_REPORT = Path("project-3-at-2026-04-25-19-49-76eb6179-mini-match-report.json")


def filename_from_reference(reference):
    parsed = urlparse(str(reference))
    path = parsed.path if parsed.scheme else str(reference)
    return PurePosixPath(unquote(path).replace("\\", "/")).name


def build_image_index(images_dir):
    image_paths = sorted(path for path in images_dir.rglob("*") if path.is_file())
    index = {}
    duplicate_names = []
    for path in image_paths:
        key = path.name.casefold()
        if key in index:
            duplicate_names.append(path.name)
        else:
            index[key] = path

    if duplicate_names:
        duplicated = ", ".join(sorted(set(duplicate_names)))
        raise ValueError(f"Image filenames are not unique: {duplicated}")
    return index


def align_records(records, image_index):
    matched_records = []
    matched_images = set()
    missing_references = Counter()

    for record in records:
        image_reference = record.get("image", "")
        filename = filename_from_reference(image_reference)
        local_image = image_index.get(filename.casefold())
        if local_image is None:
            missing_references[filename] += 1
            continue

        matched_record = dict(record)
        matched_record["source_image"] = image_reference
        matched_record["image"] = local_image.as_posix()
        matched_records.append(matched_record)
        matched_images.add(local_image.name.casefold())

    unannotated_images = sorted(
        path.as_posix()
        for key, path in image_index.items()
        if key not in matched_images
    )
    return matched_records, missing_references, unannotated_images


def build_report(records, matched_records, missing_references, unannotated_images):
    input_images = {
        filename_from_reference(record.get("image", "")).casefold()
        for record in records
    }
    matched_images = {
        filename_from_reference(record.get("image", "")).casefold()
        for record in matched_records
    }
    missing_images = [
        {"filename": filename, "annotation_rows": count}
        for filename, count in sorted(missing_references.items())
    ]
    return {
        "annotation_rows_input": len(records),
        "unique_images_referenced_input": len(input_images),
        "annotation_rows_matched": len(matched_records),
        "unique_images_matched": len(matched_images),
        "annotation_rows_dropped_missing_image": sum(missing_references.values()),
        "missing_images_referenced_by_json": missing_images,
        "unannotated_local_images_count": len(unannotated_images),
        "unannotated_local_images": unannotated_images,
        "note": (
            "Local images without annotations are listed only; they cannot be "
            "added to a supervised labelled dataset automatically."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Match annotation JSON rows to local images and rewrite their image paths."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected the input JSON root to contain a list of annotation rows.")

    image_index = build_image_index(args.images_dir)
    matched, missing, unannotated = align_records(records, image_index)
    report = build_report(records, matched, missing, unannotated)

    args.output.write_text(
        json.dumps(matched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input annotation rows: {report['annotation_rows_input']}")
    print(f"Matched annotation rows: {report['annotation_rows_matched']}")
    print(
        "Dropped annotation rows for missing images: "
        f"{report['annotation_rows_dropped_missing_image']}"
    )
    print(f"Unannotated local images: {report['unannotated_local_images_count']}")
    print(f"Wrote matched data to {args.output}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
