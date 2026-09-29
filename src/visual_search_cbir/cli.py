from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from visual_search_cbir import __version__
from visual_search_cbir.config import FeatureConfig, SearchConfig
from visual_search_cbir.dataset import scan_dataset
from visual_search_cbir.evaluation import evaluate_index
from visual_search_cbir.index import CBIRIndex, build_index
from visual_search_cbir.retrieval import compute_query_vectors, search_top_k


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-search",
        description="Classical content-based image retrieval using SIFT, Harris, BoVW, and Rocchio feedback.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check local runtime capabilities.")

    scan = subparsers.add_parser("scan", help="Inspect the supported dataset classes.")
    scan.add_argument("--dataset", type=Path, required=True)

    build = subparsers.add_parser("build-index", help="Build and save a BoVW search index.")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("artifacts/cbir_index.joblib"))

    search = subparsers.add_parser("search", help="Retrieve visually similar images.")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", type=Path, required=True)
    search.add_argument("--dataset", type=Path)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--sift-weight", type=float, default=0.60)
    search.add_argument("--harris-weight", type=float, default=0.40)

    evaluate = subparsers.add_parser("evaluate", help="Measure retrieval quality using class labels.")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path)
    evaluate.add_argument("--max-queries", type=int)

    subparsers.add_parser("app", help="Launch the Tkinter GUI.")
    return parser


def _doctor() -> int:
    print(f"Visual Search CBIR v{__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"SIFT available: {hasattr(cv2, 'SIFT_create')}")
    try:
        import tkinter  # noqa: F401

        print("Tkinter: available")
    except Exception:
        print("Tkinter: unavailable")
    return 0


def main() -> None:
    args = _parser().parse_args()

    if args.command == "doctor":
        raise SystemExit(_doctor())

    if args.command == "scan":
        scan = scan_dataset(args.dataset)
        print(json.dumps({
            "root": str(scan.root),
            "total_images": scan.total_images,
            "counts": scan.counts,
            "missing_classes": list(scan.missing_classes),
        }, indent=2))
        return

    if args.command == "build-index":
        index = build_index(
            args.dataset,
            args.output,
            FeatureConfig(),
            status_callback=print,
        )
        print(f"Index ready: {args.output} ({index.image_count} images)")
        return

    if args.command == "search":
        index = CBIRIndex.load(args.index)
        sift_query, harris_query = compute_query_vectors(index, args.query)
        hits = search_top_k(
            index,
            sift_query,
            harris_query,
            dataset_root=args.dataset,
            search_config=SearchConfig(
                top_k=args.top_k,
                sift_weight=args.sift_weight,
                harris_weight=args.harris_weight,
            ),
        )
        print(json.dumps([
            {"rank": rank, "score": hit.score, "label": hit.label, "path": str(hit.path)}
            for rank, hit in enumerate(hits, start=1)
        ], indent=2))
        return

    if args.command == "evaluate":
        index = CBIRIndex.load(args.index)
        result = evaluate_index(
            index,
            dataset_root=args.dataset,
            max_queries=args.max_queries,
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "app":
        from visual_search_cbir.gui import run_gui

        run_gui()
        return


if __name__ == "__main__":
    main()
