from __future__ import annotations

import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from visual_search_cbir.config import FeatureConfig, SearchConfig
from visual_search_cbir.dataset import detect_dataset_root, scan_dataset
from visual_search_cbir.feedback import rocchio_refine
from visual_search_cbir.index import CBIRIndex, build_index
from visual_search_cbir.retrieval import SearchHit, compute_query_vectors, search_top_k


class VisualSearchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Visual Search - SIFT + Harris CBIR")
        self.geometry("1200x780")

        self.feature_config = FeatureConfig()
        self.dataset_root: Path | None = None
        self.index: CBIRIndex | None = None
        self.index_path: Path | None = None
        self.query_path: Path | None = None
        self.sift_query: np.ndarray | None = None
        self.harris_query: np.ndarray | None = None
        self.result_items: list[SearchHit] = []
        self.result_checks: list[tk.IntVar] = []
        self.result_images: list[ImageTk.PhotoImage | None] = []
        self.query_image: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Button(top, text="Select Dataset Root", command=self.select_dataset).pack(side=tk.LEFT)
        self.dataset_label = tk.Label(top, text="Dataset: (not selected)")
        self.dataset_label.pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Build/Load Index", command=self.build_or_load_index).pack(
            side=tk.LEFT, padx=10
        )
        tk.Button(top, text="Select Query Image", command=self.select_query).pack(
            side=tk.LEFT, padx=10
        )
        self.query_label = tk.Label(top, text="Query: (not selected)")
        self.query_label.pack(side=tk.LEFT)

        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10)
        tk.Label(controls, text="w_SIFT").pack(side=tk.LEFT)
        self.sift_weight = tk.DoubleVar(value=0.60)
        tk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.sift_weight,
            length=200,
        ).pack(side=tk.LEFT, padx=8)
        tk.Label(controls, text="w_Harris").pack(side=tk.LEFT)
        self.harris_weight = tk.DoubleVar(value=0.40)
        tk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.harris_weight,
            length=200,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(controls, text="Search Top 10", command=self.search).pack(side=tk.LEFT, padx=10)
        tk.Button(controls, text="Refine (Feedback)", command=self.refine).pack(
            side=tk.LEFT, padx=10
        )

        self.status_label = tk.Label(self, text="Status: Ready", anchor="w")
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        main = tk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(main, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left, text="Query Preview").pack(anchor="w")
        self.query_canvas = tk.Label(left, bd=1, relief=tk.SOLID, width=260, height=260)
        self.query_canvas.pack(pady=8)

        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(right, text="Results (Top 10) - select up to 5 for feedback").pack(anchor="w")
        self.results_frame = tk.Frame(right)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        log_frame = tk.Frame(self)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)
        tk.Label(log_frame, text="Log").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    def _set_status(self, message: str) -> None:
        self.status_label.config(text=f"Status: {message}")
        self.update_idletasks()

    def select_dataset(self) -> None:
        raw_path = filedialog.askdirectory(title="Select dataset folder")
        if not raw_path:
            return
        self.dataset_root = detect_dataset_root(Path(raw_path))
        self.index_path = self.dataset_root / "_cbir_index.joblib"
        self.index = None
        self.dataset_label.config(text=f"Dataset: {self.dataset_root}")

        scan = scan_dataset(self.dataset_root)
        self.log(f"Selected dataset root: {scan.root}")
        self.log(f"Class counts: {scan.counts}")
        self.log(f"Missing classes: {list(scan.missing_classes)}")

    def build_or_load_index(self) -> None:
        if self.dataset_root is None or self.index_path is None:
            messagebox.showwarning("Dataset", "Select dataset root first.")
            return

        try:
            if self.index_path.exists():
                self._set_status("Loading index...")
                self.index = CBIRIndex.load(self.index_path)
                self._set_status(f"Index loaded. N={self.index.image_count}")
                return

            self._set_status("Building index...")
            self.index = build_index(
                self.dataset_root,
                self.index_path,
                self.feature_config,
                status_callback=self._build_status,
            )
            self._set_status(f"Index built. N={self.index.image_count}")
        except Exception as exc:
            self._set_status("Error")
            self.log(str(exc))
            self.log(traceback.format_exc())
            messagebox.showerror("Index Error", str(exc))

    def _build_status(self, message: str) -> None:
        self._set_status(message)
        self.log(message)

    def select_query(self) -> None:
        raw_path = filedialog.askopenfilename(
            title="Select query image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not raw_path:
            return
        self.query_path = Path(raw_path)
        self.query_label.config(text=f"Query: {self.query_path.name}")
        self.sift_query = None
        self.harris_query = None
        self._show_query_preview(self.query_path)

    def _show_query_preview(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((260, 260))
            self.query_image = ImageTk.PhotoImage(image)
            self.query_canvas.config(image=self.query_image)
        except Exception as exc:
            self.query_canvas.config(image="")
            self.log(f"Query preview failed: {exc}")

    def _current_search_config(self) -> SearchConfig:
        return SearchConfig(
            top_k=10,
            sift_weight=float(self.sift_weight.get()),
            harris_weight=float(self.harris_weight.get()),
            feedback_lambda=0.60,
        )

    def search(self) -> None:
        if self.index is None:
            messagebox.showwarning("Index", "Build or load the index first.")
            return
        if self.query_path is None:
            messagebox.showwarning("Query", "Select a query image first.")
            return

        try:
            if self.sift_query is None or self.harris_query is None:
                self.sift_query, self.harris_query = compute_query_vectors(
                    self.index,
                    self.query_path,
                )

            self.result_items = search_top_k(
                self.index,
                self.sift_query,
                self.harris_query,
                dataset_root=self.dataset_root,
                search_config=self._current_search_config(),
            )
            self._render_results(self.result_items)
            self._set_status("Done")
        except Exception as exc:
            self._set_status("Error")
            self.log(str(exc))
            self.log(traceback.format_exc())
            messagebox.showerror("Search Error", str(exc))

    def _clear_results(self) -> None:
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.result_checks = []
        self.result_images = []

    def _render_results(self, hits: list[SearchHit]) -> None:
        self._clear_results()
        columns = 5

        for rank, hit in enumerate(hits, start=1):
            row = (rank - 1) // columns
            column = (rank - 1) % columns
            frame = tk.Frame(self.results_frame, bd=1, relief=tk.GROOVE, padx=5, pady=5)
            frame.grid(row=row, column=column, padx=6, pady=6, sticky="n")

            image_tk: ImageTk.PhotoImage | None = None
            try:
                image = Image.open(hit.path).convert("RGB")
                image.thumbnail((170, 130))
                image_tk = ImageTk.PhotoImage(image)
                tk.Label(frame, image=image_tk).pack()
            except Exception as exc:
                tk.Label(frame, text="Image load failed", fg="red").pack()
                self.log(f"Could not preview {hit.path}: {exc}")
            self.result_images.append(image_tk)

            tk.Label(frame, text=f"#{rank} score={hit.score:.3f}", anchor="w").pack(fill=tk.X)
            tk.Label(frame, text=hit.path.name, wraplength=160, justify="left").pack(fill=tk.X)
            variable = tk.IntVar(value=0)
            if rank <= 5:
                tk.Checkbutton(frame, text="Relevant", variable=variable).pack(anchor="w", pady=2)
            self.result_checks.append(variable)

        for column in range(columns):
            self.results_frame.grid_columnconfigure(column, weight=1)

    def refine(self) -> None:
        if self.index is None or not self.result_items:
            messagebox.showwarning("Refine", "Run a search first.")
            return
        if self.sift_query is None or self.harris_query is None:
            return

        relevant_indices = [
            hit.index
            for rank, hit in enumerate(self.result_items[:5])
            if self.result_checks[rank].get() == 1
        ]
        if not relevant_indices:
            messagebox.showinfo("Refine", "Select at least one relevant image among the top 5.")
            return

        config = self._current_search_config()
        self.sift_query = rocchio_refine(
            self.sift_query,
            self.index.sift_histograms[relevant_indices],
            config.feedback_lambda,
        )
        self.harris_query = rocchio_refine(
            self.harris_query,
            self.index.harris_histograms[relevant_indices],
            config.feedback_lambda,
        )
        self.result_items = search_top_k(
            self.index,
            self.sift_query,
            self.harris_query,
            dataset_root=self.dataset_root,
            search_config=config,
        )
        self._render_results(self.result_items)
        self._set_status("Refined results shown")


def run_gui() -> None:
    app = VisualSearchApp()
    app.mainloop()
