
import os
import glob
import pickle
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# ====== Assignment Classes ======
REQUIRED_CLASSES = [
    "aeroplanes",
    "benches_and_chairs",
    "buildings",
    "cars",
    "doors",
    "signs",
    "windows",
]

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


DEFAULT_PARAMS = dict(
    max_image_side=512,
    sift_k=64,
    harris_k=64,
    sift_max_desc_per_img=250,
    harris_max_desc_per_img=300,
    harris_max_points=300,
    patch_size=16,
    sample_cap_sift=60000,
    sample_cap_harris=60000,
    seed=123,
)


# =========================
# Utilities
# =========================

def detect_dataset_root(selected_path: str) -> str:
    """
    Tries to find the folder that directly contains the class folders.
    Examples:
    - If user selects .../msrcorid and inside it there is Single/, use Single/
    - If user selects .../ and inside it there is msrcorid/, use that (and maybe Single/)
    - Otherwise use selected_path itself
    """
    if not selected_path:
        return selected_path

    # If "Single" exists inside selected path
    single = os.path.join(selected_path, "Single")
    if os.path.isdir(single):
        return single

    # If a single child folder exists like "msrcorid"
    child_msrc = os.path.join(selected_path, "msrcorid")
    if os.path.isdir(child_msrc):
        # if inside that "Single"
        child_single = os.path.join(child_msrc, "Single")
        if os.path.isdir(child_single):
            return child_single
        return child_msrc

    return selected_path


def _read_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _preprocess(img_bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    mx = max(h, w)
    if mx > max_side:
        scale = max_side / float(mx)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, gray


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _cosine_sim_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ q


def _list_images(dataset_root: str) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    Returns:
      paths: all image paths under REQUIRED_CLASSES (recursively)
      counts: per-class count
      missing_classes: class folders not found
    """
    paths: List[str] = []
    counts: Dict[str, int] = {}
    missing = []

    for cls in REQUIRED_CLASSES:
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            missing.append(cls)
            counts[cls] = 0
            continue

        cls_paths = []
        for ext in SUPPORTED_EXTS:
            cls_paths.extend(glob.glob(os.path.join(cls_dir, "**", f"*{ext}"), recursive=True))
            cls_paths.extend(glob.glob(os.path.join(cls_dir, "**", f"*{ext.upper()}"), recursive=True))

        cls_paths = sorted(set(cls_paths))
        counts[cls] = len(cls_paths)
        paths.extend(cls_paths)

    paths = sorted(set(paths))
    return paths, counts, missing


# =========================
# Feature extraction
# =========================

def extract_sift(gray: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 128), dtype=np.float32)
    return desc.astype(np.float32)


def extract_harris_patches(gray: np.ndarray, max_points: int, patch_size: int) -> np.ndarray:
    """
    Harris + patch descriptor (flattened, normalized patch)
    """
    gray_f = np.float32(gray) / 255.0

    blockSize = 2
    ksize = 3
    k = 0.04
    R = cv2.cornerHarris(gray_f, blockSize, ksize, k)
    R = cv2.dilate(R, None)

    if R.size == 0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    thr = 0.01 * float(R.max())
    ys, xs = np.where(R > thr)

    if len(xs) == 0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    scores = R[ys, xs]
    idx = np.argsort(scores)[::-1][:max_points]
    xs = xs[idx]
    ys = ys[idx]

    half = patch_size // 2
    H, W = gray.shape[:2]
    patches = []

    for x, y in zip(xs, ys):
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        patch = gray_f[y0:y1, x0:x1]
        if patch.shape != (patch_size, patch_size):
            continue

        vec = patch.reshape(-1).astype(np.float32)
        vec = vec - vec.mean()
        vec = vec / (vec.std() + 1e-6)
        patches.append(vec)

    if not patches:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    return np.stack(patches, axis=0).astype(np.float32)


# =========================
# BoVW Index
# =========================

def _sample_descriptors(desc_list: List[np.ndarray], cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_desc = [d for d in desc_list if d is not None and len(d) > 0]
    if not all_desc:
        return np.empty((0, 1), dtype=np.float32)
    X = np.vstack(all_desc)
    if X.shape[0] <= cap:
        return X
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx]


def _fit_kmeans(X: np.ndarray, k: int, seed: int) -> MiniBatchKMeans:
    if X.shape[0] < k:
        k = max(2, min(k, X.shape[0]))
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init="auto",
        max_iter=200,
        reassignment_ratio=0.01,
    )
    km.fit(X)
    return km


def _histogram_words(words: np.ndarray, k: int) -> np.ndarray:
    hist = np.bincount(words, minlength=k).astype(np.float32)
    hist = _l2_normalize(hist.reshape(1, -1)).reshape(-1)
    return hist


@dataclass
class CBIRIndex:
    dataset_root: str
    image_paths: List[str]
    sift_k: int
    harris_k: int
    sift_kmeans: MiniBatchKMeans
    harris_kmeans: MiniBatchKMeans
    sift_hist: np.ndarray
    harris_hist: np.ndarray

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CBIRIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_index(dataset_root: str, index_path: str, params: Dict, status_cb=None) -> CBIRIndex:
    def status(msg: str):
        if status_cb:
            status_cb(msg)

    img_paths, counts, missing = _list_images(dataset_root)
    status(f"[DATA] dataset_root={dataset_root}")
    status(f"[DATA] missing_class_folders={missing}")
    status(f"[DATA] counts_per_class={counts}")
    status(f"[DATA] total_images_found={len(img_paths)}")

    if len(img_paths) == 0:
        raise RuntimeError("No images found under required class folders. Check the dataset root you selected.")

    rng = np.random.default_rng(params["seed"])
    sift_descs: List[np.ndarray] = []
    harris_descs: List[np.ndarray] = []

    status("[FEAT] Extracting SIFT/Harris descriptors...")
    for i, p in enumerate(img_paths, 1):
        img = _read_image(p)
        if img is None:
            sift_descs.append(np.empty((0, 128), dtype=np.float32))
            harris_descs.append(np.empty((0, params["patch_size"]**2), dtype=np.float32))
            continue

        _, gray = _preprocess(img, params["max_image_side"])

        d1 = extract_sift(gray)
        if d1.shape[0] > params["sift_max_desc_per_img"]:
            idx = rng.choice(d1.shape[0], size=params["sift_max_desc_per_img"], replace=False)
            d1 = d1[idx]
        sift_descs.append(d1)

        d2 = extract_harris_patches(gray, params["harris_max_points"], params["patch_size"])
        if d2.shape[0] > params["harris_max_desc_per_img"]:
            idx = rng.choice(d2.shape[0], size=params["harris_max_desc_per_img"], replace=False)
            d2 = d2[idx]
        harris_descs.append(d2)

        if i % 50 == 0:
            status(f"[FEAT] processed {i}/{len(img_paths)}")

    # descriptor stats
    sift_total = int(sum(d.shape[0] for d in sift_descs))
    har_total = int(sum(d.shape[0] for d in harris_descs))
    status(f"[FEAT] total_sift_desc={sift_total}, total_harris_desc={har_total}")

    status("[KMEANS] Sampling & fitting SIFT vocabulary...")
    Xs = _sample_descriptors(sift_descs, cap=params["sample_cap_sift"], seed=params["seed"])
    if Xs.shape[0] < 10:
        raise RuntimeError("Not enough SIFT descriptors to build vocabulary (try another dataset root).")
    status(f"[KMEANS] SIFT samples={Xs.shape[0]} k={params['sift_k']}")
    sift_km = _fit_kmeans(Xs, k=params["sift_k"], seed=params["seed"])

    status("[KMEANS] Sampling & fitting Harris vocabulary...")
    Xh = _sample_descriptors(harris_descs, cap=params["sample_cap_harris"], seed=params["seed"] + 1)
    if Xh.shape[0] < 10:
        raise RuntimeError("Not enough Harris descriptors to build vocabulary (try another dataset root).")
    status(f"[KMEANS] Harris samples={Xh.shape[0]} k={params['harris_k']}")
    harris_km = _fit_kmeans(Xh, k=params["harris_k"], seed=params["seed"] + 1)

    status("[BOVW] Building histograms for all images...")
    N = len(img_paths)
    sift_k = sift_km.n_clusters
    harris_k = harris_km.n_clusters
    sift_hist = np.zeros((N, sift_k), dtype=np.float32)
    harris_hist = np.zeros((N, harris_k), dtype=np.float32)

    for i, (d1, d2) in enumerate(zip(sift_descs, harris_descs)):
        if d1.shape[0] > 0:
            words = sift_km.predict(d1)
            sift_hist[i] = _histogram_words(words, sift_k)
        if d2.shape[0] > 0:
            words = harris_km.predict(d2)
            harris_hist[i] = _histogram_words(words, harris_k)

        if (i + 1) % 100 == 0:
            status(f"[BOVW] histograms {i+1}/{N}")

    sift_hist = _l2_normalize(sift_hist)
    harris_hist = _l2_normalize(harris_hist)

    index = CBIRIndex(
        dataset_root=dataset_root,
        image_paths=img_paths,
        sift_k=sift_k,
        harris_k=harris_k,
        sift_kmeans=sift_km,
        harris_kmeans=harris_km,
        sift_hist=sift_hist,
        harris_hist=harris_hist,
    )

    index.save(index_path)
    status(f"[SAVE] Index saved to: {index_path}")
    return index


def compute_query_vectors(index: CBIRIndex, query_path: str, params: Dict, log_cb=None) -> Tuple[np.ndarray, np.ndarray]:
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    img = _read_image(query_path)
    if img is None:
        raise RuntimeError("Could not read query image (cv2.imread failed).")

    _, gray = _preprocess(img, params["max_image_side"])

    d1 = extract_sift(gray)
    log(f"[QUERY] SIFT desc shape: {d1.shape}")
    if d1.shape[0] > 0:
        words = index.sift_kmeans.predict(d1)
        q_sift = _histogram_words(words, index.sift_k)
    else:
        q_sift = np.zeros((index.sift_k,), dtype=np.float32)

    d2 = extract_harris_patches(gray, params["harris_max_points"], params["patch_size"])
    log(f"[QUERY] Harris desc shape: {d2.shape}")
    if d2.shape[0] > 0:
        words = index.harris_kmeans.predict(d2)
        q_harris = _histogram_words(words, index.harris_k)
    else:
        q_harris = np.zeros((index.harris_k,), dtype=np.float32)

    q_sift = _l2_normalize(q_sift.reshape(1, -1)).reshape(-1)
    q_harris = _l2_normalize(q_harris.reshape(1, -1)).reshape(-1)

    log(f"[QUERY] q_sift_norm={float(np.linalg.norm(q_sift)):.4f}, q_harris_norm={float(np.linalg.norm(q_harris)):.4f}")
    return q_sift, q_harris


def search_topk(index: CBIRIndex, q_sift: np.ndarray, q_harris: np.ndarray, topk: int, w_sift: float, w_harris: float) -> List[Tuple[int, float]]:
    sim_sift = _cosine_sim_matrix(q_sift, index.sift_hist)
    sim_harris = _cosine_sim_matrix(q_harris, index.harris_hist)
    scores = w_sift * sim_sift + w_harris * sim_harris
    order = np.argsort(scores)[::-1][:topk]
    return [(int(i), float(scores[i])) for i in order]


def rocchio_refine(q: np.ndarray, relevant_vecs: np.ndarray, lam: float = 0.6) -> np.ndarray:
    if relevant_vecs.size == 0:
        return q
    m = relevant_vecs.mean(axis=0)
    m = _l2_normalize(m.reshape(1, -1)).reshape(-1)
    q2 = (1.0 - lam) * q + lam * m
    q2 = _l2_normalize(q2.reshape(1, -1)).reshape(-1)
    return q2


# =========================
# GUI
# =========================

class CBIRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CBIR (SIFT + Harris) - Debug Version")
        self.geometry("1200x780")

        self.params = DEFAULT_PARAMS.copy()
        self.dataset_root: Optional[str] = None
        self.index: Optional[CBIRIndex] = None
        self.index_path: Optional[str] = None

        self.query_path: Optional[str] = None
        self.q_sift: Optional[np.ndarray] = None
        self.q_harris: Optional[np.ndarray] = None

        self.result_items: List[Tuple[int, float]] = []
        self.result_checks: List[tk.IntVar] = []
        self._result_imgtks: List[Optional[ImageTk.PhotoImage]] = []

        self._build_ui()
        self._startup_checks()

    def _startup_checks(self):
        if not hasattr(cv2, "SIFT_create"):
            messagebox.showerror(
                "OpenCV SIFT Missing",
                "cv2.SIFT_create not found.\nInstall opencv-contrib-python (or a recent opencv-python).\n"
                "If install fails on Windows, try:\n"
                "  pip install --user opencv-contrib-python"
            )

    def log(self, msg: str):
        print(msg)
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Button(top, text="Select Dataset Root", command=self.select_dataset).pack(side=tk.LEFT)
        self.dataset_lbl = tk.Label(top, text="Dataset: (not selected)")
        self.dataset_lbl.pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Build/Load Index", command=self.build_or_load_index).pack(side=tk.LEFT, padx=10)

        tk.Button(top, text="Select Query Image", command=self.select_query).pack(side=tk.LEFT, padx=10)
        self.query_lbl = tk.Label(top, text="Query: (not selected)")
        self.query_lbl.pack(side=tk.LEFT)

        wfrm = tk.Frame(self)
        wfrm.pack(side=tk.TOP, fill=tk.X, padx=10)

        tk.Label(wfrm, text="w_SIFT").pack(side=tk.LEFT)
        self.w_sift = tk.DoubleVar(value=0.6)
        tk.Scale(wfrm, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.w_sift, length=200).pack(side=tk.LEFT, padx=8)

        tk.Label(wfrm, text="w_Harris").pack(side=tk.LEFT)
        self.w_harris = tk.DoubleVar(value=0.4)
        tk.Scale(wfrm, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.w_harris, length=200).pack(side=tk.LEFT, padx=8)

        tk.Button(wfrm, text="Search Top 10", command=self.search).pack(side=tk.LEFT, padx=10)
        tk.Button(wfrm, text="Refine (Feedback)", command=self.refine).pack(side=tk.LEFT, padx=10)

        self.status_lbl = tk.Label(self, text="Status: Ready", anchor="w")
        self.status_lbl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

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

        # Log panel
        logfrm = tk.Frame(self)
        logfrm.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)
        tk.Label(logfrm, text="Log").pack(anchor="w")
        self.log_text = tk.Text(logfrm, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr = tk.Scrollbar(logfrm, command=self.log_text.yview)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scr.set)

        self._clear_results_widgets_only()

    def _set_status(self, msg: str):
        self.status_lbl.config(text=f"Status: {msg}")
        self.update_idletasks()

    def select_dataset(self):
        path = filedialog.askdirectory(title="Select dataset folder (contains class folders or Single)")
        if not path:
            return
        root = detect_dataset_root(path)
        self.dataset_root = root
        self.dataset_lbl.config(text=f"Dataset: {root}")
        self.index_path = os.path.join(root, "_cbir_index.pkl")
        self.index = None
        self.log(f"[UI] Selected dataset root: {root}")

        # quick scan
        paths, counts, missing = _list_images(root)
        self.log(f"[SCAN] missing_class_folders={missing}")
        self.log(f"[SCAN] counts_per_class={counts}")
        self.log(f"[SCAN] total_images_found={len(paths)}")

    def build_or_load_index(self):
        if not self.dataset_root:
            messagebox.showwarning("Dataset", "Select dataset root first.")
            return
        assert self.index_path is not None

        if os.path.exists(self.index_path):
            try:
                self._set_status("Loading existing index...")
                self.log(f"[LOAD] Loading index from {self.index_path}")
                self.index = CBIRIndex.load(self.index_path)
                self._set_status(f"Index loaded. N={len(self.index.image_paths)}")
                self.log(f"[LOAD] Index loaded. N={len(self.index.image_paths)}")
                messagebox.showinfo("Index", "Index loaded successfully.")
                return
            except Exception as e:
                self.log("[LOAD] Failed to load index. Will rebuild.")
                self.log(str(e))
                self.log(traceback.format_exc())

        try:
            self._set_status("Building index...")
            self.log("[BUILD] Building index...")
            self.index = build_index(self.dataset_root, self.index_path, self.params, status_cb=self._build_status_and_log)
            self._set_status(f"Index built. N={len(self.index.image_paths)}")
            messagebox.showinfo("Index", "Index built and saved successfully.")
        except Exception as e:
            self._set_status("Error")
            self.log("[ERROR] Index build failed:")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("Index Error", str(e))

    def _build_status_and_log(self, msg: str):
        self._set_status(msg)
        self.log(msg)

    def select_query(self):
        path = filedialog.askopenfilename(
            title="Select query image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        self.query_path = path
        self.query_lbl.config(text=f"Query: {os.path.basename(path)}")
        self._show_query_preview(path)
        self.q_sift = None
        self.q_harris = None
        self.log(f"[UI] Selected query: {path}")

    def _show_query_preview(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((260, 260))
            self._query_imgtk = ImageTk.PhotoImage(img)
            self.query_canvas.config(image=self._query_imgtk)
        except Exception as e:
            self.query_canvas.config(image="")
            self.log(f"[WARN] Query preview failed: {e}")

    # IMPORTANT: do NOT clear self.result_items here
    def _clear_results_widgets_only(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self.result_checks = []
        self._result_imgtks = []

    def search(self):
        if self.index is None:
            messagebox.showwarning("Index", "Build/Load index first.")
            return
        if not self.query_path:
            messagebox.showwarning("Query", "Select a query image first.")
            return

        try:
            # compute query vectors if needed
            if self.q_sift is None or self.q_harris is None:
                self._set_status("Computing query features...")
                self.log("[SEARCH] Computing query features...")
                self.q_sift, self.q_harris = compute_query_vectors(
                    self.index, self.query_path, self.params, log_cb=self.log
                )

            # normalize weights
            ws = float(self.w_sift.get())
            wh = float(self.w_harris.get())
            s = ws + wh
            if s <= 1e-9:
                ws, wh = 0.5, 0.5
            else:
                ws, wh = ws / s, wh / s
            self.log(f"[SEARCH] weights: w_sift={ws:.2f}, w_harris={wh:.2f}")

            # if query has no features
            if float(np.linalg.norm(self.q_sift)) < 1e-6 and float(np.linalg.norm(self.q_harris)) < 1e-6:
                messagebox.showwarning("Query Features", "No SIFT/Harris features extracted from query. Try another query image.")
                self.log("[WARN] Query features are near-zero.")
                return

            self._set_status("Searching...")
            self.log("[SEARCH] Running top-10 search...")
            self.result_items = search_topk(self.index, self.q_sift, self.q_harris, topk=10, w_sift=ws, w_harris=wh)
            self.log("[SEARCH] Top-10 indices/scores:")
            for r, (idx, sc) in enumerate(self.result_items, 1):
                self.log(f"  #{r:02d} idx={idx} score={sc:.4f} path={self.index.image_paths[idx]}")

            self._render_results(self.result_items)
            self._set_status("Done.")
        except Exception as e:
            self._set_status("Error")
            self.log("[ERROR] Search failed:")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("Search Error", str(e))

    def _render_results(self, items: List[Tuple[int, float]]):
        self._clear_results_widgets_only()

        cols = 5
        thumb = (170, 130)

        for rank, (idx, score) in enumerate(items):
            r = rank // cols
            c = rank % cols

            frm = tk.Frame(self.results_frame, bd=1, relief=tk.GROOVE, padx=5, pady=5)
            frm.grid(row=r, column=c, padx=6, pady=6, sticky="n")

            path = self.index.image_paths[idx]
            imgtk = None
            try:
                im = Image.open(path).convert("RGB")
                im.thumbnail(thumb)
                imgtk = ImageTk.PhotoImage(im)
                lbl_img = tk.Label(frm, image=imgtk)
                lbl_img.pack()
            except Exception as e:
                tk.Label(frm, text="(Image load failed)", fg="red").pack()
                self.log(f"[WARN] PIL open failed for: {path} | {e}")

            self._result_imgtks.append(imgtk)

            tk.Label(frm, text=f"#{rank+1}  score={score:.3f}", anchor="w").pack(fill=tk.X)
            tk.Label(frm, text=os.path.basename(path), wraplength=160, justify="left").pack(fill=tk.X)

            if rank < 5:
                var = tk.IntVar(value=0)
                chk = tk.Checkbutton(frm, text="Relevant", variable=var)
                chk.pack(anchor="w", pady=2)
                self.result_checks.append(var)
            else:
                self.result_checks.append(tk.IntVar(value=0))

        for c in range(cols):
            self.results_frame.grid_columnconfigure(c, weight=1)

    def refine(self):
        if self.index is None or not self.result_items:
            messagebox.showwarning("Refine", "Run a search first.")
            return

        selected = []
        for rank in range(min(5, len(self.result_items))):
            if self.result_checks[rank].get() == 1:
                selected.append(self.result_items[rank][0])

        if not selected:
            messagebox.showinfo("Refine", "Select at least one relevant image among top 5.")
            return

        try:
            self._set_status("Refining...")
            self.log(f"[RF] Selected relevant indices: {selected}")
            rel_sift = self.index.sift_hist[selected]
            rel_harris = self.index.harris_hist[selected]

            self.q_sift = rocchio_refine(self.q_sift, rel_sift, lam=0.6)
            self.q_harris = rocchio_refine(self.q_harris, rel_harris, lam=0.6)

            ws = float(self.w_sift.get())
            wh = float(self.w_harris.get())
            s = ws + wh
            if s <= 1e-9:
                ws, wh = 0.5, 0.5
            else:
                ws, wh = ws / s, wh / s

            self.result_items = search_topk(self.index, self.q_sift, self.q_harris, topk=10, w_sift=ws, w_harris=wh)
            self.log("[RF] Refined Top-10:")
            for r, (idx, sc) in enumerate(self.result_items, 1):
                self.log(f"  #{r:02d} idx={idx} score={sc:.4f} path={self.index.image_paths[idx]}")

            self._render_results(self.result_items)
            self._set_status("Refined results shown.")
        except Exception as e:
            self._set_status("Error")
            self.log("[ERROR] Refine failed:")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("Refine Error", str(e))


def main():
    app = CBIRApp()
    app.mainloop()


if __name__ == "__main__":
    main()