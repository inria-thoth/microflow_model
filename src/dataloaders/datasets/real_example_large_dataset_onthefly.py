import os
import math
import time

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import rasterio as rio
from rasterio.windows import Window


# ---------------------------------------------------------------------------
# Per-worker-process rasterio handle cache.
# Keys are (os.getpid(), filename).  Each DataLoader worker is its own OS
# process, so workers never share handles — no locking needed.
# ---------------------------------------------------------------------------
_worker_handles: dict = {}


def _get_rasterio_handle(filename: str) -> rio.DatasetReader:
    """Return a per-worker-process cached rasterio handle.

    Opening a GeoTIFF is expensive; caching one handle per worker avoids
    repeated open/seek/close overhead when reading many small windows.
    """
    key = (os.getpid(), filename)
    if key not in _worker_handles:
        _worker_handles[key] = rio.open(filename)
    return _worker_handles[key]


class RealExampleLargeDatasetOnTheFly(Dataset):
    """Sliding-window dataset for large satellite image pairs.

    Automatically chooses the fastest strategy for the machine it runs on:

    **In-memory (fast path)**
        Both images are loaded into RAM at construction time and all patches
        extracted upfront.  ``__getitem__`` is a pure numpy slice — zero I/O,
        maximum throughput.  Used whenever the images fit in available memory.

    **On-the-fly with cached handles (large-image fallback)**
        Triggered by a ``MemoryError`` during the initial load attempt.
        Each DataLoader worker process opens the GeoTIFF *once* and keeps
        the handle alive.  This avoids the repeated ``rio.open / read / close``
        cycle that made the old implementation slow, while still supporting
        images that are too large to fit in RAM.
    """

    def __init__(
            self,
            root_dir: str,
            transform=None,
            pre_template: str = "pre.tif",
            post_template: str = "post.tif",
            top: int = 0,
            left: int = 0,
            window_size: int = 1024,
            window_overlap: int = 64,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_template = pre_template
        self.post_template = post_template
        self.top = top
        self.left = left
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.image_pair_name = None

        self.pre_filename, self.post_filename = self._get_filenames(root_dir)

        # Read lightweight metadata (no pixel data yet)
        with rio.open(self.pre_filename) as src:
            self.width = src.width
            self.height = src.height
            self.crs_info = src.crs
            self.transform_info = src.transform

        print(f"[RealExampleLargeDatasetOnTheFly] Image size: {self.width} × {self.height} px")

        # ------------------------------------------------------------------
        # Choose loading strategy: try RAM first, fall back to on-the-fly
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            self._load_in_memory()
            self.in_memory = True
            print(
                f"[RealExampleLargeDatasetOnTheFly] In-memory mode — "
                f"{len(self._patches)} patches loaded in {time.time() - t0:.1f}s"
            )
        except MemoryError:
            self.in_memory = False
            self._build_patch_index()
            print(
                f"[RealExampleLargeDatasetOnTheFly] On-the-fly mode (image too large for RAM) — "
                f"{len(self._x_positions)} patches indexed in {time.time() - t0:.1f}s"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_filenames(self, root_dir: str):
        pre_filename = post_filename = None
        for filename in os.listdir(root_dir):
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(root_dir, filename)
                self.image_pair_name = filename.replace(self.pre_template, "")
            if filename.endswith(self.post_template):
                post_filename = os.path.join(root_dir, filename)
        return pre_filename, post_filename

    @staticmethod
    def _read_full_band(filename: str) -> np.ndarray:
        """Load an entire GeoTIFF band as float32, replacing NaN/0 with 0."""
        with rio.open(filename) as src:
            arr = src.read(1).astype(np.float32)
        arr[np.isnan(arr)] = 0.0
        return arr

    def _patch_positions(self):
        """Generate (x, y) top-left corner positions for all sliding windows."""
        h, w = self.height, self.width
        stride = self.window_overlap if self.window_overlap > 0 else self.window_size
        ws = self.window_size
        positions = []
        for y in range(0, h - ws + stride, stride):
            for x in range(0, w - ws + stride, stride):
                positions.append((max(0, min(x, w - ws)), max(0, min(y, h - ws))))
        return positions

    # ------------------------------------------------------------------
    # In-memory path
    # ------------------------------------------------------------------

    def _load_in_memory(self):
        """Load both images into RAM and extract all patches upfront."""
        pre = self._read_full_band(self.pre_filename)
        post = self._read_full_band(self.post_filename)
        # Build nan mask from the raw post image before zeroing
        nan_mask_full = np.isnan(post) | (post == 0)
        stack = np.stack([pre, post], axis=0)  # (2, H, W)

        patches, nan_masks, x_positions, y_positions = [], [], [], []
        for x, y in self._patch_positions():
            patch = stack[:, y: y + self.window_size, x: x + self.window_size]
            mask = nan_mask_full[y: y + self.window_size, x: x + self.window_size]
            patches.append(patch)
            nan_masks.append(mask)
            x_positions.append(x)
            y_positions.append(y)

        # This is the line that raises MemoryError for very large images
        self._patches = np.array(patches, dtype=np.float32)   # (N, 2, ws, ws)
        self._nan_masks = np.array(nan_masks, dtype=bool)     # (N, ws, ws)
        self._x_positions = x_positions
        self._y_positions = y_positions

    # ------------------------------------------------------------------
    # On-the-fly path (large-image fallback)
    # ------------------------------------------------------------------

    def _build_patch_index(self):
        """Build (x, y) index without loading pixel data."""
        positions = self._patch_positions()
        self._x_positions = [p[0] for p in positions]
        self._y_positions = [p[1] for p in positions]

    def _read_patch_otf(self, x: int, y: int):
        """Read a patch using per-worker-process cached rasterio handles."""
        window = Window(col_off=x, row_off=y, width=self.window_size, height=self.window_size)

        pre_src = _get_rasterio_handle(self.pre_filename)
        post_src = _get_rasterio_handle(self.post_filename)

        pre = pre_src.read(1, window=window, boundless=True, fill_value=0.0).astype(np.float32)
        post = post_src.read(1, window=window, boundless=True).astype(np.float32)

        nan_mask = np.isnan(post)
        post[nan_mask] = 0.0
        nan_mask = nan_mask | (post == 0)

        return np.stack([pre, post], axis=0), nan_mask

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._x_positions)

    def get_image_info(self):
        """Return (height, width, window_overlap, window_size, image_pair_name)."""
        return self.height, self.width, self.window_overlap, self.window_size, self.image_pair_name

    def get_tiff_metadata(self):
        """Return (crs, transform) for saving georeferenced output."""
        return self.crs_info, self.transform_info

    def __getitem__(self, idx: int) -> dict:
        x = self._x_positions[idx]
        y = self._y_positions[idx]

        if self.in_memory:
            patch = self._patches[idx]       # (2, ws, ws) float32
            nan_mask = self._nan_masks[idx]  # (ws, ws) bool
        else:
            patch, nan_mask = self._read_patch_otf(x, y)

        sample = {
            'pre_post_image': torch.from_numpy(patch),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(idx)
        sample['x_position'] = x
        sample['y_position'] = y
        sample['nan_mask'] = torch.from_numpy(nan_mask)
        return sample