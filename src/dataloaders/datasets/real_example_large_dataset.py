import os
import math
import time

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import rasterio as rio
from rasterio.windows import Window

# Per-worker-process rasterio file handle cache.
# Keys are (os.getpid(), filename) so each worker gets its own handle.
_worker_handles: dict = {}


def _get_rasterio_handle(filename: str) -> rio.DatasetReader:
    """Return a per-worker-process cached rasterio handle.

    Opening a GeoTIFF is expensive; keeping one handle alive per worker
    process avoids the repeated open/seek/close overhead of the naive approach.
    Each DataLoader worker is a separate OS process, so there is no sharing
    concern: every worker gets its own independent handle.
    """
    key = (os.getpid(), filename)
    if key not in _worker_handles:
        _worker_handles[key] = rio.open(filename)
    return _worker_handles[key]


class RealExampleLargeDataset(Dataset):
    """Sliding-window dataset for large satellite image pairs.

    Loading strategy (chosen automatically at construction time):

    **In-memory (fast, default)**
        The entire pre- and post-image are loaded into a single numpy array.
        ``__getitem__`` is a pure RAM slice: zero I/O, maximum throughput.
        Works well when ``2 × H × W × 4 bytes`` fits in available RAM.

    **On-the-fly with cached handles (large-image fallback)**
        Triggered by a ``MemoryError`` during the initial load.
        Each DataLoader worker process opens the GeoTIFF *once* and keeps
        the handle alive for the lifetime of that worker.  This avoids the
        repeated ``rio.open`` / ``read`` / ``close`` cycle that made the
        previous clean-repo implementation slow.
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

        # Read lightweight metadata without loading pixel data
        with rio.open(self.pre_filename) as src:
            self.width = src.width
            self.height = src.height
            self.crs_info = src.crs
            self.transform_info = src.transform

        print(f"Image size: {self.width} × {self.height} px")

        # ------------------------------------------------------------------
        # Choose loading strategy
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            self._load_in_memory()
            self.in_memory = True
            print(f"[RealExampleLargeDataset] In-memory mode — loaded in {time.time() - t0:.1f}s")
        except MemoryError:
            self.in_memory = False
            self._init_patch_index()
            print(
                f"[RealExampleLargeDataset] On-the-fly mode (image too large for RAM) — "
                f"index built in {time.time() - t0:.1f}s"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_filenames(self, root_dir: str):
        pre_filename = post_filename = None
        for filename in os.listdir(root_dir):
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(root_dir, filename)
                self.image_pair_name = filename[: -len(self.pre_template)]
            if filename.endswith(self.post_template):
                post_filename = os.path.join(root_dir, filename)
        return pre_filename, post_filename

    @staticmethod
    def _read_band(filename: str) -> np.ndarray:
        """Load a full GeoTIFF band as float32, replacing NaN with 0."""
        with rio.open(filename) as src:
            arr = src.read(1).astype(np.float32)
        arr[np.isnan(arr)] = 0.0
        return arr

    # ------------------------------------------------------------------
    # In-memory path
    # ------------------------------------------------------------------

    def _load_in_memory(self):
        """Load both images into RAM, extract and store all patches."""
        pre = self._read_band(self.pre_filename)
        post = self._read_band(self.post_filename)
        stack = np.stack([pre, post], axis=0)  # (2, H, W)

        patches, x_positions, y_positions = [], [], []
        h, w = stack.shape[1], stack.shape[2]
        stride = self.window_overlap if self.window_overlap > 0 else self.window_size
        ws = self.window_size

        for y in range(0, h - ws + stride, stride):
            for x in range(0, w - ws + stride, stride):
                y = max(0, min(y, h - ws))
                x = max(0, min(x, w - ws))
                patch = stack[:, y: y + ws, x: x + ws]
                if np.sum(np.abs(patch)) == 0:
                    continue
                patches.append(patch)
                x_positions.append(x)
                y_positions.append(y)

        # Stack once; this is the line that raises MemoryError for huge images
        self._patches = np.array(patches, dtype=np.float32)
        self._x_positions = x_positions
        self._y_positions = y_positions

    # ------------------------------------------------------------------
    # On-the-fly path (large images)
    # ------------------------------------------------------------------

    def _init_patch_index(self):
        """Build patch (x, y) index without loading pixel data."""
        h, w = self.height, self.width
        stride = self.window_overlap if self.window_overlap > 0 else self.window_size
        ws = self.window_size

        x_positions, y_positions = [], []
        for y in range(0, h - ws + stride, stride):
            for x in range(0, w - ws + stride, stride):
                y_c = max(0, min(y, h - ws))
                x_c = max(0, min(x, w - ws))
                x_positions.append(x_c)
                y_positions.append(y_c)

        self._x_positions = x_positions
        self._y_positions = y_positions

    def _read_patch_otf(self, x: int, y: int) -> np.ndarray:
        """Read a single patch using per-worker cached rasterio handles."""
        window = Window(col_off=x, row_off=y, width=self.window_size, height=self.window_size)

        pre_src = _get_rasterio_handle(self.pre_filename)
        post_src = _get_rasterio_handle(self.post_filename)

        pre = pre_src.read(1, window=window, boundless=True, fill_value=0.0).astype(np.float32)
        post = post_src.read(1, window=window, boundless=True, fill_value=0.0).astype(np.float32)
        post[np.isnan(post)] = 0.0

        return np.stack([pre, post], axis=0)  # (2, ws, ws)

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
            patch = self._patches[idx]  # (2, ws, ws), already float32
            nan_mask = np.zeros(patch.shape[1:], dtype=bool)
        else:
            patch = self._read_patch_otf(x, y)
            nan_mask = (patch[1] == 0)  # approximate: zeros treated as no-data

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
