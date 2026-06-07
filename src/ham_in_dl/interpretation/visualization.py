"""Visualization helpers for model interpretation."""

import numpy as np
from matplotlib import cm
from PIL import Image


def overlay_heatmap(
    image: Image.Image,
    heatmap,
    *,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a normalised heatmap on a PIL image."""
    image = image.convert("RGB")
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)

    if heatmap_array.shape[:2] != (image.height, image.width):
        heatmap_image = Image.fromarray((heatmap_array * 255).astype(np.uint8))
        heatmap_image = heatmap_image.resize(image.size, resample=Image.BILINEAR)
        heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    cmap = cm.get_cmap(colormap)
    colored_heatmap = (cmap(heatmap_array)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(image, dtype=np.float32)
    overlay = (1.0 - alpha) * base + alpha * colored_heatmap.astype(np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
