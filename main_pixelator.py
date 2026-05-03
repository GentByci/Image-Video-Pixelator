import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
import os
import threading
import concurrent.futures
import functools

# Pillow ≥ 9.1 exposes Dither/Resampling as enums; older versions had constants
# directly on the Image module. Support both.
try:
    _DITHER_FS = Image.Dither.FLOYDSTEINBERG
    _DITHER_NONE = Image.Dither.NONE
except AttributeError:
    _DITHER_FS = Image.FLOYDSTEINBERG
    _DITHER_NONE = Image.NONE

# How many threads to use for parallel frame processing. None = auto.
DEFAULT_WORKERS = min(8, os.cpu_count() or 4)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
GIF_EXTENSIONS = {'.gif'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
ANIMATED_EXTENSIONS = VIDEO_EXTENSIONS | GIF_EXTENSIONS

# ── Overlay charsets ─────────────────────────────────────────────────────────
# Ordered dark→bright. The overlay code maps each cell's grayscale value to an
# index in this string, so the first char is drawn on darkest cells.

OVERLAY_CHARSETS = {
    'Standard':  ' .:-=+*#%@',
    'Detailed':  ' .\'`^",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
    'Blocks':    ' \u2591\u2592\u2593\u2588',          # ░ ▒ ▓ █
    'Binary':    ' 01',
    'Halftone':  ' .:-+#@',
}

DEFAULT_OVERLAY = {
    'mode': 'none',     # 'none' | 'ascii' | 'dot'
    'charset': OVERLAY_CHARSETS['Standard'],
    'contrast': 0.5,
    'bloom': 0.0,
}

# ── Gradient Map presets ─────────────────────────────────────────────────────
# Each preset is a list of (R, G, B) gradient stops, evenly spaced from dark
# (luminance 0) to bright (luminance 255). 2-color = duotone, 3+ = polychrome.

GRADIENT_PRESETS = {
    'None':              None,
    'Sepia':             [(25, 15, 5), (130, 90, 60), (255, 225, 180)],
    'Matrix':            [(0, 0, 0), (0, 80, 30), (60, 255, 100)],
    'Cyberpunk':         [(15, 0, 60), (200, 0, 130), (0, 240, 255)],
    'Vaporwave':         [(40, 0, 80), (255, 100, 200), (0, 220, 255)],
    'Sunset':            [(40, 10, 70), (255, 90, 30), (255, 230, 130)],
    'Dusk':              [(10, 20, 60), (180, 70, 130), (255, 200, 110)],
    'Riso (Pink/Navy)':  [(20, 25, 80), (255, 90, 150)],
    'Risograph (Cyan)':  [(20, 35, 80), (50, 200, 230)],
    'B&W High Contrast': [(0, 0, 0), (255, 255, 255)],
    'Phosphor (Amber)':  [(15, 5, 0), (255, 165, 50)],
    'Ice':               [(10, 20, 50), (140, 200, 230), (240, 255, 255)],
}

# ── Palette Presets ──────────────────────────────────────────────────────────
# Each palette is a list of (R, G, B) tuples.

PALETTES = {
    "Original (auto-reduce)": None,

    "PICO-8 (16)": [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
    ],

    "GameBoy (4)": [
        (15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15),
    ],

    "Endesga-32 (32)": [
        (190, 74, 47), (215, 118, 67), (234, 212, 170), (228, 166, 114),
        (184, 111, 80), (115, 62, 57), (62, 39, 49), (162, 38, 51),
        (228, 59, 68), (247, 118, 34), (254, 174, 52), (254, 231, 97),
        (99, 199, 77), (62, 137, 72), (38, 92, 66), (25, 60, 62),
        (18, 78, 137), (0, 153, 219), (44, 232, 245), (192, 203, 220),
        (139, 155, 180), (90, 105, 136), (58, 68, 102), (38, 43, 68),
        (24, 20, 37), (104, 56, 108), (181, 80, 136), (246, 117, 122),
        (232, 183, 150), (194, 133, 105), (143, 86, 59), (55, 20, 0),
    ],

    "Endesga-16 (16)": [
        (227, 111, 71), (244, 185, 116), (251, 255, 203), (169, 219, 134),
        (82, 165, 114), (40, 89, 89), (30, 42, 65), (18, 18, 35),
        (98, 46, 76), (172, 53, 83), (227, 100, 86), (248, 198, 89),
        (106, 180, 211), (73, 120, 166), (55, 71, 122), (48, 44, 81),
    ],

    "DB32 – DawnBringer (32)": [
        (0, 0, 0), (34, 32, 52), (69, 40, 60), (102, 57, 49),
        (143, 86, 59), (223, 113, 38), (217, 160, 102), (238, 195, 154),
        (251, 242, 54), (153, 229, 80), (106, 190, 48), (55, 148, 110),
        (75, 105, 47), (82, 75, 36), (50, 60, 57), (63, 63, 116),
        (48, 96, 130), (91, 110, 225), (99, 155, 255), (95, 205, 228),
        (203, 219, 252), (255, 255, 255), (155, 173, 183), (132, 126, 135),
        (105, 106, 106), (89, 86, 82), (118, 66, 138), (172, 50, 50),
        (217, 87, 99), (215, 123, 186), (143, 151, 74), (138, 111, 48),
    ],

    "Import from file...": "import",
}


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def is_gif(path):
    return os.path.splitext(path)[1].lower() in GIF_EXTENSIONS


def is_animated(path):
    return os.path.splitext(path)[1].lower() in ANIMATED_EXTENSIONS


# ── Palette I/O ──────────────────────────────────────────────────────────────

def load_palette_from_file(path):
    """Load a palette from .hex, .gpl (GIMP), or .png swatch file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.hex':
        colors = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().lstrip('#')
                if len(line) == 6:
                    try:
                        r, g, b = int(line[0:2], 16), int(line[2:4], 16), int(line[4:6], 16)
                        colors.append((r, g, b))
                    except ValueError:
                        continue
        return colors if colors else None

    elif ext == '.gpl':
        colors = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('GIMP') or line.startswith('Name') or line.startswith('Columns'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                        colors.append((r, g, b))
                    except ValueError:
                        continue
        return colors if colors else None

    elif ext == '.png':
        img = cv2.imread(path)
        if img is None:
            return None
        # Read unique colors from a swatch image (typically 1-row or small grid)
        pixels = img.reshape(-1, 3)
        unique = np.unique(pixels, axis=0)
        return [(int(r), int(g), int(b)) for b, g, r in unique]  # BGR → RGB

    return None


# ── Color Quantization ───────────────────────────────────────────────────────

def _pil_dither_to_palette(image_bgr, palette_rgb):
    """Floyd-Steinberg dither against a fixed palette using PIL's C-implemented
    quantizer. Much faster than a hand-rolled numpy/python error-diffusion loop,
    especially for the small downsampled images we feed it.

    image_bgr: BGR uint8 numpy array (cv2 native order)
    palette_rgb: iterable of (R, G, B) tuples, up to 256 entries
    Returns BGR uint8 numpy array.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb, mode='RGB')

    # PIL's P-mode palette is always 256 entries × 3 bytes = 768 bytes total.
    # Pad short palettes with the last entry — padding with (0,0,0) would pull
    # error-diffused samples toward black for any palette without pure black.
    palette_rgb = list(palette_rgb)[:256]
    flat = []
    for r, g, b in palette_rgb:
        flat.extend([int(r), int(g), int(b)])
    pad = palette_rgb[-1] if palette_rgb else (0, 0, 0)
    while len(flat) < 768:
        flat.extend([int(pad[0]), int(pad[1]), int(pad[2])])

    pal_img = Image.new('P', (1, 1))
    pal_img.putpalette(flat)

    quantized = pil_img.quantize(palette=pal_img, dither=_DITHER_FS)
    rgb_out = np.asarray(quantized.convert('RGB'))
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


def _normalize_dither(d):
    """Translate the various dither flag forms into a canonical string.
    Accepts True/'fs'/'floyd-steinberg' for FS, 'bayer' for Bayer, anything else
    falls back to 'none'. Keeps backward compat with the old bool flag.
    """
    if d is True or d == 'fs' or d == 'floyd-steinberg':
        return 'fs'
    if d == 'bayer':
        return 'bayer'
    return 'none'


@functools.lru_cache(maxsize=4)
def _bayer_matrix(n):
    """Build n×n Bayer matrix (n must be a power of 2). Values are 0..n²-1.

    Recursive definition:
        B_1  = [[0]]
        B_2n = [[4·B_n + 0,  4·B_n + 2],
                [4·B_n + 3,  4·B_n + 1]]
    """
    if n == 1:
        return np.zeros((1, 1), dtype=np.int32)
    half = _bayer_matrix(n // 2)
    return np.block([
        [4 * half + 0, 4 * half + 2],
        [4 * half + 3, 4 * half + 1],
    ]).astype(np.int32)


@functools.lru_cache(maxsize=4)
def _bayer_threshold(n=8):
    """Bayer threshold map normalized to [-0.5, 0.5)."""
    m = _bayer_matrix(n)
    return ((m + 0.5) / (n * n) - 0.5).astype(np.float32)


def _bayer_tiled(h, w, n=8):
    """Tile the Bayer threshold to image size in one shot."""
    threshold = _bayer_threshold(n)
    th = (h + n - 1) // n
    tw = (w + n - 1) // n
    return np.tile(threshold, (th, tw))[:h, :w]


def _bayer_dither_to_palette(image_bgr, palette_rgb, n=8, strength=None):
    """Apply Bayer ordered dithering and map to a fixed palette.

    Unlike error-diffusion (FS), Bayer is purely positional — same input always
    gives same output, which is exactly the property we want for video so
    static regions don't shimmer between frames.

    `strength` controls how far each pixel can be nudged; defaults to roughly
    one channel-wise palette step, estimated from the palette size.
    """
    h, w = image_bgr.shape[:2]
    tiled = _bayer_tiled(h, w, n)
    if strength is None:
        # Heuristic: ~half the channel-wise spacing of an evenly-spread palette.
        strength = 128.0 / max(1.0, len(palette_rgb) ** (1.0 / 3.0))
    noisy = image_bgr.astype(np.float32) + tiled[..., None] * float(strength)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return map_to_palette(noisy, palette_rgb)


def _bayer_dither_fast(image_bgr, levels, n=8):
    """Bayer dither for the uniform-bucket fast quantizer."""
    h, w = image_bgr.shape[:2]
    tiled = _bayer_tiled(h, w, n)
    factor = 256 // levels
    noisy = image_bgr.astype(np.float32) + tiled[..., None] * float(factor)
    noisy = np.clip(noisy, 0, 255).astype(np.int32)
    return ((noisy // factor) * factor + factor // 2).astype(np.uint8)


# ── Gradient Map ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _build_gradient_lut(stops_tuple):
    """256×3 BGR LUT mapping luminance → gradient color.
    `stops_tuple` is a tuple of (R, G, B) so it can be cached as a key.
    """
    stops = np.asarray(stops_tuple, dtype=np.float32)  # (k, 3) RGB
    k = len(stops)
    if k < 2:
        return None
    lut = np.empty((256, 3), dtype=np.uint8)
    seg_count = k - 1
    for i in range(256):
        t = (i / 255.0) * seg_count
        seg = min(int(t), seg_count - 1)
        local_t = t - seg
        rgb = stops[seg] * (1.0 - local_t) + stops[seg + 1] * local_t
        lut[i, 0] = int(round(rgb[2]))  # B
        lut[i, 1] = int(round(rgb[1]))  # G
        lut[i, 2] = int(round(rgb[0]))  # R
    return lut


def apply_gradient_map(image_bgr, stops):
    """Replace image colours with a gradient indexed by luminance."""
    if not stops or len(stops) < 2:
        return image_bgr
    stops_tuple = tuple(tuple(int(c) for c in stop) for stop in stops)
    lut = _build_gradient_lut(stops_tuple)
    if lut is None:
        return image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return lut[gray]


# ── CRT post-processing ──────────────────────────────────────────────────────
# All the masks/maps depend only on (h, w, intensity), so they're cached and
# reused across video frames at no extra cost.

@functools.lru_cache(maxsize=8)
def _vignette_mask(h, w, strength_q):
    """Radial darkening mask (1.0 at center → ~1-strength at corners).
    strength_q is strength × 1000 so the cache key is hashable.
    """
    strength = strength_q / 1000.0
    yy, xx = np.indices((h, w)).astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    norm = np.sqrt(cx * cx + cy * cy)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(norm, 1.0)
    mask = np.clip(1.0 - dist * dist * strength, 0.0, 1.0)
    return mask.astype(np.float32)


@functools.lru_cache(maxsize=8)
def _barrel_maps(h, w, k_q):
    """Precompute (src_x, src_y) maps for cv2.remap. k > 0 = barrel."""
    k = k_q / 1000.0
    yy, xx = np.indices((h, w)).astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    r2 = nx * nx + ny * ny
    scale = 1.0 + k * r2
    src_x = (nx * scale * cx + cx).astype(np.float32)
    src_y = (ny * scale * cy + cy).astype(np.float32)
    return src_x, src_y


@functools.lru_cache(maxsize=8)
def _scanline_mask(h, w, strength_q):
    """(h, w, 1) mask darkening every other row by `strength`."""
    strength = strength_q / 1000.0
    mask = np.ones((h, w, 1), dtype=np.float32)
    mask[::2] = 1.0 - strength
    return mask


def apply_crt(image, amount):
    """Bundled CRT effect: barrel distortion → chromatic aberration →
    scanlines × vignette. `amount` is 0..1; 1.0 is strong but non-destructive.
    """
    if amount <= 0:
        return image
    amount = float(np.clip(amount, 0.0, 1.0))
    h, w = image.shape[:2]

    # 1. Barrel distortion (subtle bulge)
    k = 0.18 * amount
    src_x, src_y = _barrel_maps(h, w, int(round(k * 1000)))
    image = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 2. Chromatic aberration: shift R left, B right (BGR ⇒ ch 0 is B, ch 2 is R)
    shift = max(1, int(round(2.5 * amount)))
    if shift > 0 and w > shift:
        out = image.copy()
        out[:, shift:, 2] = image[:, :w - shift, 2]   # R ← from-left
        out[:, :w - shift, 0] = image[:, shift:, 0]   # B ← from-right
        image = out

    # 3. Scanlines × vignette in a single float multiply pass
    sl_strength = 0.45 * amount
    vg_strength = 0.65 * amount
    sl = _scanline_mask(h, w, int(round(sl_strength * 1000)))
    vg = _vignette_mask(h, w, int(round(vg_strength * 1000)))
    f = image.astype(np.float32) * sl * vg[..., None]
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def map_to_palette(image, palette_rgb, chunk_size=32768):
    """Map every pixel to the nearest color in a fixed palette (Euclidean RGB).
    Chunked to avoid building one huge (N x K x 3) intermediate that can blow up
    memory on large images / large palettes.
    """
    palette_bgr = np.asarray([(b, g, r) for r, g, b in palette_rgb], dtype=np.int32)
    flat = image.reshape(-1, 3).astype(np.int32)
    n = flat.shape[0]
    out = np.empty((n, 3), dtype=np.uint8)
    for i in range(0, n, chunk_size):
        chunk = flat[i:i + chunk_size]
        dists = np.sum((chunk[:, None, :] - palette_bgr[None, :, :]) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        out[i:i + chunk_size] = palette_bgr[idx].astype(np.uint8)
    return out.reshape(image.shape)


def quantize_fast(image, levels):
    """Fast color quantization via integer bucketing."""
    factor = 256 // levels
    return (image // factor * factor + factor // 2).astype(np.uint8)


def quantize_kmeans(image, k):
    """K-means color quantization — slower but higher quality."""
    Z = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers, labels


def apply_kmeans_centers(image, centers, chunk_size=32768):
    """Map each pixel to the nearest precomputed k-means center.

    NOTE: the previous version of this function ran cv2.kmeans again with random
    initialisation and then tried to apply the precomputed centers via the new
    labels — that's wrong (labels don't match the supplied centers) and slow.
    This is a straight nearest-neighbour mapping.
    """
    centers_int = np.asarray(centers, dtype=np.int32)
    flat = image.reshape(-1, 3).astype(np.int32)
    n = flat.shape[0]
    out = np.empty((n, 3), dtype=np.uint8)
    for i in range(0, n, chunk_size):
        chunk = flat[i:i + chunk_size]
        dists = np.sum((chunk[:, None, :] - centers_int[None, :, :]) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        out[i:i + chunk_size] = centers_int[idx].astype(np.uint8)
    return out.reshape(image.shape)


# ── Grid Overlay ─────────────────────────────────────────────────────────────

def draw_grid(image, pixel_size):
    """Draw a 1px grid at pixel block boundaries."""
    h, w = image.shape[:2]
    overlay = image.copy()
    # Use dark gray for the grid — visible on both light and dark areas
    color = (40, 40, 40)
    for x in range(0, w, pixel_size):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, pixel_size):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    # Blend for semi-transparency
    return cv2.addWeighted(overlay, 0.6, image, 0.4, 0)


# ── Overlay (ASCII / Dot Matrix) ─────────────────────────────────────────────

@functools.lru_cache(maxsize=128)
def _get_char_atlas(charset, cell_size):
    """Build an (n_chars, cell_size, cell_size) uint8 mask atlas for a charset.

    Cached so that video frames don't re-render the same characters each call.
    `charset` is a str so it's hashable; `cell_size` is an int.

    Each tile is a grayscale glyph mask (0 = background, 255 = stroke), centred
    in a cell-sized box with a monospace font sized to fit the cell. We try a
    chain of common fonts and fall back to PIL's default bitmap font.
    """
    target = max(6, int(cell_size * 0.95))
    font = None
    for name in ['consola.ttf', 'cour.ttf', 'lucon.ttf',
                 'DejaVuSansMono.ttf', 'LiberationMono-Regular.ttf',
                 'Menlo.ttc', 'Monaco.ttf', 'Courier.ttc']:
        try:
            font = ImageFont.truetype(name, target)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    atlas = np.zeros((len(charset), cell_size, cell_size), dtype=np.uint8)
    for i, ch in enumerate(charset):
        img = Image.new('L', (cell_size, cell_size), 0)
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x_off = (cell_size - w) // 2 - bbox[0]
            y_off = (cell_size - h) // 2 - bbox[1]
        except AttributeError:
            # Older Pillow without textbbox
            x_off = y_off = 0
        draw.text((x_off, y_off), ch, font=font, fill=255)
        atlas[i] = np.asarray(img)

    atlas.flags.writeable = False  # protect cached array from accidental mutation
    return atlas


@functools.lru_cache(maxsize=8)
def _get_dot_atlas(cell_size, n_levels=16):
    """Build (n_levels, cell_size, cell_size) atlas of soft circles at varying
    intensity. Used for the dot-matrix overlay; each cell picks a level based
    on its brightness, giving an LED-display feel.
    """
    yy, xx = np.mgrid[:cell_size, :cell_size].astype(np.float32)
    cy = (cell_size - 1) / 2.0
    cx = (cell_size - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = cell_size * 0.4
    # Soft falloff at the edge for a nicer-looking dot
    base = np.clip((radius - dist) * 1.5, 0.0, 1.0) * 255.0
    atlas = np.stack(
        [(base * i / max(1, n_levels - 1)).astype(np.uint8) for i in range(n_levels)]
    )
    atlas.flags.writeable = False
    return atlas


def apply_overlay(image, pixel_size, mode='none', charset=' .:-=+*#%@',
                  contrast=0.5, bloom=0.0):
    """Draw an ASCII or dot-matrix overlay on top of an already-pixelated image.

    Each pixel_size×pixel_size cell becomes a glyph (ASCII char or LED dot)
    drawn in a brightened version of the cell's colour, against a darkened
    version of that same colour.

    contrast: 0.0 keeps glyph and background equal (overlay invisible);
              1.0 pushes them to black/white (max separation).
    bloom:    0.0 = none; >0 adds a Gaussian glow tinted by the glyph colour,
              for a CRT-phosphor feel.

    Implementation note: we never call ImageDraw per-cell at runtime. The
    glyph atlas is built once (and cached), then `atlas[tile_idx]` produces the
    full mask in a single vectorised step.
    """
    if mode == 'none' or pixel_size < 2:
        return image

    h, w = image.shape[:2]
    sw = w // pixel_size
    sh = h // pixel_size
    if sw < 1 or sh < 1:
        return image

    full_h, full_w = sh * pixel_size, sw * pixel_size

    # One colour per cell from the (already pixelated) image
    cell_bgr = cv2.resize(image[:full_h, :full_w], (sw, sh),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    if mode == 'ascii' and len(charset) > 0:
        atlas = _get_char_atlas(charset, pixel_size)
    elif mode == 'dot':
        atlas = _get_dot_atlas(pixel_size, n_levels=16)
    else:
        return image

    n = atlas.shape[0]
    tile_idx = (gray.astype(np.int32) * n // 256).clip(0, n - 1)

    # Build the full overlay mask in one shot:
    #   tiles[cy, cx] = atlas[tile_idx[cy, cx]] is a (ps, ps) glyph
    #   transpose+reshape interleaves rows so the result is (sh*ps, sw*ps)
    tiles = atlas[tile_idx]                                       # (sh, sw, ps, ps)
    mask_full = tiles.transpose(0, 2, 1, 3).reshape(full_h, full_w)

    # Per-cell fg / bg colours, then upsample with NEAREST (each cell is solid)
    contrast_c = float(np.clip(contrast, 0.0, 1.0))
    cells_f = cell_bgr.astype(np.float32)
    bg_small = cells_f * (1.0 - contrast_c)
    fg_small = cells_f + (255.0 - cells_f) * contrast_c

    bg_full = cv2.resize(bg_small, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    fg_full = cv2.resize(fg_small, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

    # Composite: bg where mask is 0, fg where mask is 255, blend in between
    m = (mask_full.astype(np.float32) / 255.0)[..., None]
    composed = bg_full * (1.0 - m) + fg_full * m

    # Bloom: blur the glyph mask, tint by fg colour, additive blend
    if bloom > 0:
        bloom_c = float(np.clip(bloom, 0.0, 1.0))
        radius = max(3, int(pixel_size * bloom_c * 1.5))
        if radius % 2 == 0:
            radius += 1
        blurred = cv2.GaussianBlur(mask_full, (radius, radius), 0)
        blurred_n = (blurred.astype(np.float32) / 255.0)[..., None]
        glow = fg_full * blurred_n * bloom_c
        composed = composed + glow

    composed = np.clip(composed, 0.0, 255.0).astype(np.uint8)

    # Edges: if pixel_size doesn't divide image evenly, keep original pixels
    # in the leftover strip rather than leaving a black border.
    if (full_h, full_w) != (h, w):
        out = image.copy()
        out[:full_h, :full_w] = composed
        return out
    return composed


# ── GIF I/O ──────────────────────────────────────────────────────────────────

def read_gif_frames(path):
    """Read all frames from a GIF as BGR numpy arrays.
    Returns (frames, durations_ms). PIL handles disposal/transparency
    composition when we convert each seek'd frame to RGB.
    """
    img = Image.open(path)
    frames = []
    durations = []
    try:
        n_frames = img.n_frames
    except AttributeError:
        n_frames = 1

    for i in range(n_frames):
        img.seek(i)
        rgb = img.convert('RGB')
        np_frame = np.asarray(rgb)
        bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        frames.append(bgr)
        durations.append(int(img.info.get('duration', 100)))

    img.close()
    return frames, durations


def write_gif(frames_bgr, output_path, durations=100, loop=0):
    """Write BGR frames as an animated GIF.

    `durations` may be an int (same delay for all frames, ms) or a list of ints
    (per-frame delays in ms). `optimize=True` lets PIL build a shared palette,
    which is great for pixel art where the colour count is already small.
    """
    if not frames_bgr:
        return False
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    return True


# ── Frame / Video / GIF Processing ───────────────────────────────────────────

def pixelate_frame(frame, pixel_size, color_reduction=None, method="fast",
                   palette_rgb=None, kmeans_centers=None, show_grid=False,
                   dither='none', overlay=None,
                   gradient_map=None, crt_amount=0.0):
    """Pixelate one frame and apply the optional effect stack.

    Pipeline order: downsample → quantize (with optional dither) → gradient
    map → upsample → grid → overlay → CRT. Each step is skippable.

    Optimisation: colour mapping happens on the SMALL (downsampled) image, not
    after upscaling. Mathematically identical because the upscaled image has
    identical pixels within each block, but skips pixel_size² of redundant
    work for the lookup.

    `dither` is one of 'none', 'fs' (Floyd-Steinberg), 'bayer'. The legacy
    bool form is also accepted: True ↔ 'fs', False ↔ 'none'. FS gives organic
    error-diffused noise; Bayer gives a fixed threshold matrix that's
    temporally stable across frames (no shimmer on static regions in video).

    `overlay` is an optional dict — see `apply_overlay`.
    `gradient_map` is an optional list of (R,G,B) stops — see `apply_gradient_map`.
    `crt_amount` (0..1) bundles barrel distortion, chromatic aberration,
    scanlines, and vignette into one knob — see `apply_crt`.
    """
    height, width = frame.shape[:2]
    sw = max(1, width // pixel_size)
    sh = max(1, height // pixel_size)
    # INTER_AREA gives slightly cleaner downsamples than INTER_LINEAR when
    # shrinking by a large factor (which is exactly what happens here).
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

    dither = _normalize_dither(dither)

    # Colour mapping (with optional dither) at the SMALL scale
    if palette_rgb:
        if dither == 'fs':
            small = _pil_dither_to_palette(small, palette_rgb)
        elif dither == 'bayer':
            small = _bayer_dither_to_palette(small, palette_rgb)
        else:
            small = map_to_palette(small, palette_rgb)
    elif color_reduction and color_reduction > 0:
        if method == "kmeans":
            # Compute centers if not supplied; we then either use the labels
            # directly (no dither) or dither against the centers as a palette.
            if kmeans_centers is None:
                centers, labels = quantize_kmeans(small, color_reduction)
                kmeans_centers = np.uint8(centers)
                if dither == 'none':
                    small = kmeans_centers[labels.flatten()].reshape(small.shape)
                    kmeans_centers = None       # marker: already applied
            if kmeans_centers is not None:
                rgb_palette = [(int(r), int(g), int(b)) for b, g, r in kmeans_centers]
                if dither == 'fs':
                    small = _pil_dither_to_palette(small, rgb_palette)
                elif dither == 'bayer':
                    small = _bayer_dither_to_palette(small, rgb_palette)
                else:
                    small = apply_kmeans_centers(small, kmeans_centers)
        else:  # method == "fast"
            if dither == 'bayer':
                small = _bayer_dither_fast(small, color_reduction)
            elif dither == 'fs' and color_reduction ** 3 <= 256:
                # Build implicit fast-quantize lattice and FS-dither against it
                factor = 256 // color_reduction
                values = [factor * i + factor // 2 for i in range(color_reduction)]
                rgb_palette = [(r, g, b) for r in values for g in values for b in values]
                small = _pil_dither_to_palette(small, rgb_palette)
            else:
                # color_reduction**3 > 256 means more lattice colours than PIL's
                # 256-entry palette mode allows, so we silently fall back here.
                small = quantize_fast(small, color_reduction)

    # Gradient map (luminance → user-defined gradient). Done at small scale
    # because each cell is solid, so the per-pixel LUT cost is minimised.
    if gradient_map:
        small = apply_gradient_map(small, gradient_map)

    # Upscale with NEAREST so pixel boundaries stay crisp
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    if show_grid:
        pixelated = draw_grid(pixelated, pixel_size)

    if overlay and overlay.get('mode', 'none') != 'none':
        pixelated = apply_overlay(
            pixelated, pixel_size,
            mode=overlay.get('mode', 'none'),
            charset=overlay.get('charset', ' .:-=+*#%@'),
            contrast=overlay.get('contrast', 0.5),
            bloom=overlay.get('bloom', 0.0),
        )

    # CRT goes last so its barrel distortion warps everything else, and its
    # scanlines/vignette darken the final composed image.
    if crt_amount and crt_amount > 0:
        pixelated = apply_crt(pixelated, crt_amount)

    return pixelated


def pixelate_video(input_path, output_path, pixel_size, color_reduction, method,
                   palette_rgb=None, show_grid=False, dither='none',
                   export_frames=False, frame_skip=1,
                   progress_callback=None, cancel_flag=None,
                   n_workers=None, overlay=None,
                   gradient_map=None, crt_amount=0.0):
    """Process a video file. Frames are pixelated in parallel batches via a
    thread pool, then written sequentially through cv2.VideoWriter (which is not
    thread-safe). Worker threads release the GIL during numpy/cv2 calls, so
    real wall-clock parallelism is achieved despite the GIL.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up frame export folder
    frames_dir = None
    if export_frames:
        base = os.path.splitext(output_path)[0]
        frames_dir = base + "_frames"
        os.makedirs(frames_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    # For k-means video without a fixed palette: compute centers from first
    # frame's downsampled version. (Computing on the small image instead of the
    # full upscaled one is much faster and gives the same centers.)
    kmeans_centers = None
    first_frame_consumed = False
    if not palette_rgb and color_reduction and method == "kmeans":
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return False
        sw = max(1, width // pixel_size)
        sh = max(1, height // pixel_size)
        small = cv2.resize(first_frame, (sw, sh), interpolation=cv2.INTER_AREA)
        centers, _ = quantize_kmeans(small, color_reduction)
        kmeans_centers = np.uint8(centers)
        result = pixelate_frame(first_frame, pixel_size, color_reduction, method,
                                palette_rgb, kmeans_centers, show_grid, dither,
                                overlay, gradient_map, crt_amount)
        out.write(result)
        if frames_dir:
            cv2.imwrite(os.path.join(frames_dir, "frame_0001.png"), result)
        if progress_callback:
            progress_callback(1, total_frames)
        first_frame_consumed = True

    frame_num = 1 if first_frame_consumed else 0
    exported = 1 if first_frame_consumed else 0

    def _do_frame(f):
        return pixelate_frame(f, pixel_size, color_reduction, method,
                              palette_rgb, kmeans_centers, show_grid, dither,
                              overlay, gradient_map, crt_amount)

    # Process the rest of the frames in parallel batches. Batch size is bounded
    # so memory stays modest even for long 1080p+ videos.
    batch_size = max(1, n_workers * 4)
    cancelled = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        while True:
            if cancel_flag and cancel_flag.is_set():
                cancelled = True
                break

            # Read a batch
            batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append(frame)
            if not batch:
                break

            # Process batch in parallel; executor.map preserves input order
            for result in executor.map(_do_frame, batch):
                if cancel_flag and cancel_flag.is_set():
                    cancelled = True
                    break

                frame_num += 1
                out.write(result)

                if frames_dir and (frame_num - 1) % frame_skip == 0:
                    exported += 1
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{exported:04d}.png"), result)

                if progress_callback and total_frames > 0:
                    progress_callback(frame_num, total_frames)

            if cancelled:
                break

    cap.release()
    out.release()

    if cancelled:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    return True


def pixelate_gif(input_path, output_path, pixel_size, color_reduction, method,
                 palette_rgb=None, show_grid=False, dither='none',
                 export_frames=False, frame_skip=1,
                 progress_callback=None, cancel_flag=None,
                 n_workers=None, overlay=None,
                 gradient_map=None, crt_amount=0.0):
    """Pixelate an animated GIF.

    Mirrors `pixelate_video` semantics: when k-means is used without a fixed
    palette, the centers are computed from the first frame so the colours stay
    consistent across the whole animation.
    """
    try:
        frames, durations = read_gif_frames(input_path)
    except Exception:
        return False

    if not frames:
        return False

    total_frames = len(frames)

    # Set up frame export folder
    frames_dir = None
    if export_frames:
        base = os.path.splitext(output_path)[0]
        frames_dir = base + "_frames"
        os.makedirs(frames_dir, exist_ok=True)

    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    # Compute k-means centers from first frame (small scale)
    kmeans_centers = None
    if not palette_rgb and color_reduction and method == "kmeans":
        first = frames[0]
        h, w = first.shape[:2]
        sw = max(1, w // pixel_size)
        sh = max(1, h // pixel_size)
        small = cv2.resize(first, (sw, sh), interpolation=cv2.INTER_AREA)
        centers, _ = quantize_kmeans(small, color_reduction)
        kmeans_centers = np.uint8(centers)

    def _do_frame(f):
        return pixelate_frame(f, pixel_size, color_reduction, method,
                              palette_rgb, kmeans_centers, show_grid, dither,
                              overlay, gradient_map, crt_amount)

    processed = [None] * total_frames
    exported = 0

    # All frames are already in memory, so we can submit everything at once.
    # executor.map preserves order which is essential for animation timing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i, result in enumerate(executor.map(_do_frame, frames)):
            if cancel_flag and cancel_flag.is_set():
                return False
            processed[i] = result

            if frames_dir and i % frame_skip == 0:
                exported += 1
                cv2.imwrite(os.path.join(frames_dir, f"frame_{exported:04d}.png"), result)

            if progress_callback:
                progress_callback(i + 1, total_frames)

    if cancel_flag and cancel_flag.is_set():
        return False

    return write_gif(processed, output_path, durations=durations)


# ── GUI ───────────────────────────────────────────────────────────────────────

class PixelArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Generator")
        self.root.resizable(False, False)
        self.cancel_flag = threading.Event()
        self.processing = False
        self.imported_palette = None
        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}
        row = 0

        # ── Input file ──
        tk.Label(self.root, text="Input File:").grid(row=row, column=0, sticky='e', **pad)
        self.entry_input = tk.Entry(self.root, width=50)
        self.entry_input.grid(row=row, column=1, columnspan=2, sticky='ew', **pad)
        tk.Button(self.root, text="Browse", command=self._browse_input).grid(row=row, column=3, **pad)

        # ── Output folder ──
        row += 1
        tk.Label(self.root, text="Output Folder:").grid(row=row, column=0, sticky='e', **pad)
        self.entry_output = tk.Entry(self.root, width=50)
        self.entry_output.grid(row=row, column=1, columnspan=2, sticky='ew', **pad)
        tk.Button(self.root, text="Browse", command=self._browse_output).grid(row=row, column=3, **pad)

        # ── Pixel size slider ──
        row += 1
        tk.Label(self.root, text="Pixel Size:").grid(row=row, column=0, sticky='e', **pad)
        self.pixel_size_var = tk.IntVar(value=10)
        self.pixel_slider = tk.Scale(self.root, from_=2, to=64, orient=tk.HORIZONTAL,
                                     variable=self.pixel_size_var, length=250)
        self.pixel_slider.grid(row=row, column=1, sticky='ew', **pad)
        self._add_tooltip(self.pixel_slider, "Higher = blockier pixels. Start with 8-12.")

        # ── Palette selector ──
        row += 1
        tk.Label(self.root, text="Palette:").grid(row=row, column=0, sticky='e', **pad)
        self.palette_var = tk.StringVar(value="Original (auto-reduce)")
        palette_names = list(PALETTES.keys())
        self.palette_menu = ttk.Combobox(self.root, textvariable=self.palette_var,
                                         values=palette_names, state="readonly", width=30)
        self.palette_menu.grid(row=row, column=1, sticky='w', **pad)
        self.palette_menu.bind("<<ComboboxSelected>>", self._on_palette_change)
        self._add_tooltip(self.palette_menu, "Pick a preset palette or import your own.")

        # Palette preview strip
        self.palette_preview = tk.Label(self.root)
        self.palette_preview.grid(row=row, column=2, sticky='w', padx=4)

        # ── Color reduction (only visible for "Original" mode) ──
        row += 1
        self.color_frame = tk.Frame(self.root)
        self.color_frame.grid(row=row, column=0, columnspan=4, sticky='ew')
        self.color_reduce_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.color_frame, text="Color Reduction:", variable=self.color_reduce_var,
                       command=self._toggle_color).grid(row=0, column=0, sticky='e', **pad)
        self.color_slider = tk.Scale(self.color_frame, from_=2, to=64, orient=tk.HORIZONTAL, length=250)
        self.color_slider.set(16)
        self.color_slider.grid(row=0, column=1, sticky='ew', **pad)
        self._add_tooltip(self.color_slider, "Fewer colors = more retro. 8-16 is a good range.")

        # ── Color method (only visible for "Original" mode) ──
        row += 1
        self.method_frame = tk.Frame(self.root)
        self.method_frame.grid(row=row, column=0, columnspan=4, sticky='ew')
        tk.Label(self.method_frame, text="Color Method:").grid(row=0, column=0, sticky='e', **pad)
        self.method_var = tk.StringVar(value="fast")
        rb_frame = tk.Frame(self.method_frame)
        rb_frame.grid(row=0, column=1, sticky='w', **pad)
        tk.Radiobutton(rb_frame, text="Fast", variable=self.method_var, value="fast").pack(side=tk.LEFT)
        tk.Radiobutton(rb_frame, text="K-Means (higher quality)", variable=self.method_var,
                       value="kmeans").pack(side=tk.LEFT)

        # ── Grid + Dither toggles ──
        row += 1
        toggle_frame = tk.Frame(self.root)
        toggle_frame.grid(row=row, column=0, columnspan=4, sticky='w')
        self.grid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(toggle_frame, text="Show Pixel Grid", variable=self.grid_var).pack(
            side=tk.LEFT, padx=12, pady=4)
        tk.Label(toggle_frame, text="Dither:").pack(side=tk.LEFT, padx=(20, 4), pady=4)
        self.dither_var = tk.StringVar(value="None")
        dither_menu = ttk.Combobox(toggle_frame, textvariable=self.dither_var,
            values=["None", "Floyd-Steinberg", "Bayer"],
            state="readonly", width=14)
        dither_menu.pack(side=tk.LEFT, padx=2, pady=4)
        self._add_tooltip(dither_menu,
                          "Floyd-Steinberg: organic, error-diffused noise.\n"
                          "Bayer: ordered threshold matrix — temporally\n"
                          "stable across video frames (no shimmer).")

        # ── Overlay (ASCII / Dot Matrix) ──
        row += 1
        self.overlay_panel = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        self.overlay_panel.grid(row=row, column=0, columnspan=4, sticky='ew',
                                padx=8, pady=4)

        # Mode + Charset row (always-visible part)
        mode_row = tk.Frame(self.overlay_panel)
        mode_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(mode_row, text="Overlay:").pack(side=tk.LEFT, padx=(4, 4))
        self.overlay_mode_var = tk.StringVar(value="None")
        self.overlay_mode_menu = ttk.Combobox(mode_row,
            textvariable=self.overlay_mode_var,
            values=["None", "ASCII", "Dot Matrix"],
            state="readonly", width=12)
        self.overlay_mode_menu.pack(side=tk.LEFT, padx=4)
        self.overlay_mode_menu.bind("<<ComboboxSelected>>", self._on_overlay_change)

        self.overlay_charset_label = tk.Label(mode_row, text="Charset:")
        self.overlay_charset_var = tk.StringVar(value="Standard")
        self.overlay_charset_menu = ttk.Combobox(mode_row,
            textvariable=self.overlay_charset_var,
            values=list(OVERLAY_CHARSETS.keys()),
            state="readonly", width=12)
        # We pack/unpack these together via _on_overlay_change

        # Sliders row (only shown when overlay is active)
        self.overlay_sliders_row = tk.Frame(self.overlay_panel)
        tk.Label(self.overlay_sliders_row, text="Contrast:").pack(side=tk.LEFT, padx=(4, 2))
        self.overlay_contrast_var = tk.IntVar(value=50)
        self.overlay_contrast_slider = tk.Scale(self.overlay_sliders_row,
            from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.overlay_contrast_var, length=160, showvalue=True)
        self.overlay_contrast_slider.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.overlay_contrast_slider,
                          "0 = glyph invisible (matches background).\n"
                          "100 = full black/white separation.")

        tk.Label(self.overlay_sliders_row, text="Bloom:").pack(side=tk.LEFT, padx=(12, 2))
        self.overlay_bloom_var = tk.IntVar(value=0)
        self.overlay_bloom_slider = tk.Scale(self.overlay_sliders_row,
            from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.overlay_bloom_var, length=160, showvalue=True)
        self.overlay_bloom_slider.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.overlay_bloom_slider,
                          "Gaussian glow tinted by glyph colour.\n"
                          "Phosphor / CRT feel at higher values.")

        # ── Gradient Map + CRT ──
        row += 1
        fx_panel = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        fx_panel.grid(row=row, column=0, columnspan=4, sticky='ew', padx=8, pady=4)

        gm_row = tk.Frame(fx_panel)
        gm_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(gm_row, text="Gradient Map:").pack(side=tk.LEFT, padx=(4, 4))
        self.gradient_var = tk.StringVar(value="None")
        gm_menu = ttk.Combobox(gm_row, textvariable=self.gradient_var,
            values=list(GRADIENT_PRESETS.keys()),
            state="readonly", width=20)
        gm_menu.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(gm_menu,
                          "Map luminance to a duotone/polychrome gradient.\n"
                          "Replaces image colours; stacks on top of palette.")

        # Compact gradient preview swatch next to the dropdown
        self.gradient_preview = tk.Label(gm_row)
        self.gradient_preview.pack(side=tk.LEFT, padx=8)
        gm_menu.bind("<<ComboboxSelected>>", self._on_gradient_change)

        crt_row = tk.Frame(fx_panel)
        crt_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(crt_row, text="CRT Effect:").pack(side=tk.LEFT, padx=(4, 4))
        self.crt_var = tk.IntVar(value=0)
        self.crt_slider = tk.Scale(crt_row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.crt_var, length=240, showvalue=True)
        self.crt_slider.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self.crt_slider,
                          "Bundled CRT post-fx: barrel distortion,\n"
                          "chromatic aberration, scanlines, and vignette.\n"
                          "0 = off, 100 = strong.")

        # ── Animated content (video / GIF): Export Frames ──
        row += 1
        self.video_frame = tk.Frame(self.root)
        self.video_frame.grid(row=row, column=0, columnspan=4, sticky='ew')

        self.export_frames_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.video_frame, text="Export Frames as PNGs",
                       variable=self.export_frames_var,
                       command=self._toggle_frame_skip).grid(row=0, column=0, sticky='w', padx=12, pady=2)

        tk.Label(self.video_frame, text="Keep every:").grid(row=1, column=0, sticky='e', padx=12)
        self.frame_skip_var = tk.IntVar(value=1)
        self.frame_skip_slider = tk.Scale(self.video_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                          variable=self.frame_skip_var, length=200)
        self.frame_skip_slider.grid(row=1, column=1, sticky='w', padx=4)
        self.frame_skip_label = tk.Label(self.video_frame, text="frame (1 = all frames)", fg="gray")
        self.frame_skip_label.grid(row=1, column=2, sticky='w')
        self.frame_skip_slider.config(command=self._update_skip_label)

        # Start hidden — only show when an animated file is loaded
        self.video_frame.grid_remove()

        # ── Buttons ──
        row += 1
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=8)
        self.btn_generate = tk.Button(btn_frame, text="Generate Pixel Art", command=self._process,
                                      bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
                                      padx=16, pady=4)
        self.btn_generate.pack(side=tk.LEFT, padx=5)
        self.btn_cancel = tk.Button(btn_frame, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

        # ── Progress ──
        row += 1
        self.progress_bar = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress_bar.grid(row=row, column=0, columnspan=4, padx=10, pady=2)
        self.progress_bar.grid_remove()

        # ── Status ──
        row += 1
        self.status_label = tk.Label(self.root, text="Select an image, video or GIF to get started.",
                                     fg="gray", wraplength=450)
        self.status_label.grid(row=row, column=0, columnspan=4, **pad)

        # ── Preview ──
        row += 1
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=row, column=0, columnspan=4, pady=10)

    # ── Tooltips ──

    def _add_tooltip(self, widget, text):
        tip = tk.Toplevel(widget)
        tip.withdraw()
        tip.overrideredirect(True)
        label = tk.Label(tip, text=text, bg="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("Arial", 9), padx=4, pady=2)
        label.pack()

        def show(e):
            tip.geometry(f"+{e.x_root+10}+{e.y_root+10}")
            tip.deiconify()

        def hide(e):
            tip.withdraw()

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _toggle_frame_skip(self):
        state = tk.NORMAL if self.export_frames_var.get() else tk.DISABLED
        self.frame_skip_slider.config(state=state)

    def _update_skip_label(self, val):
        n = int(val)
        if n == 1:
            self.frame_skip_label.config(text="frame (1 = all frames)")
        else:
            self.frame_skip_label.config(text=f"frames (keep every {n}th frame)")

    # ── Palette Preview Strip ──

    def _draw_palette_preview(self, colors):
        """Draw a small color strip showing the palette colors.
        Vectorised with numpy slice assignment instead of per-pixel putpixel
        loops — same output, dramatically faster for big palettes.
        """
        if not colors:
            self.palette_preview.config(image='')
            self.palette_preview.image = None
            return
        swatch_w = max(4, min(12, 200 // len(colors)))
        swatch_h = 20
        total_w = swatch_w * len(colors)
        arr = np.zeros((swatch_h, total_w, 3), dtype=np.uint8)
        for i, (r, g, b) in enumerate(colors):
            arr[:, i * swatch_w:(i + 1) * swatch_w] = (r, g, b)
        img = Image.fromarray(arr, mode='RGB')
        photo = ImageTk.PhotoImage(img)
        self.palette_preview.config(image=photo)
        self.palette_preview.image = photo

    # ── Palette Change ──

    def _on_palette_change(self, event=None):
        name = self.palette_var.get()
        val = PALETTES.get(name)

        if val == "import":
            filetypes = [
                ("Palette files", "*.hex *.gpl *.png"),
                ("HEX palette", "*.hex"),
                ("GIMP palette", "*.gpl"),
                ("PNG swatch", "*.png"),
                ("All files", "*.*"),
            ]
            path = filedialog.askopenfilename(title="Import Palette", filetypes=filetypes)
            if not path:
                self.palette_var.set("Original (auto-reduce)")
                self._on_palette_change()
                return
            colors = load_palette_from_file(path)
            if not colors:
                messagebox.showerror("Import Error", f"Could not read palette from:\n{path}")
                self.palette_var.set("Original (auto-reduce)")
                self._on_palette_change()
                return
            self.imported_palette = colors
            self.status_label.config(text=f"Imported palette with {len(colors)} colors from {os.path.basename(path)}")
            self._draw_palette_preview(colors)
            self.color_frame.grid_remove()
            self.method_frame.grid_remove()
            return

        if val is None:
            # Original mode — show color reduction controls
            self.imported_palette = None
            self.color_frame.grid()
            self.method_frame.grid()
            self._draw_palette_preview(None)
        else:
            # Preset palette
            self.imported_palette = None
            self.color_frame.grid_remove()
            self.method_frame.grid_remove()
            self._draw_palette_preview(val)

    # ── UI Callbacks ──

    def _toggle_color(self):
        state = tk.NORMAL if self.color_reduce_var.get() else tk.DISABLED
        self.color_slider.config(state=state)

    def _on_overlay_change(self, event=None):
        """Show/hide charset selector and sliders depending on overlay mode."""
        mode = self.overlay_mode_var.get()
        if mode == "None":
            self.overlay_charset_label.pack_forget()
            self.overlay_charset_menu.pack_forget()
            self.overlay_sliders_row.pack_forget()
        elif mode == "ASCII":
            self.overlay_charset_label.pack(side=tk.LEFT, padx=(12, 2))
            self.overlay_charset_menu.pack(side=tk.LEFT, padx=2)
            self.overlay_sliders_row.pack(fill=tk.X, padx=4, pady=2)
        else:  # Dot Matrix
            self.overlay_charset_label.pack_forget()
            self.overlay_charset_menu.pack_forget()
            self.overlay_sliders_row.pack(fill=tk.X, padx=4, pady=2)

    def _on_gradient_change(self, event=None):
        """Draw a thin swatch preview of the selected gradient."""
        name = self.gradient_var.get()
        stops = GRADIENT_PRESETS.get(name)
        if not stops:
            self.gradient_preview.config(image='')
            self.gradient_preview.image = None
            return
        lut = _build_gradient_lut(tuple(tuple(int(c) for c in s) for s in stops))
        if lut is None:
            self.gradient_preview.config(image='')
            self.gradient_preview.image = None
            return
        # lut is BGR; build a (16, 120, 3) RGB strip
        strip_w, strip_h = 120, 16
        idx = np.linspace(0, 255, strip_w).astype(np.int32)
        bgr = lut[idx]                                       # (120, 3) BGR
        rgb = bgr[:, [2, 1, 0]]                              # (120, 3) RGB
        arr = np.broadcast_to(rgb[None, :, :], (strip_h, strip_w, 3)).copy()
        photo = ImageTk.PhotoImage(Image.fromarray(arr, mode='RGB'))
        self.gradient_preview.config(image=photo)
        self.gradient_preview.image = photo

    def _get_overlay_dict(self):
        """Translate UI overlay state into the dict pixelate_frame expects."""
        mode_map = {"None": "none", "ASCII": "ascii", "Dot Matrix": "dot"}
        mode = mode_map.get(self.overlay_mode_var.get(), "none")
        charset_name = self.overlay_charset_var.get()
        charset = OVERLAY_CHARSETS.get(charset_name, OVERLAY_CHARSETS['Standard'])
        return {
            'mode': mode,
            'charset': charset,
            'contrast': self.overlay_contrast_var.get() / 100.0,
            'bloom': self.overlay_bloom_var.get() / 100.0,
        }

    def _browse_input(self):
        filetypes = [
            ("Image, Video & GIF", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif "
                                   "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
            ("GIFs", "*.gif"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        self.entry_input.delete(0, tk.END)
        self.entry_input.insert(0, path)

        if not self.entry_output.get():
            self.entry_output.insert(0, os.path.dirname(path))

        if is_animated(path):
            # Both videos and GIFs use the animated panel
            self.color_reduce_var.set(False)
            self._toggle_color()
            self.method_var.set("fast")
            self.video_frame.grid()
            kind = "GIF" if is_gif(path) else "Video"
            self.status_label.config(text=f"{kind} detected. Color reduction off by default (slow on animated content).")
        else:
            self.color_reduce_var.set(True)
            self._toggle_color()
            self.video_frame.grid_remove()
            self.status_label.config(text="Ready.")

    def _browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def _get_output_path(self, input_path):
        output_dir = self.entry_output.get().strip()
        if not output_dir:
            output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(input_path))
        if is_gif(input_path):
            return os.path.join(output_dir, f"{base}_pixel_art.gif")
        if is_video(input_path):
            return os.path.join(output_dir, f"{base}_pixel_art.mp4")
        return os.path.join(output_dir, f"{base}_pixel_art{ext}")

    # ── Processing ──

    def _get_palette_rgb(self):
        """Return the active palette as a list of (R,G,B) or None for original mode."""
        name = self.palette_var.get()
        if name == "Import from file..." and self.imported_palette:
            return self.imported_palette
        val = PALETTES.get(name)
        if val and val != "import":
            return val
        return None

    def _process(self):
        input_path = self.entry_input.get().strip()
        if not input_path:
            messagebox.showwarning("No file", "Please select an input file first.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("File not found", f"Cannot find:\n{input_path}")
            return

        pixel_size = self.pixel_size_var.get()
        palette_rgb = self._get_palette_rgb()
        show_grid = self.grid_var.get()

        # Dither dropdown → canonical string
        dither_map = {"None": "none", "Floyd-Steinberg": "fs", "Bayer": "bayer"}
        dither = dither_map.get(self.dither_var.get(), "none")

        overlay = self._get_overlay_dict()
        gradient_map = GRADIENT_PRESETS.get(self.gradient_var.get())
        crt_amount = self.crt_var.get() / 100.0

        # Color reduction only applies in "Original" mode
        if palette_rgb:
            color_reduction = None
            method = "fast"
        else:
            use_color = self.color_reduce_var.get()
            color_reduction = self.color_slider.get() if use_color else None
            method = self.method_var.get()

        output_path = self._get_output_path(input_path)

        # Bundle the post-pixelation effects so we don't keep growing positional args
        fx = {
            'overlay': overlay,
            'gradient_map': gradient_map,
            'crt_amount': crt_amount,
        }

        if is_gif(input_path):
            export_frames = self.export_frames_var.get()
            frame_skip = self.frame_skip_var.get()
            self._process_animated(input_path, output_path, pixel_size, color_reduction,
                                   method, palette_rgb, show_grid, dither, fx,
                                   export_frames, frame_skip,
                                   processor=pixelate_gif, kind="GIF")
        elif is_video(input_path):
            export_frames = self.export_frames_var.get()
            frame_skip = self.frame_skip_var.get()
            self._process_animated(input_path, output_path, pixel_size, color_reduction,
                                   method, palette_rgb, show_grid, dither, fx,
                                   export_frames, frame_skip,
                                   processor=pixelate_video, kind="Video")
        else:
            self._process_image(input_path, output_path, pixel_size, color_reduction,
                                method, palette_rgb, show_grid, dither, fx)

    def _process_image(self, input_path, output_path, pixel_size, color_reduction,
                       method, palette_rgb, show_grid, dither, fx):
        self.status_label.config(text="Processing image...")
        self.root.update_idletasks()

        image = cv2.imread(input_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        result = pixelate_frame(image, pixel_size, color_reduction, method,
                                palette_rgb, show_grid=show_grid, dither=dither,
                                overlay=fx['overlay'],
                                gradient_map=fx['gradient_map'],
                                crt_amount=fx['crt_amount'])
        cv2.imwrite(output_path, result)
        self.status_label.config(text=f"Saved to {output_path}")
        self._display_image(output_path)

    def _process_animated(self, input_path, output_path, pixel_size, color_reduction,
                          method, palette_rgb, show_grid, dither, fx,
                          export_frames, frame_skip,
                          processor, kind):
        """Unified runner for video and GIF inputs. `processor` is the function
        that does the actual work (pixelate_video or pixelate_gif).
        """
        self.cancel_flag.clear()
        self.processing = True
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_bar.grid()
        self.status_label.config(text=f"Processing {kind.lower()}...")

        def on_progress(current, total):
            if total <= 0:
                return
            pct = (current / total) * 100
            self.progress_bar['value'] = pct
            self.status_label.config(text=f"Frame {current}/{total} ({pct:.0f}%)")
            self.root.update_idletasks()

        def run():
            success = processor(input_path, output_path, pixel_size, color_reduction,
                                method, palette_rgb, show_grid, dither,
                                export_frames, frame_skip,
                                on_progress, self.cancel_flag,
                                overlay=fx['overlay'],
                                gradient_map=fx['gradient_map'],
                                crt_amount=fx['crt_amount'])
            self.processing = False
            self.btn_generate.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.DISABLED)

            if self.cancel_flag.is_set():
                self.status_label.config(text="Cancelled.")
                self.progress_bar.grid_remove()
                return

            if success:
                msg = f"Saved to {output_path}"
                if export_frames:
                    frames_dir = os.path.splitext(output_path)[0] + "_frames"
                    msg += f"  |  Frames exported to {frames_dir}"
                self.status_label.config(text=msg)

                # Show a preview of the first frame
                if is_gif(output_path):
                    try:
                        img = Image.open(output_path)
                        img.seek(0)
                        self._display_pil(img.convert('RGB'))
                    except Exception:
                        pass
                else:
                    cap = cv2.VideoCapture(output_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self._display_frame(frame)
            else:
                self.status_label.config(text=f"Error processing {kind.lower()}.")
            self.progress_bar.grid_remove()

        threading.Thread(target=run, daemon=True).start()

    def _cancel(self):
        self.cancel_flag.set()

    # ── Display ──

    def _display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_pil(Image.fromarray(frame_rgb))

    def _display_image(self, path):
        self._display_pil(Image.open(path))

    def _display_pil(self, image):
        # Make a copy first so we don't mutate the caller's image
        image = image.copy()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size
        max_size = 350
        ratio = min(max_size / w, max_size / h, 1.0)
        if ratio < 1.0:
            image = image.resize((int(w * ratio), int(h * ratio)), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    PixelArtApp(root)
    root.mainloop()
