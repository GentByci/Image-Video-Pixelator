import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import threading

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

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


# ── Color Quantaization ───────────────────────────────────────────────────────

def map_to_palette(image, palette_rgb):
    """Map every pixel to the nearest color in a fixed palette (Euclidean RGB)."""
    # palette_rgb is list of (R,G,B), image is BGR
    palette_bgr = np.array([(b, g, r) for r, g, b in palette_rgb], dtype=np.uint8)
    flat = image.reshape(-1, 3).astype(np.int16)
    # Broadcast distance calculation
    dists = np.sum((flat[:, None, :] - palette_bgr[None, :, :].astype(np.int16)) ** 2, axis=2)
    indices = np.argmin(dists, axis=1)
    return palette_bgr[indices].reshape(image.shape).astype(np.uint8)


def quantize_fast(image, levels):
    """Fast color quantization via bit-shifting."""
    factor = 256 // levels
    return (image // factor * factor + factor // 2).astype(np.uint8)


def quantize_kmeans(image, k):
    """K-means color quantization — slower but higher quality."""
    Z = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers, labels


def apply_kmeans_palette(image, centers, k):
    """Map pixels to a precomputed k-means palette."""
    Z = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    result = np.uint8(centers)[labels.flatten()]
    return result.reshape(image.shape)


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


# ── Frame / Video Processing ─────────────────────────────────────────────────

def pixelate_frame(frame, pixel_size, color_reduction=None, method="fast",
                   palette_rgb=None, kmeans_centers=None, show_grid=False):
    height, width = frame.shape[:2]
    sw = max(1, width // pixel_size)
    sh = max(1, height // pixel_size)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    # Color mapping
    if palette_rgb:
        pixelated = map_to_palette(pixelated, palette_rgb)
    elif color_reduction and color_reduction > 0:
        if method == "fast":
            pixelated = quantize_fast(pixelated, color_reduction)
        elif method == "kmeans" and kmeans_centers is not None:
            pixelated = apply_kmeans_palette(pixelated, kmeans_centers, color_reduction)
        elif method == "kmeans":
            centers, labels = quantize_kmeans(pixelated, color_reduction)
            result = np.uint8(centers)[labels.flatten()]
            pixelated = result.reshape(pixelated.shape)

    if show_grid:
        pixelated = draw_grid(pixelated, pixel_size)

    return pixelated


def pixelate_video(input_path, output_path, pixel_size, color_reduction, method,
                   palette_rgb=None, show_grid=False,
                   progress_callback=None, cancel_flag=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # For k-means video without a fixed palette: compute centers from first frame
    kmeans_centers = None
    if not palette_rgb and color_reduction and method == "kmeans":
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return False
        sw = max(1, width // pixel_size)
        sh = max(1, height // pixel_size)
        small = cv2.resize(first_frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        kmeans_centers, _ = quantize_kmeans(pix, color_reduction)
        kmeans_centers = np.uint8(kmeans_centers)
        out.write(pixelate_frame(first_frame, pixel_size, color_reduction, method,
                                 palette_rgb, kmeans_centers, show_grid))
        if progress_callback:
            progress_callback(1, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    frame_num = 1 if kmeans_centers is not None else 0
    while True:
        if cancel_flag and cancel_flag.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(pixelate_frame(frame, pixel_size, color_reduction, method,
                                 palette_rgb, kmeans_centers, show_grid))
        frame_num += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_num, total_frames)

    cap.release()
    out.release()

    if cancel_flag and cancel_flag.is_set():
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    return True


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

        # ── Grid overlay ──
        row += 1
        self.grid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Show Pixel Grid", variable=self.grid_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=12, pady=4)
        self._add_tooltip_for_row(row, "Draws grid lines at pixel boundaries — useful as a drawing reference.")

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
        self.status_label = tk.Label(self.root, text="Select an image or video to get started.",
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

    def _add_tooltip_for_row(self, row, text):
        # For checkbuttons that span columns — attach tooltip to the root at that row
        pass  # Tooltips on checkbuttons are tricky; skipping for now

    # ── Palette Preview Strip ──

    def _draw_palette_preview(self, colors):
        """Draw a small color strip showing the palette colors."""
        if not colors:
            self.palette_preview.config(image='')
            self.palette_preview.image = None
            return
        swatch_w = max(4, min(12, 200 // len(colors)))
        swatch_h = 20
        img = Image.new("RGB", (swatch_w * len(colors), swatch_h))
        for i, (r, g, b) in enumerate(colors):
            for x in range(swatch_w):
                for y in range(swatch_h):
                    img.putpixel((i * swatch_w + x, y), (r, g, b))
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

    def _browse_input(self):
        filetypes = [
            ("Image & Video", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
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

        if is_video(path):
            self.color_reduce_var.set(False)
            self._toggle_color()
            self.method_var.set("fast")
            self.status_label.config(text="Video detected. Color reduction off by default (slow on video).")
        else:
            self.color_reduce_var.set(True)
            self._toggle_color()
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

        # Color reduction only applies in "Original" mode
        if palette_rgb:
            color_reduction = None
            method = "fast"
        else:
            use_color = self.color_reduce_var.get()
            color_reduction = self.color_slider.get() if use_color else None
            method = self.method_var.get()

        output_path = self._get_output_path(input_path)

        if is_video(input_path):
            self._process_video(input_path, output_path, pixel_size, color_reduction,
                                method, palette_rgb, show_grid)
        else:
            self._process_image(input_path, output_path, pixel_size, color_reduction,
                                method, palette_rgb, show_grid)

    def _process_image(self, input_path, output_path, pixel_size, color_reduction,
                       method, palette_rgb, show_grid):
        self.status_label.config(text="Processing image...")
        self.root.update_idletasks()

        image = cv2.imread(input_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        result = pixelate_frame(image, pixel_size, color_reduction, method,
                                palette_rgb, show_grid=show_grid)
        cv2.imwrite(output_path, result)
        self.status_label.config(text=f"Saved to {output_path}")
        self._display_image(output_path)

    def _process_video(self, input_path, output_path, pixel_size, color_reduction,
                       method, palette_rgb, show_grid):
        self.cancel_flag.clear()
        self.processing = True
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_bar.grid()
        self.status_label.config(text="Processing video...")

        def on_progress(current, total):
            pct = (current / total) * 100
            self.progress_bar['value'] = pct
            self.status_label.config(text=f"Frame {current}/{total} ({pct:.0f}%)")
            self.root.update_idletasks()

        def run():
            success = pixelate_video(input_path, output_path, pixel_size, color_reduction,
                                     method, palette_rgb, show_grid, on_progress, self.cancel_flag)
            self.processing = False
            self.btn_generate.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.DISABLED)

            if self.cancel_flag.is_set():
                self.status_label.config(text="Cancelled.")
                self.progress_bar.grid_remove()
                return

            if success:
                self.status_label.config(text=f"Saved to {output_path}")
                cap = cv2.VideoCapture(output_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self._display_frame(frame)
            else:
                self.status_label.config(text="Error processing video.")
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
        w, h = image.size
        max_size = 350
        ratio = min(max_size / w, max_size / h, 1.0)
        image = image.resize((int(w * ratio), int(h * ratio)))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    PixelArtApp(root)
    root.mainloop()
