# Image and Video Pixelator

Convert images and videos into pixel art. Built for artists who want to generate pixel art references from real photos and footage.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Image & Video pixelation** — supports PNG, JPG, BMP, WEBP, MP4, AVI, MOV, MKV, and more
- **Artist palettes** — PICO-8, GameBoy, Endesga-32, Endesga-16, DawnBringer-32, or import your own (.hex, .gpl, .png)
- **Grid overlay** — toggleable pixel grid for counting and reference
- **Two color modes** — Fast (instant) or K-Means (higher quality)
- **Video processing** — progress bar, cancel button, threaded so the UI stays responsive

## Quick Start

### Option 1: Download the exe
Grab `Pixelator.exe` from [Releases](../../releases) and run it. No install needed.

### Option 2: Run from source
```bash
pip install -r requirements.txt
python main_pixelator.py
```

## How to Use

1. Click **Browse** to select an image or video
2. Set your **Output Folder** (defaults to the input file's directory)
3. Adjust the **Pixel Size** slider (higher = blockier)
4. Pick a **Palette** from the dropdown, or choose "Import from file..." to load your own
5. Toggle **Show Pixel Grid** if you want grid lines as a drawing reference
6. Click **Generate Pixel Art**

<img width="645" height="620" alt="image" src="https://github.com/user-attachments/assets/08b36077-0872-45d5-9d59-64fbf11a13cd" />


## Requirements

- Python 3.8+
- opencv-python
- numpy
- Pillow
