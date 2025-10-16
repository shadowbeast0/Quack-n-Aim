# Quack n Aim

_Webcam-controlled carnival duck shooter powered by OpenCV, MediaPipe, and Pygame._

Move the on-screen crosshair by raising your **index finger**; fire a shot by flashing **two fingers** (e.g., index + middle) or by quickly making a fist and releasing. Survive timed rounds, pop moving targets, chain combos, and chase a high score — all hands-free.

![Quack n Aim Demo](docs/demo.gif)

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.9+ -blue.svg">Python 3.9+</a>
  <a href="https://img.shields.io/badge/OpenCV-Computer%20Vision-brightgreen">OpenCV</a>
  <a href="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange">MediaPipe</a>
  <a href="https://img.shields.io/badge/Pygame-Arcade%20Loop-lightgrey">Pygame</a>
</p>

---

## ✨ Features

- **Hands-only input:** index-finger **point-to-move**, **two-finger** gesture to shoot (rising-edge detection + cooldown).
- **Timed rounds:** classic arcade pressure with a visible countdown (default: ~60s, configurable).
- **Stage progression:** early stages start easy; target speed and density ramp up.
- **Smooth pointer:** high-FPS loop with motion smoothing for stable aiming.
- **Simple visuals:** clean black backdrop keeps focus on targets and pointer.
- **Configurable:** tweak resolution, FPS, round time, target size/count, lanes, mirroring, etc.

---

## 🕹️ How to Play

1. Ensure your webcam is connected and visible to the OS.
2. Launch the game (see **Quick Start**).
3. Hold your hand up to the camera:
   - **Move:** raise your **index finger** — the crosshair follows.
   - **Shoot:** quickly show **two or more fingers** to fire or show your hand, make a fist and release quickly.
4. Pop targets, chain hits, and beat the clock. Your score and stages cleared show at the end.
5. Press **Esc** (or close the window) to exit.

---

## 📦 Requirements

- **Python:** 3.9–3.12 recommended  
- **Dependencies:** `opencv-python`, `mediapipe`, `pygame`, `numpy`

---

## 🚀 Quick Start

> If your main script is named `game.py` (as in this repo), replace accordingly if you use a different filename.

**Using venv (cross-platform):**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python game.py
