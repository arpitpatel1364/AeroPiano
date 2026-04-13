# 🎹 AeroPiano

<p align="center">
  <strong>An immersive, AI-powered spatial piano that turns the air into your stage.</strong><br>
  Built with Computer Vision & Real-time Digital Signal Processing.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-00C5CD?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Pygame-111111?style=for-the-badge&logo=pygame&logoColor=white" alt="Pygame">
</p>

---

## Overview

**AeroPiano** is a futuristic musical instrument that leverages advanced hand tracking to allow anyone to play a virtual piano in 3D space. By tracking 21 hand landmarks in real-time, the system maps your finger tips to virtual piano keys, triggering a high-fidelity sound engine that uses additive synthesis to mimic the physics of a real grand piano.

> [!NOTE]
> This project is a fusion of **Computer Vision** (Mediapipe) and **Physics-based Sound Synthesis**, designed to provide a low-latency, tactile-feeling musical experience without physical contact.

---

## Key Features

-  **Precision Tracking:** Powered by MediaPipe's hand tracking, supporting 10-finger multi-touch and complex chords.
-  **Studio-Grade Synthesis:** Not just samples—AeroPiano uses **Additive Synthesis** with per-harmonic exponential decay and inharmonicity modeling.
-  **Dynamic UI:** Glassmorphic interface with neon glow, particle physics bursts on key strikes, and real-time hand skeleton overlay.
-  **Spatial Control:** Interactive "Pinch-to-Zoom" and drag gestures to reposition the piano in your workspace.
-  **Stereo Imaging:** High-fidelity stereo panning where low notes are staged left and high notes right, creating a realistic soundstage.

---

## Technical Architecture

### Computer Vision Pipeline
The vision system uses `mediapipe` to extract 3D hand coordinates. We focus on the **index tip (LANDMARK_8)** and other finger tips, mapping their normalized camera coordinates to our virtual piano key regions.
- **Inertia Filtering:** Smooths out hand jitters for stable play.
- **Gesture HUD:** Real-time feedback on tracking state and system volume.

###  Sound Engine (DSP)
Unlike standard MIDI players, AeroPiano generates sound from scratch using `NumPy`:
1. **Harmonic Ratios:** 10 harmonics per note with piano-accurate relative amplitudes.
2. **Inharmonicity (Stiffness):** Simulates the physical stiffness of piano strings using the formula `f_n = n * f_0 * sqrt(1 + B * n^2)`.
3. **Soft Saturation:** Uses `tanh` waveshaping to add warmth and prevent clipping.
4. **Envelope Generation:** Exponential decay that scales with frequency (higher notes decay faster).

---

##  Getting Started

### Prerequisites
- **Python 3.10+**
- A standard HD Webcam

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AeroPiano.git
   cd AeroPiano
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

---

##  Controls & Gestures

| Command | Action |
| :--- | :--- |
| `Q` / `ESC` | Exit Application |
| `M` | Toggle Mirror Mode (for intuitive play) |
| `H` | Toggle Hand Skeleton Visualization |
| `S` | Toggle Sustain Pedal |
| `SPACE` | Take a High-Res Snapshot |
| `Pinch` | Zoom the piano interface in/out |
| `Arrows` | Reposition the piano on screen |

---

##  Motivation (Fun & Learning)

This project was built primarily **for fun and as a learning exercise** in:
1. **Computer Vision:** Mastering real-time landmark detection and coordinate mapping.
2. **Digital Signal Processing (DSP):** Understanding how to synthesize complex instruments from pure sine waves and numpy arrays.
3. **User Interface Design:** Crafting a "glassmorphic" aesthetic in a low-level graphics library like Pygame.

It serves as a playground for experimenting with human-computer interaction (HCI) and shows how AI can be used to create magical, non-traditional user experiences.

---

##  Proof of Work

| Proof 1 | Proof 2 |
| :---: | :---: |
| ![Proof 1](Screenshot%202026-04-13%20141353.png) | ![Proof 2](Screenshot%202026-04-13%20141820.png) |

---

## License & Credits

- Developed by **@Arpitpatel1364**(GITHUB)
- **Core Tech:** OpenCV, MediaPipe, Pygame, NumPy.
- Inspired by the intersection of music and technology.

---

<p align="center">
  Made for who is developer and loves music.
</p>
