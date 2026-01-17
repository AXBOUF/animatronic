
---

````markdown
## Animatronic Eyes – Real-Time Eye Tracking Prototype 👀

A small computer vision project where animatronic-style eyes follow the viewer in real time by detecting eye position from a camera feed.

This is a **rapid 20-minute prototype** built to explore real-time feedback loops between computer vision and visual movement, not a polished or production-ready system.

---

## What this project does

- Captures live video from a camera
- Detects eyes using basic computer vision techniques
- Maps detected eye position to on-screen eye movement
- Creates the illusion that the eyes are tracking the viewer

---

## Project structure

```text
.
├── src/
│   └── eyeball_tracker.py     # core eye-tracking logic
├── templates/
│   └── index.html             # simple frontend for visualization
├── experiments/
│   └── phase1/                # early experimentation and iterations
└── README.md
````

---

## Tech stack

* Python
* OpenCV
* Basic real-time computer vision

---

## How to run

1. Install dependencies:

   ```bash
   pip install opencv-python
   ```

2. Run the tracker:

   ```bash
   python src/eyeball_tracker.py
   ```

---

## Camera source note (important)

By default, the code uses:

```python
cv2.VideoCapture(0)
```

If you are using an **external camera**, you may need to change the camera index:

* `0` → built-in / default webcam
* `1`, `2`, … → external cameras

Example:

```python
cv2.VideoCapture(1)
```

Adjust this value until the correct camera feed appears.

---

## Notes

* This is a quick experimental build, not a production system
* Accuracy depends on lighting and camera quality
* Intended for learning, curiosity, and rapid prototyping

---

## Possible extensions

* Smoother eye movement interpolation
* Face tracking instead of eye-only detection
* Physical animatronic eyes using servo motors
* Web-based version using WebRTC

