# 😴 Driver Drowsiness Detection System

A real-time AI-powered road safety system that detects driver drowsiness using computer vision and triggers an alarm before accidents happen.

## 🚨 Problem
Every year 50,000+ road accidents in Pakistan and 1.35 million globally are caused by driver fatigue. Existing solutions are expensive and only available in high-end cars.

## ✅ Solution
A lightweight Python app that uses just a **webcam** to:
- Monitor blink rate in real-time
- Detect if eyes are closed for 2+ seconds
- Trigger loud alarm instantly
- No expensive hardware required!

## 🧠 How It Works
- **Normal driver** = 15-20 blinks/min ✅
- **Drowsy driver** = 25+ blinks/min ⚠️ → Alarm!
- **Sleeping** = Eyes closed 2+ seconds 😴 → Alarm!

## 🛠️ Tech Stack
- Python
- OpenCV
- Tkinter (UI)
- Haar Cascade Classifier

## ▶️ How to Run
```bash
pip install opencv-python pillow
python drowsiness_detector.py
```

## 👩‍💻 Built By
**Zaira Shahid** — AI/ML Developer
[LinkedIn](https://www.linkedin.com/in/zaira-shahid-) | [GitHub](https://github.com/Zaira-Shahid)
