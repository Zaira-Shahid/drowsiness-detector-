import cv2
import winsound
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from collections import deque


# ════════════════════════════════════════════════════
#   DRIVER DROWSINESS DETECTION SYSTEM
#   Built by Zaira Shahid | AI/ML Project
# ════════════════════════════════════════════════════

class DrowsinessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System — Zaira Shahid")
        self.root.state("zoomed")
        self.root.configure(bg="#080b14")
        self.root.resizable(True, True)

        # ── Detection State
        self.running = False
        self.cap = None
        self.alarm_on = False
        self.total_alerts = 0
        self.session_start = None

        # ── Blink Rate Tracking
        self.eye_closed_frames = 0
        self.eyes_were_closed = False
        self.MIN_BLINK_FRAMES = 2         # Min frames = valid blink start
        self.MAX_BLINK_FRAMES = 60        # > 60 frames = eyes stuck closed (sleeping)
        self.FPS = 30                     # Approximate camera FPS

        # Rolling window: timestamps of last blinks (60 sec window)
        self.blink_timestamps = deque()
        self.WINDOW_SECONDS = 60          # 1 minute window
        self.NORMAL_MAX_BPM = 20          # Normal: up to 20 blinks/min
        self.WARNING_BPM = 25             # Warning: 25+ blinks/min
        self.DANGER_BPM = 30              # Danger: 30+ blinks/min

        # Eyes closed too long = sleeping
        self.eyes_closed_start = None
        self.SLEEP_SECONDS = 2.0          # 2 sec eyes closed = sleeping

        # ── Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.build_ui()
        self.update_clock()
        self.animate_status_dot()

    # ══════════════════════════════════════════════════
    #   UI
    # ══════════════════════════════════════════════════
    def build_ui(self):
        BG       = "#080b14"
        CARD     = "#0f1422"
        BORDER   = "#1e2d50"
        GREEN    = "#00ff88"
        BLUE     = "#38bdf8"
        ORANGE   = "#fb923c"
        PINK     = "#f472b6"
        RED      = "#ff3333"
        GRAY     = "#4a5568"
        WHITE    = "#e2e8f0"
        MONO     = "Courier"

        # ─── Header ───────────────────────────────────
        header = tk.Frame(self.root, bg=BG, pady=12)
        header.pack(fill="x", padx=25)

        left_head = tk.Frame(header, bg=BG)
        left_head.pack(side="left")

        tk.Label(left_head,
                 text="◈  DRIVER DROWSINESS DETECTION",
                 font=(MONO, 17, "bold"),
                 fg=GREEN, bg=BG).pack(anchor="w")

        tk.Label(left_head,
                 text="    Real-time AI Road Safety System  •  Python + OpenCV",
                 font=(MONO, 9),
                 fg=GRAY, bg=BG).pack(anchor="w")

        right_head = tk.Frame(header, bg=BG)
        right_head.pack(side="right")

        self.clock_label = tk.Label(right_head,
                                    font=(MONO, 12),
                                    fg=BLUE, bg=BG)
        self.clock_label.pack(anchor="e")

        self.live_dot = tk.Label(right_head, text="⬤  OFFLINE",
                                 font=(MONO, 10, "bold"),
                                 fg=RED, bg=BG)
        self.live_dot.pack(anchor="e")

        # Divider
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=25)

        # ─── Button Bar ───────────────────────────────
        btn_bar = tk.Frame(self.root, bg=BG, pady=8)
        btn_bar.pack(fill="x", padx=25)

        self.start_btn = tk.Button(btn_bar,
                                   text="▶   START MONITORING",
                                   font=(MONO, 11, "bold"),
                                   bg=GREEN, fg="#080b14",
                                   activebackground="#00cc66",
                                   relief="flat", cursor="hand2",
                                   padx=25, pady=9,
                                   command=self.start_monitoring)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(btn_bar,
                                  text="■   STOP MONITORING",
                                  font=(MONO, 11, "bold"),
                                  bg="#1e2d50", fg=GRAY,
                                  activebackground="#2a3a60",
                                  relief="flat", cursor="hand2",
                                  padx=25, pady=9,
                                  state="disabled",
                                  command=self.stop_monitoring)
        self.stop_btn.pack(side="left")

        # Alert badge
        self.alert_badge = tk.Label(btn_bar,
                                    text="",
                                    font=(MONO, 11, "bold"),
                                    fg=RED, bg=BG)
        self.alert_badge.pack(side="right")

        # Divider
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=25)

        # ─── Content ──────────────────────────────────
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=10)

        # ── Right Panel ───────────────────────────────
        right = tk.Frame(content, bg=BG, width=320)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        # Status
        self._section(right, "DRIVER STATUS")
        self.status_label = tk.Label(right, text="STANDBY",
                                     font=(MONO, 20, "bold"),
                                     fg=GRAY, bg=CARD,
                                     pady=10)
        self.status_label.pack(fill="x", padx=2, pady=(0, 10))

        # Blink Rate
        self._section(right, "BLINK RATE  (per minute)")
        bpm_frame = tk.Frame(right, bg=CARD)
        bpm_frame.pack(fill="x", padx=2)

        self.bpm_label = tk.Label(bpm_frame, text="0",
                                  font=(MONO, 40, "bold"),
                                  fg=GREEN, bg=CARD)
        self.bpm_label.pack(side="left", padx=(15, 0))

        bpm_info = tk.Frame(bpm_frame, bg=CARD)
        bpm_info.pack(side="left", padx=15, pady=10)

        tk.Label(bpm_info, text="bpm",
                 font=(MONO, 12), fg=GRAY,
                 bg=CARD).pack(anchor="w")

        self.bpm_status = tk.Label(bpm_info, text="Normal",
                                   font=(MONO, 10, "bold"),
                                   fg=GREEN, bg=CARD)
        self.bpm_status.pack(anchor="w")

        # BPM progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("bpm.Horizontal.TProgressbar",
                        troughcolor="#1e2d50",
                        background=GREEN)
        self.bpm_bar = ttk.Progressbar(right, length=290,
                                        mode='determinate',
                                        maximum=35,
                                        style="bpm.Horizontal.TProgressbar")
        self.bpm_bar.pack(pady=(4, 2), padx=15)

        # Range legend
        legend = tk.Frame(right, bg=BG)
        legend.pack(fill="x", padx=15, pady=(2, 8))
        tk.Label(legend, text="0─────15", font=(MONO, 7),
                 fg="#2d5a3d", bg=BG).pack(side="left")
        tk.Label(legend, text="20─25", font=(MONO, 7),
                 fg="#7a5a00", bg=BG).pack(side="left", padx=8)
        tk.Label(legend, text="30+", font=(MONO, 7),
                 fg="#7a1a1a", bg=BG).pack(side="left")

        # Eyes Closed Duration
        self._section(right, "EYES CLOSED DURATION")
        self.eyes_closed_label = tk.Label(right, text="0.0 sec",
                                          font=(MONO, 28, "bold"),
                                          fg=BLUE, bg=CARD,
                                          pady=8)
        self.eyes_closed_label.pack(fill="x", padx=2, pady=(0, 10))

        # Total Alerts
        self._section(right, "TOTAL ALERTS THIS SESSION")
        self.alert_count_label = tk.Label(right, text="0",
                                          font=(MONO, 32, "bold"),
                                          fg=ORANGE, bg=CARD,
                                          pady=8)
        self.alert_count_label.pack(fill="x", padx=2, pady=(0, 10))

        # Session + Last Alert row
        row = tk.Frame(right, bg=BG)
        row.pack(fill="x", pady=(0, 8))

        left_col = tk.Frame(row, bg=BG)
        left_col.pack(side="left", fill="x", expand=True)
        self._section(left_col, "SESSION")
        self.session_label = tk.Label(left_col, text="00:00:00",
                                      font=(MONO, 13, "bold"),
                                      fg=BLUE, bg=CARD, pady=6)
        self.session_label.pack(fill="x", padx=2)

        right_col = tk.Frame(row, bg=BG)
        right_col.pack(side="right", fill="x", expand=True, padx=(8, 0))
        self._section(right_col, "LAST ALERT")
        self.last_alert_label = tk.Label(right_col, text="--:--",
                                         font=(MONO, 13, "bold"),
                                         fg=PINK, bg=CARD, pady=6)
        self.last_alert_label.pack(fill="x", padx=2)

        # ── Camera Feed ───────────────────────────────
        left_panel = tk.Frame(content, bg=BG)
        left_panel.pack(side="left", fill="both", expand=True)

        cam_border = tk.Frame(left_panel, bg=GREEN, padx=1, pady=1)
        cam_border.pack(fill="both", expand=True)

        cam_inner = tk.Frame(cam_border, bg=CARD)
        cam_inner.pack(fill="both", expand=True)

        self.cam_label = tk.Label(cam_inner, bg="#0a0e1a",
                                  text="📷\n\nCamera feed will appear here\n\nClick  ▶ START MONITORING  to begin",
                                  font=(MONO, 13), fg="#1e3a5f")
        self.cam_label.pack(fill="both", expand=True)

        # ─── Footer ───────────────────────────────────
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=25)
        tk.Label(self.root,
                 text="Built by Zaira Shahid  •  AI / ML Project  •  Road Safety System  •  Python + OpenCV + Computer Vision",
                 font=(MONO, 8), fg="#2a3a5a", bg=BG).pack(pady=6)

    def _section(self, parent, title):
        tk.Label(parent, text=title,
                 font=("Courier", 8, "bold"),
                 fg="#4a6a9a", bg="#080b14",
                 anchor="w").pack(fill="x", padx=2, pady=(6, 1))

    # ══════════════════════════════════════════════════
    #   ANIMATION
    # ══════════════════════════════════════════════════
    def animate_status_dot(self):
        if self.running:
            current = self.live_dot.cget("fg")
            self.live_dot.config(fg="#00ff88" if current == "#0d2a1a" else "#0d2a1a")
        self.root.after(600, self.animate_status_dot)

    def update_clock(self):
        self.clock_label.config(
            text=datetime.now().strftime("%H:%M:%S  •  %d %b %Y"))
        if self.running and self.session_start:
            elapsed = int(time.time() - self.session_start)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            self.session_label.config(text=f"{h:02}:{m:02}:{s:02}")
        self.root.after(1000, self.update_clock)

    # ══════════════════════════════════════════════════
    #   CONTROLS
    # ══════════════════════════════════════════════════
    def start_monitoring(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.start_btn.config(state="disabled", bg="#1e2d50", fg="#4a5568")
        self.stop_btn.config(state="normal", bg="#ff3333", fg="white",
                             activebackground="#cc0000")
        self.live_dot.config(text="⬤  LIVE", fg="#00ff88")
        # Reset
        self.blink_timestamps.clear()
        self.eye_closed_frames = 0
        self.eyes_were_closed = False
        self.total_alerts = 0
        self.eyes_closed_start = None
        self.session_start = time.time()
        self.alert_count_label.config(text="0")
        self.last_alert_label.config(text="--:--")
        self.alert_badge.config(text="")
        threading.Thread(target=self.detect_loop, daemon=True).start()

    def stop_monitoring(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state="normal", bg="#00ff88", fg="#080b14")
        self.stop_btn.config(state="disabled", bg="#1e2d50", fg="#4a5568")
        self.live_dot.config(text="⬤  OFFLINE", fg="#ff3333")
        self.status_label.config(text="STANDBY", fg="#4a5568")
        self.bpm_label.config(text="0", fg="#00ff88")
        self.bpm_status.config(text="Normal", fg="#00ff88")
        self.bpm_bar["value"] = 0
        self.eyes_closed_label.config(text="0.0 sec", fg="#38bdf8")
        self.session_label.config(text="00:00:00")
        self.cam_label.config(image="",
                              text="📷\n\nCamera feed will appear here\n\nClick  ▶ START MONITORING  to begin")

    def play_alarm(self):
        if not self.alarm_on:
            self.alarm_on = True
            for _ in range(5):
                winsound.Beep(1400, 250)
                time.sleep(0.05)
                winsound.Beep(700, 250)
                time.sleep(0.05)
            self.alarm_on = False

    # ══════════════════════════════════════════════════
    #   CORE DETECTION LOOP
    # ══════════════════════════════════════════════════
    def detect_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()
            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Clean old blink timestamps outside 60s window
            while (self.blink_timestamps and
                   now - self.blink_timestamps[0] > self.WINDOW_SECONDS):
                self.blink_timestamps.popleft()

            blinks_per_min = len(self.blink_timestamps)

            # ── Face detection
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(80, 80),
                maxSize=(int(fw * 0.85), int(fh * 0.85)))

            status = "NO FACE"
            status_color = "#fb923c"
            eyes_open_now = False
            eyes_closed_duration = 0.0
            alert_triggered = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 136), 2)

                # Upper 55% of face = eye region
                ey_end = y + int(h * 0.55)
                roi_gray  = gray[y:ey_end, x:x+w]
                roi_color = frame[y:ey_end, x:x+w]

                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=8,
                    minSize=(25, 25),
                    maxSize=(int(w * 0.45), int(h * 0.35)))

                if len(eyes) >= 2:
                    # ── Eyes OPEN ──────────────────────
                    eyes_open_now = True

                    # Blink completed?
                    if (self.eyes_were_closed and
                            self.MIN_BLINK_FRAMES <= self.eye_closed_frames <= self.MAX_BLINK_FRAMES):
                        self.blink_timestamps.append(now)

                    self.eye_closed_frames = 0
                    self.eyes_were_closed = False
                    self.eyes_closed_start = None

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color,
                                      (ex, ey), (ex+ew, ey+eh),
                                      (56, 189, 248), 2)
                else:
                    # ── Eyes CLOSED ────────────────────
                    self.eye_closed_frames += 1
                    self.eyes_were_closed = True

                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = now

                    eyes_closed_duration = now - self.eyes_closed_start

            # ── Status & Alarm Logic ──────────────────

            if len(faces) == 0:
                # No face
                cv2.putText(frame, "Adjust camera — face not detected",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (251, 146, 60), 2)
                status = "NO FACE"
                status_color = "#fb923c"

            elif eyes_open_now:
                # Eyes open — check blink rate
                if blinks_per_min <= self.NORMAL_MAX_BPM:
                    status = "AWAKE  ✓"
                    status_color = "#00ff88"
                elif blinks_per_min < self.DANGER_BPM:
                    status = "DROWSY  ⚠"
                    status_color = "#fb923c"
                    alert_triggered = True
                else:
                    status = "⚠  VERY DROWSY!"
                    status_color = "#ff3333"
                    alert_triggered = True

            else:
                # Eyes closed — check duration
                if eyes_closed_duration >= self.SLEEP_SECONDS:
                    status = "⚠  SLEEPING!"
                    status_color = "#ff3333"
                    alert_triggered = True
                else:
                    status = "EYES CLOSING..."
                    status_color = "#fb923c"

            # ── Draw overlay if alert ─────────────────
            if alert_triggered:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 150), -1)
                cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

                label = ("SLEEPING! WAKE UP!"
                         if eyes_closed_duration >= self.SLEEP_SECONDS
                         else f"HIGH BLINK RATE: {blinks_per_min} bpm!")
                cv2.putText(frame, label,
                            (int(fw * 0.06), int(fh * 0.55)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (0, 0, 255), 3)

                self.total_alerts += 1
                now_str = datetime.now().strftime("%H:%M")
                threading.Thread(target=self.play_alarm, daemon=True).start()
                self.root.after(0, lambda t=now_str:
                                self.last_alert_label.config(text=t))
                self.root.after(0, lambda:
                                self.alert_badge.config(
                                    text=f"🚨  ALERT #{self.total_alerts}"))

                # Cool down — avoid spam
                time.sleep(3)
                self.root.after(0, lambda: self.alert_badge.config(text=""))

            # ── Blink rate color ──────────────────────
            if blinks_per_min <= self.NORMAL_MAX_BPM:
                bpm_clr = "#00ff88"
                bpm_txt = "Normal ✓"
            elif blinks_per_min < self.DANGER_BPM:
                bpm_clr = "#fb923c"
                bpm_txt = "⚠ Warning"
            else:
                bpm_clr = "#ff3333"
                bpm_txt = "⚠ Danger!"

            # ── Update UI ────────────────────────────
            closed_str = f"{eyes_closed_duration:.1f} sec"
            self.root.after(0, self._update_ui,
                            status, status_color,
                            blinks_per_min, bpm_clr, bpm_txt,
                            closed_str, self.total_alerts)

            # ── Send frame to canvas ──────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((860, 560))
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self._update_frame, imgtk)

        self.running = False

    def _update_frame(self, imgtk):
        self.cam_label.imgtk = imgtk
        self.cam_label.config(image=imgtk, text="")

    def _update_ui(self, status, s_clr,
                   bpm, bpm_clr, bpm_txt,
                   closed_str, alerts):
        self.status_label.config(text=status, fg=s_clr)
        self.bpm_label.config(text=str(bpm), fg=bpm_clr)
        self.bpm_status.config(text=bpm_txt, fg=bpm_clr)
        self.bpm_bar["value"] = min(bpm, 35)
        self.eyes_closed_label.config(text=closed_str)
        self.alert_count_label.config(text=str(alerts))


# ════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectorApp(root)
    root.mainloop()