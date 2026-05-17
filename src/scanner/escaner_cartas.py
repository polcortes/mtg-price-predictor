import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Callable
import os
import sys
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Silence OpenCV's noisy backend warnings on Windows
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

class CardDetector:
    CARD_WIDTH = 450
    CARD_HEIGHT = 630
    MIN_AREA = 5000
    CANNY_LOW = 50
    CANNY_HIGH = 150

    def detect(self, frame):
        """Return (annotated_frame, warped_card_or_None)."""
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < self.MIN_AREA:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.1 * peri, True)
            if len(approx) != 4:
                continue

            cv2.drawContours(display, [approx], 0, (0, 255, 0), 3)
            pts = approx.reshape(4, 2).astype("float32")
            rect = self._order_corners(pts)
            dst = np.array(
                [[0, 0], [self.CARD_WIDTH, 0],
                 [self.CARD_WIDTH, self.CARD_HEIGHT], [0, self.CARD_HEIGHT]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (self.CARD_WIDTH, self.CARD_HEIGHT))
            return display, warped

        return display, None

    @staticmethod
    def _order_corners(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


def _open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows to avoid backend noise."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    return cap


def list_cameras(max_test: int = 6) -> list:
    """Return indices of cameras that can be opened. Suppresses stderr during probing."""
    # Redirect stderr to suppress OpenCV's C-level warnings while probing
    devnull_fd = None
    saved_stderr_fd = None
    try:
        if sys.platform == "win32":
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
    except OSError:
        pass

    available = []
    try:
        for i in range(max_test):
            cap = _open_camera(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
    finally:
        try:
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)
        except OSError:
            pass

    return available


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------

class CardScannerApp(tk.Toplevel):
    PREVIEW_W = 640
    PREVIEW_H = 480
    CARD_PREV_W = CardDetector.CARD_WIDTH // 2
    CARD_PREV_H = CardDetector.CARD_HEIGHT // 2
    POLL_MS = 33  # ~30 fps

    def __init__(self, master: tk.Misc, output_dir: str = "cartas_recortadas", callback: Optional[Callable[[str], None]] = None):
        super().__init__(master)
        self.title("MTG Card Scanner")
        self.resizable(False, False)
        self.grab_set()  # make it modal

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.counter = 0

        self.detector = CardDetector()
        self.cap: Optional[cv2.VideoCapture] = None
        self._job = None
        self._pending_card: Optional[np.ndarray] = None
        self._paused = False
        self._cam_ids: list = []
        self._on_save_callback = callback

        # Strong PhotoImage references — kept on self so GC never collects them
        # before tkinter's Tcl layer finishes rendering.
        self._photo_preview: Optional[ImageTk.PhotoImage] = None
        self._photo_card: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        # Probe cameras only after the window exists and Tcl is ready
        self.after(100, self._populate_cameras)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(bar, text="Camera:").pack(side="left")
        self.camera_var = tk.StringVar(value="Detecting cameras…")
        self.camera_combo = ttk.Combobox(
            bar, textvariable=self.camera_var, state="disabled", width=22
        )
        self.camera_combo.pack(side="left", padx=(4, 12))

        self.btn_start = ttk.Button(bar, text="Start", command=self._start, state="disabled")
        self.btn_start.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(bar, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=2)

        # Live preview — use a Label, not a Canvas, for reliable image display on Windows
        self.preview_label = ttk.Label(self, background="black",
                                       width=self.PREVIEW_W, anchor="center")
        self.preview_label.pack(padx=8, pady=4)
        # Reserve fixed pixel size so the window doesn't jump
        self.preview_label.config(width=self.PREVIEW_W)
        placeholder = tk.PhotoImage(width=self.PREVIEW_W, height=self.PREVIEW_H)
        self.preview_label.config(image=placeholder)
        self._placeholder_preview = placeholder  # keep ref

        # Card preview + action panel
        card_frame = ttk.LabelFrame(self, text="Detected Card")
        card_frame.pack(fill="x", padx=8, pady=4)

        self.card_label = ttk.Label(card_frame, background="#d0d0d0", anchor="center")
        self.card_label.pack(side="left", padx=8, pady=8)
        placeholder2 = tk.PhotoImage(width=self.CARD_PREV_W, height=self.CARD_PREV_H)
        self.card_label.config(image=placeholder2)
        self._placeholder_card = placeholder2  # keep ref

        action_panel = ttk.Frame(card_frame)
        action_panel.pack(side="left", padx=16, anchor="center")

        self.lbl_status = ttk.Label(action_panel, text="No card detected yet.", wraplength=200)
        self.lbl_status.pack(pady=(0, 12))

        self.btn_save = ttk.Button(
            action_panel, text="Save card", command=self._save_card, state="disabled"
        )
        self.btn_save.pack(fill="x", pady=4)

        self.btn_retry = ttk.Button(
            action_panel, text="Try again", command=self._retry, state="disabled"
        )
        self.btn_retry.pack(fill="x", pady=4)

        # Status bar
        self.statusbar = ttk.Label(self, text="Detecting cameras…", anchor="w", relief="sunken")
        self.statusbar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _populate_cameras(self):
        self._set_status("Detecting cameras…")
        cams = list_cameras()
        if not cams:
            self.camera_combo["values"] = ["No cameras found"]
            self.camera_var.set("No cameras found")
            self._set_status("No cameras found.")
            return
        labels = [f"Camera {i}" for i in cams]
        self.camera_combo["values"] = labels
        self.camera_combo.config(state="readonly")
        self.camera_combo.current(0)
        self._cam_ids = cams
        self.btn_start.config(state="normal")
        self._set_status("Ready. Select a camera and press Start.")

    def _selected_camera_id(self) -> int:
        return self._cam_ids[self.camera_combo.current()]

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def _start(self):
        cam_id = self._selected_camera_id()
        self.cap = _open_camera(cam_id)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open Camera {cam_id}.")
            self.cap = None
            return

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.camera_combo.config(state="disabled")
        self._paused = False
        self._pending_card = None
        self._set_status("Scanning — hold a card up to the camera.")
        self._schedule_poll()

    def _stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.camera_combo.config(state="readonly")
        # Restore blank placeholders
        self.preview_label.config(image=self._placeholder_preview)
        self._photo_preview = None
        self._set_status("Stopped.")

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _schedule_poll(self):
        self._job = self.after(self.POLL_MS, self._poll)

    def _poll(self):
        self._job = None
        if self._paused or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._set_status("Camera read error — try restarting.")
            return

        display, card = self.detector.detect(frame)
        self._update_label(self.preview_label, display, self.PREVIEW_W, self.PREVIEW_H, "preview")

        if card is not None and self._pending_card is None:
            self._pending_card = card
            self._paused = True
            self._update_label(self.card_label, card, self.CARD_PREV_W, self.CARD_PREV_H, "card")
            self.lbl_status.config(text="Card detected!\nSave it or try again.")
            self.btn_save.config(state="normal")
            self.btn_retry.config(state="normal")
            self._set_status("Card detected — choose an action.")
            return  # don't reschedule while paused

        self._schedule_poll()

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def _save_card(self):
        if self._pending_card is None:
            return
        path = self.output_dir / f"carta_{self.counter}.jpg"
        cv2.imwrite(str(path), self._pending_card)
        self.counter += 1
        self.lbl_status.config(text=f"Saved as {path.name}.")
        self._set_status(f"Saved: {path.name}")
        print(f"{self._on_save_callback=}")
        if self._on_save_callback:
            self._on_save_callback(str(path))
        # self._resume()
        
        # app.protocol("WM_DELETE_WINDOW", lambda: (app._on_close(), root.destroy()))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _retry(self):
        self.card_label.config(image=self._placeholder_card)
        self._photo_card = None
        self.lbl_status.config(text="Scanning again…")
        self._resume()

    def _resume(self):
        self._pending_card = None
        self.btn_save.config(state="disabled")
        self.btn_retry.config(state="disabled")
        self._paused = False
        self._set_status("Scanning…")
        self._schedule_poll()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_label(self, label: ttk.Label, frame: np.ndarray, w: int, h: int, slot: str):
        """Convert an OpenCV BGR frame to a PhotoImage and assign it to a Label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        photo = ImageTk.PhotoImage(image=img, master=self)  # master= binds to our Tk root
        # Store on self BEFORE assigning to the label — prevents any GC window
        if slot == "preview":
            self._photo_preview = photo
            label.config(image=self._photo_preview)
        else:
            self._photo_card = photo
            label.config(image=self._photo_card)

    def _set_status(self, msg: str):
        self.statusbar.config(text=msg)

    def _on_close(self):
        self._stop()
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def scan_card(master: Optional[tk.Misc] = None, output_dir: str = "cartas_recortadas", callback: Optional[Callable[[str], None]] = None):
    """Open the scanner window.

    If *master* is given (a live Tk root or Toplevel), the scanner opens as a
    modal child window on the **calling thread** — no extra thread needed.

    If *master* is None the function creates its own Tk root and blocks until
    the window is closed (useful when running the scanner standalone).
    """
    if master is not None:
        # Embedded in an existing app — open as a modal Toplevel and return
        # immediately; the caller's mainloop drives everything.
        CardScannerApp(master, output_dir=output_dir, callback=callback)
        return

    # Standalone mode
    root = tk.Tk()
    root.withdraw()  # hide the empty root window
    app = CardScannerApp(root, output_dir=output_dir, callback=callback)
    app.protocol("WM_DELETE_WINDOW", lambda: (app._on_close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    scan_card()