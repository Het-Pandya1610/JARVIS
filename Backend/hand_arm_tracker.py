import cv2  # type: ignore
import numpy as np  # type: ignore
import mediapipe as mp  # type: ignore
import pyautogui  # type: ignore
import time
from collections import deque
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress logs
os.environ["DISABLE_XNNPACK"] = "1"

"""
GestureOS – control (almost) your whole PC with hand + arm tracking.

Dependencies (Windows):
    pip install opencv-python mediapipe pyautogui numpy

Tips:
  • Move the mouse to the top-left corner to instantly abort (PyAutoGUI failsafe)
  • Press Q in the video window to quit
  • Run the script in good lighting; keep your hand ~40–80cm from camera

Default mappings (can be edited in ACTIONS section):
  MOUSE
    • Index tip moves cursor (with smoothing)
    • Pinch (thumb+index) – Left click
    • Hold pinch >0.35s – Drag; release to drop
    • Index+Middle pinch – Right click
  KEYBOARD / EDITING
    • ✊ Fist – Save (Ctrl+S)
    • ✋ Full Palm – Select All (Ctrl+A)
    • ✌️ Victory (index+middle up only) – Copy (Ctrl+C)
    • 🤟 Three fingers up (index+middle+ring) – Paste (Ctrl+V)
    • 🤙 Thumb+Index wide (spread) – Undo (Ctrl+Z)
  WINDOWS / NAVIGATION
    • Horizontal swipe (wrist) Left/Right – Alt+Tab / Win+Tab
    • Vertical swipe Up/Down – Scroll page
    • Palm open held ~1s – Open File Explorer
    • Palm open + swipe Right – Open Browser (Chrome)
    • Fist held ~1.5s – Close window (Alt+F4)
  MEDIA / SYSTEM (guarded with cooldowns)
    • Palm facing camera quick tap – Play/Pause (Space)
    • Palm up steady + move up/down – Volume Up/Down
    • Both hands visible for 2s – Lock workstation

NOTE: Mappings are conservative to avoid accidental triggers. Tweak thresholds/cooldowns below.
"""

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Colors (BGR)
COL_HAND = (255, 200, 50)
COL_PALM = (255, 120, 60)
COL_FINGERS = (255, 180, 80)
COL_FOREARM = (60, 220, 255)
COL_JOINT = (255, 255, 255)
COL_SHOULDER = (200, 255, 200)
COL_WRIST = (180, 100, 255)   # purple wrist
COL_TEXT = (255, 255, 255)

FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
PALM_IDXS = [0, 1, 5, 9, 13, 17]

POSE = mp_pose.PoseLandmark

# Screen properties
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.00

# Gesture state / smoothing
cursor_hist = deque(maxlen=5)
last_action_time = {}

# Adjustable thresholds
PINCH_DIST_PX = 40
RIGHT_PINCH_EXTRA = 30  # index-middle pinch threshold
DRAG_HOLD_SEC = 0.35
CLOSE_HOLD_SEC = 1.5
PALM_OPEN_MIN_FINGERS = 4
COOLDOWN_DEFAULT = 0.8
COOLDOWN_CLICK = 0.25
COOLDOWN_SCROLL = 0.15
VOLUME_STEP_PIXELS = 50
SWIPE_THRESH_PIX = 100
LOCK_BOTH_HANDS_SEC = 2.0

# ROI for mapping camera coords to screen (fraction of frame width/height)
ROI_MARGIN = 0.10  # ignore 10% frame border to reduce jitter at edges

# ------------------------------------------------------------
# Draw helpers (from your base code, kept and expanded)
# ------------------------------------------------------------
def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_palm(img, hand, w, h):
    pts = np.array([to_px(hand.landmark[i], w, h) for i in PALM_IDXS], np.int32)
    hull = cv2.convexHull(pts)
    overlay = img.copy()
    cv2.fillPoly(overlay, [hull], COL_PALM)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.polylines(img, [hull], True, COL_PALM, 1, cv2.LINE_AA)


def draw_fingers(img, hand, w, h):
    for chain in FINGER_CHAINS.values():
        for i in range(len(chain) - 1):
            ax, ay = to_px(hand.landmark[chain[i]], w, h)
            bx, by = to_px(hand.landmark[chain[i + 1]], w, h)
            cv2.line(img, (ax, ay), (bx, by), COL_FINGERS, 2, cv2.LINE_AA)
    for idx in [4, 8, 12, 16, 20]:
        x, y = to_px(hand.landmark[idx], w, h)
        cv2.circle(img, (x, y), 3, COL_FINGERS, -1)


def draw_hand_joints(img, hand, w, h):
    for lm in hand.landmark:
        x, y = to_px(lm, w, h)
        cv2.circle(img, (x, y), 2, COL_JOINT, -1)
    wx, wy = to_px(hand.landmark[0], w, h)
    cv2.circle(img, (wx, wy), 6, COL_WRIST, -1)
    cv2.putText(img, "Wrist", (wx + 8, wy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WRIST, 2)


def draw_pose_arm(img, pose_lms, side, w, h):
    if side == "left":
        wrist = POSE.LEFT_WRIST
        elbow = POSE.LEFT_ELBOW
        shoulder = POSE.LEFT_SHOULDER
        lbl = "L"
    else:
        wrist = POSE.RIGHT_WRIST
        elbow = POSE.RIGHT_ELBOW
        shoulder = POSE.RIGHT_SHOULDER
        lbl = "R"

    lms = pose_lms.landmark
    if (
        lms[wrist].visibility < 0.5
        or lms[elbow].visibility < 0.5
        or lms[shoulder].visibility < 0.5
    ):
        return

    wx, wy = to_px(lms[wrist], w, h)
    ex, ey = to_px(lms[elbow], w, h)
    sx, sy = to_px(lms[shoulder], w, h)

    cv2.circle(img, (wx, wy), 6, COL_WRIST, -1)
    cv2.putText(img, f"{lbl}-Wrist", (wx + 8, wy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WRIST, 2)

    cv2.line(img, (wx, wy), (ex, ey), COL_FOREARM, 3, cv2.LINE_AA)
    cv2.putText(
        img,
        f"{lbl}-Forearm",
        ((wx + ex) // 2, (wy + ey) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COL_FOREARM,
        2,
    )

    cv2.circle(img, (ex, ey), 6, COL_JOINT, -1)
    cv2.putText(img, f"{lbl}-Elbow", (ex + 8, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_JOINT, 2)

    cv2.circle(img, (sx, sy), 7, COL_SHOULDER, -1)
    cv2.putText(
        img, f"{lbl}-Shoulder", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_SHOULDER, 2
    )


# ------------------------------------------------------------
# Gesture utilities
# ------------------------------------------------------------

def _dist(a, b, w, h):
    ax, ay = to_px(a, w, h)
    bx, by = to_px(b, w, h)
    return np.hypot(bx - ax, by - ay)


def fingers_up(hand):
    """Return list of 5 booleans [thumb, index, middle, ring, pinky] whether each finger is up."""
    lms = hand.landmark
    up = [False] * 5
    # Thumb: compare x depending on handedness is tricky; use tip to IP distance relative to wrist
    up[0] = lms[4].y < lms[3].y - 0.005  # rough vertical heuristic
    # Other fingers: tip above PIP
    up[1] = lms[8].y < lms[6].y
    up[2] = lms[12].y < lms[10].y
    up[3] = lms[16].y < lms[14].y
    up[4] = lms[20].y < lms[18].y
    return up


def is_pinch(hand, w, h, thresh=PINCH_DIST_PX):
    return _dist(hand.landmark[4], hand.landmark[8], w, h) < thresh


def is_two_finger_pinch(hand, w, h, extra=RIGHT_PINCH_EXTRA):
    # index-middle tips together
    return _dist(hand.landmark[8], hand.landmark[12], w, h) < (PINCH_DIST_PX + extra)


def is_palm_open(hand):
    up = fingers_up(hand)
    return sum(up[1:]) >= PALM_OPEN_MIN_FINGERS  # count non-thumb fingers


def is_fist(hand):
    up = fingers_up(hand)
    return sum(up) == 0


def is_victory(hand):
    up = fingers_up(hand)
    return up[1] and up[2] and not up[3] and not up[4]


def is_three_up(hand):
    up = fingers_up(hand)
    return up[1] and up[2] and up[3] and not up[4]


def is_undo_spread(hand, w, h):
    # thumb-index far apart horizontally
    return _dist(hand.landmark[4], hand.landmark[8], w, h) > PINCH_DIST_PX * 2


# ------------------------------------------------------------
# Action trigger helpers
# ------------------------------------------------------------

def cooldown(name, secs):
    t = time.time()
    last = last_action_time.get(name, 0)
    if t - last >= secs:
        last_action_time[name] = t
        return True
    return False


def move_cursor_to(ix, iy, frame_w, frame_h):
    # Map camera coords (ix,iy in pixels) inside an ROI to screen space with smoothing
    rx0, ry0 = int(ROI_MARGIN * frame_w), int(ROI_MARGIN * frame_h)
    rx1, ry1 = int((1 - ROI_MARGIN) * frame_w), int((1 - ROI_MARGIN) * frame_h)
    x = np.clip(ix, rx0, rx1)
    y = np.clip(iy, ry0, ry1)
    # Normalize within ROI
    nx = (x - rx0) / max(1, (rx1 - rx0))
    ny = (y - ry0) / max(1, (ry1 - ry0))
    sx = int(nx * SCREEN_W)
    sy = int(ny * SCREEN_H)
    cursor_hist.append((sx, sy))
    avgx = int(np.mean([p[0] for p in cursor_hist]))
    avgy = int(np.mean([p[1] for p in cursor_hist]))
    pyautogui.moveTo(avgx, avgy, duration=0)  # instant, we already smooth


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    drag_active = False
    pinch_start_t = 0.0
    palm_hold_t = 0.0
    fist_hold_t = 0.0
    both_hands_seen_t = 0.0

    # For swipe detection using wrist trajectory
    wrist_hist = deque(maxlen=6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_res = hands.process(rgb)
        pose_res = pose.process(rgb)

        # Pose overlay (optional visual aid)
        if pose_res.pose_landmarks is not None:
            draw_pose_arm(frame, pose_res.pose_landmarks, "left", w, h)
            draw_pose_arm(frame, pose_res.pose_landmarks, "right", w, h)

        hands_count = 0

        if hand_res.multi_hand_landmarks:
            # Draw + control for each hand
            for hand in hand_res.multi_hand_landmarks:
                hands_count += 1
                draw_palm(frame, hand, w, h)
                draw_fingers(frame, hand, w, h)
                draw_hand_joints(frame, hand, w, h)

                # === Cursor control with index fingertip ===
                ix, iy = to_px(hand.landmark[8], w, h)
                move_cursor_to(ix, iy, w, h)

                # === Clicks / Drag ===
                if is_pinch(hand, w, h):
                    if pinch_start_t == 0:
                        pinch_start_t = time.time()
                    held = time.time() - pinch_start_t
                    if held >= DRAG_HOLD_SEC and not drag_active:
                        if cooldown("drag_start", 0.2):
                            pyautogui.mouseDown()
                            drag_active = True
                            cv2.putText(frame, "DRAGGING", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif held < DRAG_HOLD_SEC:
                        # quick pinch = click
                        if cooldown("click", COOLDOWN_CLICK):
                            pyautogui.click()
                            cv2.putText(frame, "CLICK", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    pinch_start_t = 0
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False

                # Right click with index-middle pinch
                if is_two_finger_pinch(hand, w, h):
                    if cooldown("rclick", 0.6):
                        pyautogui.click(button="right")
                        cv2.putText(frame, "RIGHT CLICK", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # === Editing / shortcuts ===
                if is_fist(hand):
                    if fist_hold_t == 0:
                        fist_hold_t = time.time()
                    # Long hold closes window (guarded)
                    if time.time() - fist_hold_t > CLOSE_HOLD_SEC:
                        if cooldown("close_window", 2.5):
                            pyautogui.hotkey("alt", "f4")
                            cv2.putText(frame, "ALT+F4", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        if cooldown("save", COOLDOWN_DEFAULT):
                            pyautogui.hotkey("ctrl", "s")
                            cv2.putText(frame, "SAVE (Ctrl+S)", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    fist_hold_t = 0

                if is_palm_open(hand):
                    if palm_hold_t == 0:
                        palm_hold_t = time.time()
                    # Hold to open File Explorer
                    if time.time() - palm_hold_t > 1.0:
                        if cooldown("explorer", 4.0):
                            try:
                                pyautogui.hotkey("win", "e")  # reliable shortcut
                            except Exception:
                                pass
                            cv2.putText(frame, "Open Explorer", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    palm_hold_t = 0

                if is_victory(hand) and cooldown("copy", COOLDOWN_DEFAULT):
                    pyautogui.hotkey("ctrl", "c")
                    cv2.putText(frame, "COPY (Ctrl+C)", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if is_three_up(hand) and cooldown("paste", COOLDOWN_DEFAULT):
                    pyautogui.hotkey("ctrl", "v")
                    cv2.putText(frame, "PASTE (Ctrl+V)", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if is_undo_spread(hand, w, h) and cooldown("undo", 1.0):
                    pyautogui.hotkey("ctrl", "z")
                    cv2.putText(frame, "UNDO (Ctrl+Z)", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # === Scroll / Swipes (use wrist path) ===
                wx, wy = to_px(hand.landmark[0], w, h)
                wrist_hist.append((wx, wy, time.time()))
                if len(wrist_hist) >= 2:
                    (x0, y0, t0) = wrist_hist[0]
                    (x1, y1, t1) = wrist_hist[-1]
                    dx, dy = x1 - x0, y1 - y0
                    dt = max(1e-3, t1 - t0)
                    vx, vy = dx / dt, dy / dt
                    # horizontal swipe
                    if abs(dx) > SWIPE_THRESH_PIX and abs(vx) > 400 and cooldown("swipe_lr", 1.2):
                        if dx < 0:
                            pyautogui.hotkey("alt", "tab")
                            cv2.putText(frame, "ALT+TAB", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            pyautogui.hotkey("win", "tab")
                            cv2.putText(frame, "WIN+TAB", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    # vertical swipe -> scroll
                    if abs(dy) > SWIPE_THRESH_PIX and abs(vy) > 400 and cooldown("swipe_ud", COOLDOWN_SCROLL):
                        pyautogui.scroll(-int(np.sign(dy) * 300))

                # === Media / Volume ===
                # Quick palm tap to send SPACE (play/pause)
                if is_palm_open(hand) and cooldown("media_toggle", 1.2):
                    # Heuristic: if recent vertical velocity small and palm was closed before, treat as tap.
                    if len(wrist_hist) >= 2:
                        dy = wrist_hist[-1][1] - wrist_hist[0][1]
                        if abs(dy) < 25:
                            pyautogui.press("space")
                            cv2.putText(frame, "PLAY/PAUSE", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Volume by vertical movement while palm open
                if is_palm_open(hand) and len(wrist_hist) >= 2:
                    dy = wrist_hist[-1][1] - wrist_hist[-2][1]
                    if dy < -VOLUME_STEP_PIXELS and cooldown("vol_up", 0.25):
                        pyautogui.press("volumeup")
                    elif dy > VOLUME_STEP_PIXELS and cooldown("vol_down", 0.25):
                        pyautogui.press("volumedown")

            # Both-hands system lock (guarded)
            if hands_count >= 2:
                if both_hands_seen_t == 0:
                    both_hands_seen_t = time.time()
                elif time.time() - both_hands_seen_t > LOCK_BOTH_HANDS_SEC and cooldown("lock", 10.0):
                    try:
                        # Lock workstation (Windows)
                        import subprocess
                        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
                        cv2.putText(frame, "LOCK WORKSTATION", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    except Exception:
                        pass
            else:
                both_hands_seen_t = 0

        # HUD
        cv2.putText(
            frame,
            "GestureOS – Press Q to quit | Move to top-left to failsafe",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2,
        )

        # Draw ROI box
        rx0, ry0 = int(ROI_MARGIN * w), int(ROI_MARGIN * h)
        rx1, ry1 = int((1 - ROI_MARGIN) * w), int((1 - ROI_MARGIN) * h)
        cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (100, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Cursor ROI", (rx0 + 8, ry0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)

        cv2.imshow("GestureOS – Hand & Arm Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()


if __name__ == "__main__":
    main()
