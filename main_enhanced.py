"""
main_enhanced.py
================
Enhanced X-ray Baggage Scanner — OPIXray Edition

Improvements over the original main.py
---------------------------------------
• English UI                              (was French)
• 5-class detection with per-class colours (was 1 binary "Danger" class)
• Threat level panel  (CLEAR / LOW / HIGH)
• Live detection statistics sidebar
• Per-image detection history log
• Keyboard shortcuts: SPACE = pause, R = random, P = positive, N = negative, Q = quit
• Resizable window support
• Confidence threshold slider (drag the bar at the bottom)

Usage
-----
    python main_enhanced.py
    python main_enhanced.py --model model/model_trained.pt   # trained model
"""

import argparse
import os
import sys
import random
import threading
import time
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pygame
except ImportError:
    raise SystemExit("pygame not installed.  Run: pip install pygame")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — detection disabled.")

# ── Constants ──────────────────────────────────────────────────────────────
WIN_W, WIN_H    = 1340, 750
SIDEBAR_W       = 280
CONVEYOR_X      = SIDEBAR_W + 10
CONVEYOR_W      = WIN_W - SIDEBAR_W - 20
CONVEYOR_H      = 560
CONVEYOR_Y      = (WIN_H - CONVEYOR_H) // 2 + 20
SCROLL_SPEED    = 2
IMG_HEIGHT_RATIO= 0.68
MIN_SPACING     = 30

# Per-class colour palette (R, G, B)
CLASS_COLORS = {
    0: (255,  80,  80),   # Folding Knife    — red
    1: (255, 180,  30),   # Straight Knife   — amber
    2: ( 60, 200,  60),   # Scissors         — green
    3: ( 80, 140, 255),   # Utility Knife    — blue
    4: (200,  60, 220),   # Multi-tool Knife — purple
}
DEFAULT_COLOR = (200, 200, 200)

# UI colours
C_BG        = ( 18,  22,  38)
C_PANEL     = ( 26,  32,  52)
C_BORDER    = ( 60,  80, 130)
C_WHITE     = (255, 255, 255)
C_GRAY      = (160, 170, 190)
C_DARK_GRAY = ( 80,  90, 110)
C_CONVEYOR  = ( 30,  36,  56)
C_CLEAR     = ( 50, 200,  80)
C_LOW       = (230, 180,  30)
C_HIGH      = (220,  40,  40)

THREAT_NONE = 0
THREAT_LOW  = 1
THREAT_HIGH = 2

HISTORY_MAX = 12    # lines in the history log

# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="",
                        help="Path to trained .pt model (empty = auto-detect)")
    parser.add_argument("--images", default=".",
                        help="Root folder containing images/[train|test]/[pos|neg]/")
    parser.add_argument("--conf",   type=float, default=0.30,
                        help="Detection confidence threshold (default 0.30)")
    return parser.parse_args()

# ── Model wrapper ──────────────────────────────────────────────────────────

class Detector:
    CLASS_NAMES = [
        "Folding_Knife",
        "Straight_Knife",
        "Scissors",
        "Utility_Knife",
        "Multi-tool_Knife",
    ]

    def __init__(self, model_dir="model", explicit_model="", conf=0.30):
        self.model       = None
        self.class_names = {}
        self.conf        = conf
        self._lock       = threading.Lock()
        self._cache      = {}

        if not YOLO_AVAILABLE:
            return

        # Locate model
        candidates = []
        if explicit_model:
            candidates.append(Path(explicit_model))
        candidates += [
            Path(model_dir) / "model_trained.pt",
            Path(model_dir) / "model_exported.pt",
            Path(model_dir) / "model_exported.onnx",
        ]

        loaded = False
        for p in candidates:
            if p.exists():
                try:
                    self.model = YOLO(str(p), task="detect")
                    print(f"✅  Model loaded: {p}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[WARN] Could not load {p}: {e}")

        if not loaded:
            print("[WARN] No model found. Falling back to yolov8m.pt …")
            try:
                self.model = YOLO("yolov8m.pt")
            except Exception as e:
                print(f"[ERROR] {e}")
                return

        # Load class names
        classes_file = Path(model_dir) / "classes.txt"
        if classes_file.exists():
            with open(classes_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        self.class_names[int(parts[0])] = parts[1].strip()
        else:
            if self.model and self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = {i: n for i, n in enumerate(self.CLASS_NAMES)}

        print(f"   Classes: {self.class_names}")

    def detect(self, pil_image: Image.Image, img_path: str):
        if self.model is None:
            return []
        if img_path in self._cache:
            return self._cache[img_path]

        img_arr = np.array(pil_image)
        try:
            results = self.model(img_arr, conf=self.conf, verbose=False)
        except Exception as e:
            print(f"[Detection error] {e}")
            return []

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"Class{cls_id}")
                color    = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf_val, "cls_id": cls_id,
                    "cls_name": cls_name, "color": color,
                })

        with self._lock:
            self._cache[img_path] = detections
        return detections


# ── Scrolling image object ─────────────────────────────────────────────────

class XrayImage:
    def __init__(self, img_path: str, conveyor_rect, target_h: int):
        self.path        = img_path
        self.detections  = []
        self.done        = False
        self.orig_w = self.orig_h = 1

        try:
            self.pil_img = Image.open(img_path).convert("RGB")
            self.orig_w, self.orig_h = self.pil_img.size
            ratio   = target_h / self.orig_h
            disp_w  = int(self.orig_w * ratio)
            self.pg_img = pygame.image.fromstring(
                self.pil_img.resize((disp_w, target_h), Image.LANCZOS).tobytes(),
                (disp_w, target_h), "RGB"
            )
        except Exception as e:
            print(f"[IMG load error] {e}")
            self.pg_img  = pygame.Surface((300, target_h))
            self.pg_img.fill((60, 60, 60))
            self.pil_img = None
            self.done    = True

        self.disp_w, self.disp_h = self.pg_img.get_size()
        self.x = conveyor_rect.x - self.disp_w - 10
        self.y = conveyor_rect.y + (conveyor_rect.height - self.disp_h) // 2
        self.conveyor_rect = conveyor_rect
        self.finished  = False

    def update(self, speed):
        self.x += speed
        if self.x > self.conveyor_rect.right + 50:
            self.finished = True

    def set_detections(self, raw_detections):
        sx = self.disp_w / max(self.orig_w, 1)
        sy = self.disp_h / max(self.orig_h, 1)
        self.detections = []
        for d in raw_detections:
            det = d.copy()
            det["rx1"] = int(d["x1"] * sx)
            det["ry1"] = int(d["y1"] * sy)
            det["rx2"] = int(d["x2"] * sx)
            det["ry2"] = int(d["y2"] * sy)
            self.detections.append(det)
        self.done = True

    def in_scan_zone(self):
        cx = self.x + self.disp_w / 2
        zone_cx = self.conveyor_rect.centerx
        return abs(cx - zone_cx) < self.conveyor_rect.width * 0.22

    def draw(self, surface, clip_rect, font_det):
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        surface.blit(self.pg_img, (self.x, self.y))

        for det in self.detections:
            ax1 = self.x + det["rx1"]
            ay1 = self.y + det["ry1"]
            ax2 = self.x + det["rx2"]
            ay2 = self.y + det["ry2"]
            r, g, b = det["color"]
            pygame.draw.rect(surface, (r, g, b),
                             (ax1, ay1, ax2 - ax1, ay2 - ay1), 3)
            label = f"{det['cls_name']}  {det['conf']:.2f}"
            txt   = font_det.render(label, True, C_WHITE)
            bg_r  = pygame.Rect(ax1, ay1 - txt.get_height() - 4,
                                 txt.get_width() + 6, txt.get_height() + 4)
            pygame.draw.rect(surface, (r, g, b), bg_r)
            surface.blit(txt, (ax1 + 3, ay1 - txt.get_height() - 2))

        surface.set_clip(old_clip)


# ── Image manager ──────────────────────────────────────────────────────────

class ImageManager:
    def __init__(self, base_folder, conveyor_rect, detector: Detector):
        self.base       = Path(base_folder)
        self.rect       = conveyor_rect
        self.detector   = detector
        self.target_h   = int(conveyor_rect.height * IMG_HEIGHT_RATIO)
        self.active     = []
        self.queue      = deque()
        self.paused     = False

    def _rand_path(self, split="test", label=None):
        if label is None:
            label = random.choice(["pos", "neg"])
        folder = self.base / "images" / split / label
        if not folder.exists():
            return None
        imgs = [f for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        return str(random.choice(imgs)) if imgs else None

    def enqueue(self, split="test", label=None):
        p = self._rand_path(split, label)
        if p:
            self.queue.append(p)
            self._try_load()

    def _try_load(self):
        if not self.queue:
            return
        if self.active:
            last = self.active[-1]
            if last.x < self.rect.x - MIN_SPACING:
                return
        path = self.queue.popleft()
        img  = XrayImage(path, self.rect, self.target_h)
        self.active.append(img)
        if img.pil_img and not img.done:
            t = threading.Thread(target=self._detect, args=(img,), daemon=True)
            t.start()

    def _detect(self, img: XrayImage):
        dets = self.detector.detect(img.pil_img, img.path)
        img.set_detections(dets)

    def update(self):
        if self.paused:
            return self._current_dets()
        for img in self.active:
            img.update(SCROLL_SPEED)
        self.active = [i for i in self.active if not i.finished]
        self._try_load()
        return self._current_dets()

    def _current_dets(self):
        for img in self.active:
            if img.in_scan_zone() and img.detections:
                return img.detections
        return []

    def draw(self, surface, clip_rect, font_det):
        for img in self.active:
            img.draw(surface, clip_rect, font_det)


# ── UI drawing helpers ─────────────────────────────────────────────────────

def draw_panel(surface, rect, color=C_PANEL, border=C_BORDER, radius=10):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 2, border_radius=radius)


def draw_text(surface, text, font, color, x, y, align="left"):
    surf = font.render(text, True, color)
    if align == "center":
        x -= surf.get_width() // 2
    elif align == "right":
        x -= surf.get_width()
    surface.blit(surf, (x, y))
    return surf.get_height()


# ── Main application ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("OPIXray — X-ray Dangerous Object Detection")
    clock  = pygame.time.Clock()

    # Fonts
    font_title  = pygame.font.SysFont("Arial", 20, bold=True)
    font_body   = pygame.font.SysFont("Arial", 15)
    font_small  = pygame.font.SysFont("Arial", 13)
    font_det    = pygame.font.SysFont("Arial", 13, bold=True)
    font_big    = pygame.font.SysFont("Arial", 28, bold=True)
    font_mono   = pygame.font.SysFont("Courier New", 12)

    # Conveyor rect
    conv_rect = pygame.Rect(CONVEYOR_X, CONVEYOR_Y, CONVEYOR_W, CONVEYOR_H)

    # Detector + manager
    detector = Detector(
        model_dir="model",
        explicit_model=args.model,
        conf=args.conf
    )
    manager  = ImageManager(args.images, conv_rect, detector)

    # State
    threat_level  = THREAT_NONE
    conf_threshold= args.conf
    history_log   = deque(maxlen=HISTORY_MAX)
    class_counts  = defaultdict(int)
    total_scanned = 0
    last_dets     = []
    paused        = False

    # Slider rect (confidence threshold bar)
    SLIDER_Y     = WIN_H - 38
    SLIDER_X0    = CONVEYOR_X + 10
    SLIDER_W     = CONVEYOR_W - 20
    SLIDER_H     = 12
    slider_drag  = False

    def conf_to_x(c):
        return int(SLIDER_X0 + c * SLIDER_W)

    def x_to_conf(x):
        return max(0.05, min(0.95, (x - SLIDER_X0) / SLIDER_W))

    running = True
    while running:
        mouse = pygame.mouse.get_pos()

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    manager.paused = paused
                elif event.key == pygame.K_r:
                    manager.enqueue("test", None)
                elif event.key == pygame.K_p:
                    manager.enqueue("test", "pos")
                elif event.key == pygame.K_n:
                    manager.enqueue("test", "neg")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    sx = conf_to_x(conf_threshold)
                    if abs(mouse[0] - sx) < 14 and abs(mouse[1] - SLIDER_Y) < 14:
                        slider_drag = True
                    # Button clicks
                    bx = 14
                    for i, (label, split, lbl) in enumerate([
                        ("Random",  "test", None),
                        ("Positive","test", "pos"),
                        ("Negative","test", "neg"),
                    ]):
                        by = WIN_H // 2 + i * 56 - 70
                        br = pygame.Rect(bx, by, SIDEBAR_W - 28, 44)
                        if br.collidepoint(mouse):
                            manager.enqueue(split, lbl)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    slider_drag = False

            elif event.type == pygame.MOUSEMOTION:
                if slider_drag:
                    conf_threshold  = x_to_conf(mouse[0])
                    detector.conf   = conf_threshold

        # ── Update ──────────────────────────────────────────────────────
        current_dets = manager.update()

        if current_dets != last_dets:
            if current_dets:
                total_scanned += 1
                for d in current_dets:
                    class_counts[d["cls_name"]] += 1
                names = list({d["cls_name"] for d in current_dets})
                history_log.appendleft(
                    f"⚠  {', '.join(names)} — conf {max(d['conf'] for d in current_dets):.2f}"
                )
                threat_level = THREAT_HIGH if any(d["conf"] > 0.6 for d in current_dets) \
                               else THREAT_LOW
            else:
                if last_dets:           # just cleared
                    history_log.appendleft("✔  CLEAR")
                    total_scanned += 1
                threat_level = THREAT_NONE
            last_dets = current_dets

        # ── Draw background ──────────────────────────────────────────────
        screen.fill(C_BG)

        # ── Left sidebar ─────────────────────────────────────────────────
        sb_rect = pygame.Rect(0, 0, SIDEBAR_W, WIN_H)
        draw_panel(screen, sb_rect, C_PANEL, C_BORDER, 0)

        # Title
        draw_text(screen, "OPIXray Scanner", font_title, C_WHITE,
                  SIDEBAR_W // 2, 14, "center")
        draw_text(screen, "X-ray Threat Detection", font_small, C_GRAY,
                  SIDEBAR_W // 2, 38, "center")
        pygame.draw.line(screen, C_BORDER, (10, 58), (SIDEBAR_W - 10, 58), 1)

        # Threat panel
        tcolor = [C_CLEAR, C_LOW, C_HIGH][threat_level]
        tlabel = ["✔  CLEAR", "⚠  LOW THREAT", "🚨 HIGH THREAT"][threat_level]
        t_rect = pygame.Rect(10, 68, SIDEBAR_W - 20, 54)
        draw_panel(screen, t_rect, (*tcolor, 40) if False else C_PANEL, tcolor, 8)
        pygame.draw.rect(screen, tcolor, t_rect, 3, border_radius=8)
        draw_text(screen, tlabel, font_big if threat_level == THREAT_HIGH else font_title,
                  tcolor, SIDEBAR_W // 2, 82, "center")

        # Class legend
        y = 136
        draw_text(screen, "CLASSES", font_small, C_GRAY, 14, y)
        y += 18
        for cls_id, name in enumerate(Detector.CLASS_NAMES):
            col = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
            pygame.draw.rect(screen, col, (14, y + 3, 12, 12), border_radius=2)
            cnt = class_counts.get(name, 0)
            draw_text(screen, f"{name.replace('_',' ')}  ({cnt})", font_small,
                      C_WHITE, 32, y)
            y += 20

        # Stats
        y += 10
        pygame.draw.line(screen, C_BORDER, (10, y), (SIDEBAR_W - 10, y), 1)
        y += 8
        draw_text(screen, "STATISTICS", font_small, C_GRAY, 14, y);  y += 18
        draw_text(screen, f"Total scanned : {total_scanned}", font_small, C_WHITE, 14, y); y += 18
        dangerous = sum(v for k, v in class_counts.items())
        draw_text(screen, f"Threats found : {dangerous}", font_small, C_LOW, 14, y); y += 18
        draw_text(screen, f"Conf threshold: {conf_threshold:.2f}", font_small, C_GRAY, 14, y); y += 14

        # Control buttons
        y = WIN_H // 2 - 70
        for label, col in [("Random  [R]",  C_BORDER),
                            ("Positive [P]", C_CLEAR),
                            ("Negative [N]", C_GRAY)]:
            br = pygame.Rect(14, y, SIDEBAR_W - 28, 44)
            hover = br.collidepoint(mouse)
            bc    = tuple(min(255, v + 40) for v in col) if hover else col
            draw_panel(screen, br, C_PANEL, bc, 8)
            draw_text(screen, label, font_body, C_WHITE, SIDEBAR_W // 2, y + 13, "center")
            y += 56

        # Pause indicator
        if paused:
            p_rect = pygame.Rect(14, y, SIDEBAR_W - 28, 34)
            draw_panel(screen, p_rect, C_PANEL, C_LOW, 8)
            draw_text(screen, "⏸  PAUSED  [SPACE]", font_small, C_LOW,
                      SIDEBAR_W // 2, y + 9, "center")
        else:
            p_rect = pygame.Rect(14, y, SIDEBAR_W - 28, 34)
            draw_panel(screen, p_rect, C_PANEL, C_DARK_GRAY, 8)
            draw_text(screen, "SPACE to Pause", font_small, C_GRAY,
                      SIDEBAR_W // 2, y + 9, "center")

        # Detection history log
        log_y = WIN_H - 32 - HISTORY_MAX * 17 - 28
        pygame.draw.line(screen, C_BORDER, (10, log_y), (SIDEBAR_W - 10, log_y), 1)
        log_y += 6
        draw_text(screen, "HISTORY", font_small, C_GRAY, 14, log_y);  log_y += 18
        for entry in history_log:
            col = C_HIGH if "⚠" in entry or "🚨" in entry else C_CLEAR
            draw_text(screen, entry[:34], font_mono, col, 8, log_y)
            log_y += 16

        # ── Conveyor ─────────────────────────────────────────────────────
        draw_panel(screen, pygame.Rect(CONVEYOR_X - 4, CONVEYOR_Y - 4,
                                       CONVEYOR_W + 8, CONVEYOR_H + 8),
                   C_CONVEYOR, C_BORDER, 10)
        pygame.draw.rect(screen, C_CONVEYOR, conv_rect, border_radius=8)

        # Scan zone indicator lines
        zone_cx = conv_rect.centerx
        zone_hw  = int(conv_rect.width * 0.22)
        pygame.draw.line(screen, C_BORDER,
                         (zone_cx - zone_hw, CONVEYOR_Y),
                         (zone_cx - zone_hw, CONVEYOR_Y + CONVEYOR_H), 1)
        pygame.draw.line(screen, C_BORDER,
                         (zone_cx + zone_hw, CONVEYOR_Y),
                         (zone_cx + zone_hw, CONVEYOR_Y + CONVEYOR_H), 1)
        # Glowing scan line when threat detected
        scan_col = C_HIGH if threat_level == THREAT_HIGH else \
                   C_LOW  if threat_level == THREAT_LOW  else C_BORDER
        pygame.draw.line(screen, scan_col,
                         (zone_cx, CONVEYOR_Y + 4),
                         (zone_cx, CONVEYOR_Y + CONVEYOR_H - 4), 2)

        manager.draw(screen, conv_rect, font_det)

        # Conveyor belt rail lines
        for off in [10, 20]:
            pygame.draw.line(screen, C_DARK_GRAY,
                             (CONVEYOR_X, CONVEYOR_Y + CONVEYOR_H - off),
                             (CONVEYOR_X + CONVEYOR_W, CONVEYOR_Y + CONVEYOR_H - off), 1)

        # Title bar above conveyor
        draw_text(screen, "BAGGAGE SCANNER  —  OPIXray Dangerous Object Detection",
                  font_title, C_GRAY, CONVEYOR_X + CONVEYOR_W // 2, 10, "center")

        # ── Confidence slider ─────────────────────────────────────────────
        pygame.draw.rect(screen, C_DARK_GRAY,
                         (SLIDER_X0, SLIDER_Y, SLIDER_W, SLIDER_H), border_radius=4)
        fill_w = int(conf_threshold * SLIDER_W)
        pygame.draw.rect(screen, C_BORDER,
                         (SLIDER_X0, SLIDER_Y, fill_w, SLIDER_H), border_radius=4)
        sx = conf_to_x(conf_threshold)
        pygame.draw.circle(screen, C_WHITE, (sx, SLIDER_Y + SLIDER_H // 2), 9)
        draw_text(screen, f"Confidence: {conf_threshold:.2f}",
                  font_small, C_GRAY, SLIDER_X0, SLIDER_Y - 18)
        draw_text(screen, "← drag to adjust →",
                  font_small, C_DARK_GRAY, SLIDER_X0 + SLIDER_W // 2, SLIDER_Y - 18, "center")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
