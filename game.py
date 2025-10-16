#!/usr/bin/env python
import sys
import math
import time
import random
import threading
import platform
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
import pygame
import pygame.gfxdraw




GAME_W, GAME_H = 1280, 720
FPS = 120

NUM_TARGETS = 9
LANES = 3
TARGET_SIZE = (120, 120)
PRICE_MIN, PRICE_MAX = 20, 150

MIRROR_CAMERA = True
MAX_STAGE = 1
ROUND_SECONDS = 300  


POINTER_GAIN = 1.85
DELTA_SMOOTH = 0.42
JITTER_DEADZONE_CAM = 2.0
MAX_DELTA_GAME = 120


FIST_ENTER_METRIC = 0.50
FIST_EXIT_METRIC  = 0.54
EXT_DY_RATIO = 0.18
CURL_DY_RATIO = 0.07
REQUIRED_CURLED_COUNT = 3
THUMB_EXT_RATIO = 0.27
FIST_STICK_S = 0.30
SHOT_COOLDOWN_S = 0.16


PREFERRED_INDEX = 0
PREFERRED_RESOLUTIONS = [(640,480), (960,540), (1280,720)]
FPS_LIMIT_CAMERA = 45


DUCK_SPRING_K = 40.0
DUCK_SPRING_D = 14.0


WHITE  = (245,245,245)
CYAN   = (30,210,230)
RED    = (230,60,60)
GOLD   = (255,208,0)
WOOD_DARK = (92,62,40)
WOOD_LIGHT= (141,100,70)




class GestureState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pointer = (GAME_W//2, GAME_H//2)
        self.trigger_pulled = False
        self.cam_ok = False
        self.hand_ok = False
        self.is_fist = False
        self.is_multi = False  

    def update(self, pointer, trigger_now, cam_ok, hand_ok, is_fist, is_multi):
        with self.lock:
            self.pointer = pointer
            if trigger_now:
                self.trigger_pulled = True
            self.cam_ok = cam_ok
            self.hand_ok = hand_ok
            self.is_fist = is_fist
            self.is_multi = is_multi

    def consume_trigger(self):
        with self.lock:
            t = self.trigger_pulled
            self.trigger_pulled = False
            return t

    def snapshot(self):
        with self.lock:
            return (self.pointer, self.cam_ok, self.hand_ok, self.is_fist, self.is_multi)

GSTATE = GestureState()




def _dist(a, b, w, h):
    dx = (a.x - b.x) * w
    dy = (a.y - b.y) * h
    return (dx*dx + dy*dy) ** 0.5

def _ref_len(land, w, h):
    return _dist(land[0], land[9], w, h) + 1e-6

_FINGER_TIP = { 'index':8, 'middle':12, 'ring':16, 'pinky':20 }
_FINGER_PIP = { 'index':6, 'middle':10, 'ring':14, 'pinky':18 }

def _is_extended_y(land, finger, h, ref):
    tip = land[_FINGER_TIP[finger]]
    pip = land[_FINGER_PIP[finger]]
    dy = (pip.y - tip.y) * h
    return dy > (EXT_DY_RATIO * ref)

def _is_curled_y(land, finger, h, ref):
    tip = land[_FINGER_TIP[finger]]
    pip = land[_FINGER_PIP[finger]]
    dy = (tip.y - pip.y) * h
    return dy > (CURL_DY_RATIO * ref)

def _palm_center(land):
    xs = [land[0].x, land[5].x, land[17].x]
    ys = [land[0].y, land[5].y, land[17].y]
    class P: pass
    p = P(); p.x = sum(xs)/3.0; p.y = sum(ys)/3.0
    return p

def _fist_metric(land, w, h):
    palm = _palm_center(land)
    ref = _ref_len(land, w, h)
    tips = [8,12,16,20]
    tip_ds = [_dist(land[i], palm, w, h)/ref for i in tips]
    thumb_d = _dist(land[4], palm, w, h)/ref
    return max(max(tip_ds), thumb_d * 0.9)

def _curled_count(land, w, h):
    ref = _ref_len(land, w, h)
    return sum(_is_curled_y(land, f, h, ref) for f in ('index','middle','ring','pinky'))

def _extended_count_four(land, w, h):
    ref = _ref_len(land, w, h)
    return sum(_is_extended_y(land, f, h, ref) for f in ('index','middle','ring','pinky'))

def _thumb_extended(land, w, h):
    palm = _palm_center(land)
    ref = _ref_len(land, w, h)
    return (_dist(land[4], palm, w, h) / ref) > THUMB_EXT_RATIO

def _extended_total(land, w, h):
    return _extended_count_four(land, w, h) + (1 if _thumb_extended(land, w, h) else 0)

def _index_extended(land, w, h):
    ref = _ref_len(land, w, h)
    return _is_extended_y(land, 'index', h, ref)

def _is_fist(land, w, h):
    return (_curled_count(land, w, h) >= REQUIRED_CURLED_COUNT) or (_fist_metric(land, w, h) < FIST_ENTER_METRIC)




def _auto_open_camera():
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif platform.system() == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    def try_open(index, backend, res_list):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None, None, None
        
        actual_w = actual_h = None
        ok = False
        for (rw, rh) in res_list:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            cap.set(cv2.CAP_PROP_FPS, FPS_LIMIT_CAMERA)
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_h, actual_w = frame.shape[:2]
                break
        if not ok:
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_h, actual_w = frame.shape[:2]
        if not ok:
            cap.release()
            return None, None, None
        return cap, actual_w, actual_h

    
    for b in backends:
        cap, w, h = try_open(PREFERRED_INDEX, b, PREFERRED_RESOLUTIONS)
        if cap: return cap, w, h
    
    for b in backends:
        for i in range(6):
            if i == PREFERRED_INDEX: continue
            cap, w, h = try_open(i, b, PREFERRED_RESOLUTIONS)
            if cap: return cap, w, h
    return None, None, None

def camera_worker():
    cap, cam_w, cam_h = _auto_open_camera()
    cam_ok = cap is not None
    if not cam_ok:
        print("❌ ERROR: Could not open camera.")
        GSTATE.update((GAME_W/2, GAME_H/2), False, False, False, False, False)
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    
    ema_px = GAME_W / 2
    ema_py = GAME_H / 2
    prev_cam_xy = None
    ema_dx = 0.0
    ema_dy = 0.0

    
    last_fist_ts = 0.0
    fist_active = False
    was_multi = False
    last_shot_ts = 0.0

    frame_interval = 1.0 / float(FPS_LIMIT_CAMERA)
    last_time = 0.0

    while True:
        now = time.time()
        if now - last_time < frame_interval:
            time.sleep(0.001)
            continue
        last_time = now

        ok, frame = cap.read()
        if not ok or frame is None:
            GSTATE.update((ema_px, ema_py), False, cam_ok, False, False, False)
            time.sleep(0.01)
            continue

        if MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        trigger_now = False
        hand_ok = bool(res.multi_hand_landmarks)
        is_fist = False
        is_multi = False
        idx_up = False

        if hand_ok:
            land = res.multi_hand_landmarks[0].landmark

            
            is_fist = _is_fist(land, w, h)
            idx_up = _index_extended(land, w, h)
            total_ext = _extended_total(land, w, h)
            is_multi = total_ext >= 2

            
            if is_fist:
                last_fist_ts = now
                fist_active = True
            else:
                if (now - last_fist_ts) > FIST_STICK_S:
                    fist_active = False

            
            idx_tip = land[8]
            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)

            if idx_up:
                if prev_cam_xy is None:
                    prev_cam_xy = (cx, cy)
                    ema_dx = ema_dy = 0.0
                else:
                    dx_cam = cx - prev_cam_xy[0]
                    dy_cam = cy - prev_cam_xy[1]
                    if abs(dx_cam) < JITTER_DEADZONE_CAM: dx_cam = 0
                    if abs(dy_cam) < JITTER_DEADZONE_CAM: dy_cam = 0

                    dx_game = dx_cam * POINTER_GAIN * (GAME_W / w)
                    dy_game = dy_cam * POINTER_GAIN * (GAME_H / h)
                    dx_game = max(-MAX_DELTA_GAME, min(MAX_DELTA_GAME, dx_game))
                    dy_game = max(-MAX_DELTA_GAME, min(MAX_DELTA_GAME, dy_game))

                    ema_dx = (1 - DELTA_SMOOTH) * ema_dx + DELTA_SMOOTH * dx_game
                    ema_dy = (1 - DELTA_SMOOTH) * ema_dy + DELTA_SMOOTH * dy_game

                    ema_px = max(0, min(GAME_W, ema_px + ema_dx))
                    ema_py = max(0, min(GAME_H, ema_py + ema_dy))
                    prev_cam_xy = (cx, cy)
            else:
                prev_cam_xy = None
                ema_dx *= 0.85
                ema_dy *= 0.85

            
            if is_multi and not was_multi and (now - last_shot_ts) >= SHOT_COOLDOWN_S:
                trigger_now = True
                last_shot_ts = now
            was_multi = is_multi

        else:
            prev_cam_xy = None
            ema_dx *= 0.85
            ema_dy *= 0.85
            if (now - last_fist_ts) > FIST_STICK_S:
                fist_active = False
            was_multi = False

        GSTATE.update((float(ema_px), float(ema_py)), trigger_now,
                      cam_ok=True, hand_ok=hand_ok, is_fist=fist_active, is_multi=is_multi)




def rounded_rect(surface, rect, color, radius=12, width=0):
    pygame.draw.rect(surface, color, rect, width, border_radius=radius)

def _lerp(a, b, t): return a + (b - a) * max(0.0, min(1.0, t))

def _lerp_color(c1, c2, t):
    return (int(_lerp(c1[0], c2[0], t)), int(_lerp(c1[1], c2[1], t)), int(_lerp(c1[2], c2[2], t)))

def draw_panel(surface, rect):
    
    rounded_rect(surface, rect.move(4,4), (0,0,0,140), radius=16)
    
    rounded_rect(surface, rect, (24,28,42,230), radius=16)
    
    inner = rect.inflate(-2, -2)
    rounded_rect(surface, inner, (255,255,255,18), radius=14, width=2)

    
    highlight = pygame.Surface((rect.w, rect.h//3), pygame.SRCALPHA)
    for y in range(highlight.get_height()):
        a = int(80 * (1 - y / max(1, highlight.get_height()-1)))
        pygame.draw.line(highlight, (255,255,255,a), (0,y), (rect.w, y))
    surface.blit(highlight, (rect.x, rect.y))

def draw_progress_bar(surface, rect, frac):
    
    rounded_rect(surface, rect, (30,34,48,220), radius=8)
    
    fill_w = int(rect.w * max(0.0, min(1.0, frac)))
    if fill_w > 0:
        fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.h)
        col = _lerp_color((60,200,120), (220,70,60), 1 - frac)
        rounded_rect(surface, fill_rect, (*col, 255), radius=8)
    
    rounded_rect(surface, rect, (255,255,255,24), radius=8, width=1)

def draw_badge(surface, x, y, text, ok, font):
    pad_x, pad_y = 10, 6
    txt = font.render(text, True, WHITE if ok else (200,205,215))
    w = txt.get_width() + pad_x*2
    h = txt.get_height() + pad_y*2
    rect = pygame.Rect(x, y, w, h)
    bg = (36,120,92,220) if ok else (60,64,78,200)
    rounded_rect(surface, rect, bg, radius=12)
    surface.blit(txt, (x + pad_x, y + pad_y))
    return w + 8  

def draw_button(surface, rect, text, font, hover=False):
    base = (50,56,84,230)
    hl   = (70,140,110,240)
    color = hl if hover else base
    rounded_rect(surface, rect, color, radius=12)
    label = font.render(text, True, WHITE)
    surface.blit(label, (rect.centerx - label.get_width()//2, rect.centery - label.get_height()//2))




class Particle:
    def __init__(self, pos, color):
        self.x, self.y = pos
        self.vx = random.uniform(-160,160)
        self.vy = random.uniform(-220,-40)
        self.life = random.uniform(0.35, 0.6)
        self.age = 0.0
        self.col = color

    def update(self, dt):
        self.age += dt
        self.vy += 650*dt
        self.x += self.vx*dt
        self.y += self.vy*dt

    def draw(self, surf):
        if self.age >= self.life: return
        a = max(0, 255 * (1 - self.age/self.life))
        pygame.gfxdraw.filled_circle(surf, int(self.x), int(self.y), 3, (*self.col, int(a)))
        pygame.gfxdraw.aacircle(surf, int(self.x), int(self.y), 3, self.col)




class Target:
    def __init__(self, lane_idx, center_y, price, moving, amp, speed, phase, sprite):
        self.lane_idx = lane_idx
        self.price = price
        self.alive = True
        self.flash_t = 0.0

        w, h = TARGET_SIZE
        self.rect = pygame.Rect(0, center_y - h//2, w, h)

        
        self.moving = moving
        self.amp = amp
        self.speed = speed
        self.phase = phase
        self.base_x = 0

        
        self.x = 0.0
        self.vx = 0.0

        self.lane_left = 0
        self.lane_right = GAME_W

        self.sprite = sprite

    def on_hit(self):
        self.flash_t = 0.18
        self.alive = False

    def assign_lane_bounds(self, lane_left, lane_right):
        self.lane_left = lane_left
        self.lane_right = lane_right

    def place_non_overlap(self, occupied_on_lane):
        w, h = self.rect.size
        for _ in range(200):
            x = random.randint(self.lane_left + 10, self.lane_right - w - 10)
            r = pygame.Rect(x, self.rect.y, w, h)
            if not any(r.colliderect(o) for o in occupied_on_lane):
                self.rect = r
                self.base_x = r.x
                self.x = float(self.base_x)
                return
        self.rect.x = self.lane_left + 10
        self.base_x = self.rect.x
        self.x = float(self.base_x)

    def update(self, dt, t_global):
        if not self.alive: return
        if self.flash_t > 0: self.flash_t -= dt

        target_x = self.base_x
        if self.moving:
            target_x = self.base_x + math.sin(self.phase + t_global * self.speed) * self.amp
            target_x = max(self.lane_left+5, min(self.lane_right - self.rect.w - 5, target_x))

        ax = DUCK_SPRING_K * (target_x - self.x) - DUCK_SPRING_D * self.vx
        self.vx += ax * dt
        self.x += self.vx * dt
        self.rect.x = int(self.x)

    def draw(self, surf):
        if not self.alive: return
        x, y, w, h = self.rect
        pygame.gfxdraw.filled_ellipse(surf, x+w//2, y+h-8, w//3, 8, (0,0,0,60))
        if self.flash_t > 0:
            glow = pygame.Surface((w, h), pygame.SRCALPHA)
            glow.fill((255,255,255, int(120*self.flash_t/0.18)))
            surf.blit(self.sprite, (x,y))
            surf.blit(glow, (x,y), special_flags=pygame.BLEND_ADD)
        else:
            surf.blit(self.sprite, (x,y))

    def draw_price_badge(self, surf, font):
        if not self.alive: return
        label = f"+{self.price}"
        tx = self.rect.centerx
        ty = self.rect.top - 10
        txt = font.render(label, True, WHITE)
        surf.blit(font.render(label, True, (0,0,0)), (tx - txt.get_width()//2 +1, ty - txt.get_height()//2 +1))
        surf.blit(txt, (tx - txt.get_width()//2, ty - txt.get_height()//2))




def stage_params(stage_idx):
    """
    Early stages easier:
      - Lower moving fraction / amplitudes / speeds at the start.
    """
    s = max(1, min(MAX_STAGE, stage_idx))
    move_frac = min(1.0, 0.08 + 0.14 * (s - 1))   
    amp = 46 + 12 * (s - 1)                       
    speed_mult = 0.85 + 0.20 * (s - 1)            
    return move_frac, amp, speed_mult

def compute_shelves():
    ys = [int(GAME_H * f) for f in np.linspace(0.32, 0.78, LANES)]
    shelves = []
    margin = 60
    for y in ys:
        rect = pygame.Rect(margin, y, GAME_W - 2*margin, 16)
        shelves.append(rect)
    return ys, shelves

def draw_shelves(screen, shelves):
    for shelf in shelves:
        r = shelf
        pygame.draw.rect(screen, (0,0,0,60), r.move(0,6), border_radius=8)
        pygame.draw.rect(screen, WOOD_DARK, r, border_radius=8)
        pygame.draw.rect(screen, WOOD_LIGHT, r.inflate(0,-6), border_radius=6)




def _fmt_time_left(sec):
    sec = max(0, int(sec))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def main():
    pygame.init()
    pygame.display.set_caption("Carnival Duck Shoot — 1:00 Round (Index-Move / 2+ Shoot)")
    screen = pygame.display.set_mode((GAME_W, GAME_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)
    small = pygame.font.SysFont("Arial", 20)
    tiny  = pygame.font.SysFont("Arial", 16)  

    
    try:
        duck_png = pygame.image.load("duck.png").convert_alpha()
    except Exception as e:
        duck_png = None
        print("Warning: duck.png not found. Using placeholder ducks.", e)

    if duck_png is not None:
        duck_sprite = pygame.transform.smoothscale(duck_png, TARGET_SIZE)
    else:
        duck_sprite = pygame.Surface(TARGET_SIZE, pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(duck_sprite, TARGET_SIZE[0]//2, TARGET_SIZE[1]//2, TARGET_SIZE[0]//2-6, (252,225,120))
        pygame.gfxdraw.aacircle(duck_sprite, TARGET_SIZE[0]//2, TARGET_SIZE[1]//2, TARGET_SIZE[0]//2-6, (185,165,70))

    
    bg = pygame.Surface((GAME_W, GAME_H))
    bg.fill((0,0,0))

    
    cam_thr = threading.Thread(target=camera_worker, daemon=True)
    cam_thr.start()

    
    lane_ys, shelves = compute_shelves()
    run_start_time = time.time()
    stage = 1
    stages_cleared = 0
    move_frac, move_amp, move_speed = stage_params(stage)
    score = 0
    help_visible = False  

    
    targets = []
    rng = random.Random(42)

    def init_targets():
        nonlocal targets
        targets = []
        moving_count = int(round(NUM_TARGETS * move_frac))
        flags = [True]*moving_count + [False]*(NUM_TARGETS - moving_count)
        rng.shuffle(flags)
        lane_to_rects = defaultdict(list)
        for i in range(NUM_TARGETS):
            lane_idx = i % LANES
            y_center = lane_ys[lane_idx] - 12 - TARGET_SIZE[1]//2
            t = Target(
                lane_idx=lane_idx,
                center_y=y_center,
                price=rng.randint(PRICE_MIN, PRICE_MAX),
                moving=flags[i],
                amp=move_amp,
                speed=move_speed,
                phase=rng.uniform(0, 2*math.pi),
                sprite=duck_sprite
            )
            t.assign_lane_bounds(shelves[lane_idx].left, shelves[lane_idx].right)
            t.place_non_overlap(lane_to_rects[lane_idx])
            lane_to_rects[lane_idx].append(t.rect.copy())
            targets.append(t)

    init_targets()

    particles = []
    running = True
    global_time = 0.0
    POINTER_RADIUS = 16  

    def reset_game():
        nonlocal run_start_time, stage, stages_cleared, move_frac, move_amp, move_speed, score, particles
        run_start_time = time.time()
        stage = 1
        stages_cleared = 0
        move_frac, move_amp, move_speed = stage_params(stage)
        score = 0
        particles.clear()
        init_targets()

    def show_end_screen(won=False):
        
        screen.fill((0,0,0))
        if won and duck_png is not None:
            
            img = duck_png
            iw, ih = img.get_width(), img.get_height()
            max_w = int(GAME_W * 0.92)
            max_h = int(GAME_H * 0.92)
            scale = min(max_w/iw, max_h/ih)
            scaled = pygame.transform.smoothscale(img, (int(iw*scale), int(ih*scale)))
            screen.blit(scaled, (GAME_W//2 - scaled.get_width()//2, GAME_H//2 - scaled.get_height()//2))
            
            veil = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            veil.fill((0,0,0,90))
            screen.blit(veil, (0,0))
        else:
            
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            overlay.fill((0,0,0,200))
            screen.blit(overlay, (0,0))

        title_text = "WINNER!" if won else "TIME UP!"
        title_col = GOLD if won else WHITE
        title = font.render(title_text, True, title_col)
        t1 = small.render(f"Stages cleared: {stages_cleared}", True, WHITE)
        t2 = small.render(f"Total time: {int(time.time() - run_start_time)} s", True, WHITE)
        t3 = small.render(f"Final score: {score}", True, WHITE)

        screen.blit(title, (GAME_W//2 - title.get_width()//2, GAME_H//2 - 120))
        screen.blit(t1, (GAME_W//2 - t1.get_width()//2, GAME_H//2 - 60))
        screen.blit(t2, (GAME_W//2 - t2.get_width()//2, GAME_H//2 - 30))
        screen.blit(t3, (GAME_W//2 - t3.get_width()//2, GAME_H//2))

        
        btn_w, btn_h, gap = 200, 52, 24
        restart_rect = pygame.Rect(0,0,btn_w,btn_h)
        quit_rect    = pygame.Rect(0,0,btn_w,btn_h)
        restart_rect.center = (GAME_W//2 - (btn_w//2 + gap//2), GAME_H//2 + 90)
        quit_rect.center    = (GAME_W//2 + (btn_w//2 + gap//2), GAME_H//2 + 90)

        waiting = True
        while waiting:
            mx, my = pygame.mouse.get_pos()
            draw_button(screen, restart_rect, "Restart (R)", small, restart_rect.collidepoint(mx,my))
            draw_button(screen, quit_rect,    "Quit (Q)",    small, quit_rect.collidepoint(mx,my))
            pygame.display.flip()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return "quit"
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    if e.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                        return "restart"
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if restart_rect.collidepoint(e.pos):
                        return "restart"
                    if quit_rect.collidepoint(e.pos):
                        return "quit"

    def next_stage():
        nonlocal stage, stages_cleared, move_frac, move_amp, move_speed
        stages_cleared += 1
        stage = min(MAX_STAGE, stage + 1)
        move_frac, move_amp, move_speed = stage_params(stage)
        init_targets()

    while running:
        dt = clock.tick(FPS) / 1000.0
        global_time += dt

        
        elapsed = time.time() - run_start_time
        time_left = ROUND_SECONDS - elapsed
        if time_left <= 0:
            action = show_end_screen(won=False)
            if action == "restart":
                reset_game()
                continue
            elif action == "quit":
                pygame.quit()
                sys.exit(0)
            else:
                break

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        
        (px, py), cam_ok, hand_ok, is_fist, is_multi = GSTATE.snapshot()
        ptx, pty = int(px), int(py)

        
        for t in targets:
            t.update(dt, global_time)

        
        if GSTATE.consume_trigger():
            hit_target = None
            for t in targets:
                if t.alive and t.rect.inflate(8, 8).collidepoint(ptx, pty):
                    hit_target = t
                    break

            if hit_target is not None:
                score += hit_target.price
                hit_target.on_hit()
                for _ in range(22):
                    particles.append(Particle(hit_target.rect.center, GOLD))
            else:
                for _ in range(10):
                    particles.append(Particle((ptx, pty), (235,235,235)))

            
            if all(not t.alive for t in targets):
                if stage == MAX_STAGE:
                    
                    stages_cleared += 1
                    action = show_end_screen(won=True)
                    if action == "restart":
                        reset_game()
                        continue
                    elif action == "quit":
                        pygame.quit()
                        sys.exit(0)
                    else:
                        running = False
                        break
                else:
                    next_stage()

        
        for p in particles[:]:
            p.update(dt)
            if p.age >= p.life:
                particles.remove(p)

        
        screen.blit(bg, (0,0))
        draw_shelves(screen, shelves)

        for t in targets:
            t.draw(screen)
            t.draw_price_badge(screen, small)

        for p in particles:
            p.draw(screen)

        
        hovering = any(t.alive and t.rect.inflate(8,8).collidepoint(ptx, pty) for t in targets)
        color = RED if hovering else CYAN
        for r in (POINTER_RADIUS, POINTER_RADIUS-1, POINTER_RADIUS-2, POINTER_RADIUS-3):
            if r > 0:
                pygame.gfxdraw.aacircle(screen, ptx, pty, r, color)
        pygame.gfxdraw.filled_circle(screen, ptx, pty, 4, color)

        
        hud_h = 96
        hud_rect = pygame.Rect(14, 12, 720, hud_h)
        draw_panel(screen, hud_rect)

        
        bar_rect = pygame.Rect(hud_rect.left + 14, hud_rect.top + 12, hud_rect.w - 28, 10)
        draw_progress_bar(screen, bar_rect, max(0.0, min(1.0, time_left / ROUND_SECONDS)))

        
        stage_txt = font.render(f"Stage: {stage}/{MAX_STAGE}", True, WHITE)
        screen.blit(stage_txt, (hud_rect.left + 14, hud_rect.top + 30))

        score_txt = tiny.render(f"Score: {score}", True, WHITE)
        timer_txt = tiny.render(f"Time Left: {_fmt_time_left(time_left)}", True, WHITE)

        
        chip_y = hud_rect.top + 30
        chip_pad = 10
        
        score_bg = pygame.Surface((score_txt.get_width()+20, score_txt.get_height()+12), pygame.SRCALPHA)
        rounded_rect(score_bg, score_bg.get_rect(), (35,40,58,220), radius=10)
        score_bg.blit(score_txt, (10, 6))
        screen.blit(score_bg, (hud_rect.right - score_bg.get_width() - 14, chip_y))

        
        timer_bg = pygame.Surface((timer_txt.get_width()+20, timer_txt.get_height()+12), pygame.SRCALPHA)
        rounded_rect(timer_bg, timer_bg.get_rect(), (35,40,58,220), radius=10)
        timer_bg.blit(timer_txt, (10, 6))
        screen.blit(timer_bg, (hud_rect.right - score_bg.get_width() - timer_bg.get_width() - 14 - chip_pad, chip_y))

        
        badges_y = hud_rect.top + 64
        x = hud_rect.left + 14
        x += draw_badge(screen, x, badges_y, "CAM",   cam_ok, tiny)
        x += draw_badge(screen, x, badges_y, "HAND",  hand_ok, tiny)
        x += draw_badge(screen, x, badges_y, "FIST",  is_fist, tiny)
        x += draw_badge(screen, x, badges_y, "2+F",   is_multi, tiny)
        

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()

