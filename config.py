# config.py
import torch

# ========== НАСТРОЙКИ МОДЕЛИ ==========
MODEL_PATH = "models/gesture_cnn_best_2.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== НАСТРОЙКИ КАМЕРЫ ==========
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ========== НАСТРОЙКИ РАСПОЗНАВАНИЯ ==========
CONFIDENCE_THRESHOLD = 0.8      # порог уверенности (80%)
GESTURE_COOLDOWN = 1.5          # задержка между жестами (сек)

# ========== НАСТРОЙКИ УПРАВЛЕНИЯ ==========
MOUSE_SENSITIVITY = 1.5         # чувствительность курсора
SMOOTHING_FACTOR = 0.5          # сглаживание
SCROLL_AMOUNT = 200             # количество щелчков при скролле
SCROLL_REPEAT = 5               # повторы скролла

# ========== НАСТРОЙКИ GUI ==========
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# ========== КАРТА ЖЕСТОВ ==========
GESTURE_NAMES = {
    0: "fist",
    1: "index_up",
    2: "two_fingers",
    3: "three_fingers",
    4: "four_fingers",
    5: "open_palm",
    6: "thumbs_up",
    7: "finger_gun",
    8: "rock",
    9: "ok"
}

GESTURE_COMMANDS = {
    0: "none",
    1: "left_click",
    2: "right_click",
    3: "scroll_up",
    4: "scroll_down",
    5: "move_mouse",
    6: "double_click",
    7: "back",
    8: "play_pause",
    9: "enter"
}