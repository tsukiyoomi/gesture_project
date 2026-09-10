# gesture_recognition.py
"""
ФИНАЛЬНОЕ ПРИЛОЖЕНИЕ: УПРАВЛЕНИЕ КОМПЬЮТЕРОМ С ПОМОЩЬЮ ЖЕСТОВ
- Перемещение курсора — открытая ладонь (жест 5) — стабильный!
- Левый клик — указательный палец (жест 1)
- Коэффициент усиления курсора: 2.5x (чтобы доставать до краёв)
"""

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from torchvision import transforms
import time
import numpy as np
import pyautogui
import os
import urllib.request
from torchvision import models

# ========== 1. НАСТРОЙКИ ==========
MODEL_PATH = "models/gesture_cnn_best_2.pth"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.8  # порог уверенности (80%)
SHOW_LANDMARKS = True
SMOOTHING_FACTOR = 0.5

# ЗАДЕРЖКА МЕЖДУ ЖЕСТАМИ (секунды)
GESTURE_COOLDOWN = 1.5

# СКРОЛЛИНГ
SCROLL_AMOUNT = 200
SCROLL_REPEAT = 5

# КУРСОР: коэффициент усиления (чем больше, тем быстрее курсор)
MOUSE_SENSITIVITY = 2.5

# КАРТА ЖЕСТОВ
GESTURE_NAMES = {
    0: "fist",
    1: "index_up",  # указательный палец → ЛЕВЫЙ КЛИК
    2: "two_fingers",  # два пальца → ПРАВЫЙ КЛИК
    3: "three_fingers",  # три пальца → СКРОЛЛ ВВЕРХ
    4: "four_fingers",  # четыре пальца → СКРОЛЛ ВНИЗ
    5: "open_palm",  # открытая ладонь → ПЕРЕМЕЩЕНИЕ КУРСОРА
    6: "thumbs_up",
    7: "finger_gun",
    8: "rock",
    9: "ok"
}

# Соответствие жестов командам
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

print("=" * 60)
print("ЗАПУСК СИСТЕМЫ УПРАВЛЕНИЯ КОМПЬЮТЕРОМ С ПОМОЩЬЮ ЖЕСТОВ")
print("=" * 60)
print("Новая карта жестов:")
print("   • Открытая ладонь (5) → перемещение курсора")
print("   • Указательный палец (1) → левый клик")
print("   • Два пальца (2) → правый клик")
print("   • Три пальца (3) → скролл вверх")
print("   • Четыре пальца (4) → скролл вниз")
print("=" * 60)

pyautogui.FAILSAFE = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


class GestureRecognizer:
    """Класс для распознавания жестов и управления компьютером (сохранённая логика)"""

    def __init__(self, model_path=MODEL_PATH, camera_index=CAMERA_INDEX):
        self.model_path = model_path
        self.camera_index = camera_index

        # Параметры управления
        self.smoothing_factor = SMOOTHING_FACTOR
        self.cooldown = GESTURE_COOLDOWN
        self.sensitivity = MOUSE_SENSITIVITY
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        # Загрузка модели
        self.model = self._load_model()

        # Инициализация MediaPipe
        self.detector = self._init_mediapipe()

        # Состояние
        self.prev_x, self.prev_y = pyautogui.position()
        self.last_action_time = 0
        self.is_running = False
        self.cap = None
        self.frame_count = 0

        # Размер экрана
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Размер экрана: {self.screen_width}x{self.screen_height}")
        print(f"Чувствительность курсора: {self.sensitivity}x")
        print(f"Задержка между жестами: {self.cooldown} сек")

        # Трансформации для CNN
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Загрузка обученной модели CNN"""
        print(f"\nЗагрузка модели из {self.model_path}...")
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            print("Модель успешно загружена!")
        else:
            print(f"ВНИМАНИЕ: Модель не найдена по пути {self.model_path}")

        model = model.to(device)
        model.eval()
        return model

    def _init_mediapipe(self):
        """Инициализация MediaPipe"""
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Загрузка модели MediaPipe...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Модель MediaPipe загружена!")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = HandLandmarker.create_from_options(options)
        print("MediaPipe инициализирован!")
        return detector

    def _smooth_move(self, finger_x, finger_y):
        """
        Перемещение курсора с усилением и сглаживанием
        finger_x, finger_y — координаты пальца в кадре (0..1)
        """
        # Применяем инверсию по оси X, если включена
        if self.invert_x:
            finger_x = 1.0 - finger_x

        # Усиление: движение пальца на 50% кадра = курсор на весь экран
        target_x = int((finger_x - 0.5) * self.screen_width * self.sensitivity + self.screen_width / 2)
        target_y = int((finger_y - 0.5) * self.screen_height * self.sensitivity + self.screen_height / 2)

        # Ограничение границами экрана
        target_x = max(0, min(target_x, self.screen_width - 1))
        target_y = max(0, min(target_y, self.screen_height - 1))

        # Экспоненциальное сглаживание
        curr_x = int(self.prev_x * self.smoothing_factor + target_x * (1 - self.smoothing_factor))
        curr_y = int(self.prev_y * self.smoothing_factor + target_y * (1 - self.smoothing_factor))

        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y

    def _can_execute(self, gesture_id):
        """Проверка задержки (кроме курсора)"""
        if gesture_id == 5:  # перемещение курсора — без задержки
            return True
        current_time = time.time()
        return (current_time - self.last_action_time) >= self.cooldown

    def _execute_command(self, gesture_id, confidence):
        """Выполнение команды по жесту"""
        command = GESTURE_COMMANDS.get(gesture_id, "none")

        if not self._can_execute(gesture_id):
            return "cooldown"

        # Для команд, кроме курсора, обновляем время задержки
        if gesture_id != 5:
            self.last_action_time = time.time()

        if command == "left_click":
            pyautogui.click()
            print(f"🖱️ Левый клик (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "right_click":
            pyautogui.rightClick()
            print(f"🖱️ Правый клик (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "double_click":
            pyautogui.doubleClick()
            print(f"🖱️ Двойной клик (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "scroll_up":
            for _ in range(SCROLL_REPEAT):
                pyautogui.scroll(SCROLL_AMOUNT)
            print(f"📜 Скролл вверх (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "scroll_down":
            for _ in range(SCROLL_REPEAT):
                pyautogui.scroll(-SCROLL_AMOUNT)
            print(f"📜 Скролл вниз (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "back":
            pyautogui.hotkey('alt', 'left')
            print(f"⬅️ Назад (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "play_pause":
            pyautogui.press('playpause')
            print(f"⏯️ Play/Pause (жест {gesture_id}, {confidence:.1f}%)")

        elif command == "enter":
            pyautogui.press('enter')
            print(f"✅ Enter (жест {gesture_id}, {confidence:.1f}%)")

        return command

    def start(self):
        """Запуск видеопотока"""
        self.is_running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_count = 0

        if not self.cap.isOpened():
            raise RuntimeError(f"Ошибка: не удалось открыть камеру с индексом {self.camera_index}")

        print("Камера запущена")

    def process_frame(self, frame):
        """
        Обработка одного кадра (основная логика распознавания)
        Возвращает:
            - frame_with_marks: кадр с нарисованными точками
            - gesture_id: идентификатор жеста
            - confidence: уверенность
            - gesture_name: название жеста
            - command: выполненная команда
            - hand_detected: обнаружена ли рука
        """
        h, w, _ = frame.shape
        display_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        gesture_id = -1
        confidence = 0.0
        gesture_name = "none"
        hand_detected = False
        palm_center = None
        command = "none"

        if detection_result.hand_landmarks:
            hand_detected = True
            hand = detection_result.hand_landmarks[0]

            if len(hand) > 0:
                palm_center = (hand[0].x, hand[0].y)

            # Выделение ROI
            min_x = min([lm.x for lm in hand])
            max_x = max([lm.x for lm in hand])
            min_y = min([lm.y for lm in hand])
            max_y = max([lm.y for lm in hand])

            padding = 0.2
            roi_x1 = max(0, int((min_x - padding) * w))
            roi_y1 = max(0, int((min_y - padding) * h))
            roi_x2 = min(w, int((max_x + padding) * w))
            roi_y2 = min(h, int((max_y + padding) * h))

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform(roi_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        gesture_id = predicted.item()
                        confidence = confidence.item() * 100
                        gesture_name = GESTURE_NAMES.get(gesture_id, "unknown")

            # ===== ПЕРЕМЕЩЕНИЕ КУРСОРА (жест 5 — открытая ладонь) =====
            if gesture_id == 5 and palm_center is not None and confidence >= self.confidence_threshold * 100:
                self._smooth_move(palm_center[0], palm_center[1])
                command = "move_mouse"

            # ===== ВЫПОЛНЕНИЕ КОМАНД (остальные жесты) =====
            elif gesture_id != 5 and confidence >= self.confidence_threshold * 100:
                command = self._execute_command(gesture_id, confidence)

            # Отрисовка ключевых точек
            if SHOW_LANDMARKS:
                for landmark in hand:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)

                if palm_center:
                    pcx, pcy = int(palm_center[0] * w), int(palm_center[1] * h)
                    cv2.circle(display_frame, (pcx, pcy), 8, (0, 0, 255), -1)

                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        return {
            'frame_with_marks': display_frame,
            'gesture_id': gesture_id,
            'confidence': confidence,
            'gesture_name': gesture_name,
            'command': command,
            'hand_detected': hand_detected,
            'palm_center': palm_center
        }

    def get_frame(self):
        """Получение следующего кадра с камеры"""
        if not self.is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def stop(self):
        """Остановка видеопотока"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.close()
        cv2.destroyAllWindows()
        print("Система остановлена")

    def update_settings(self, sensitivity=None, cooldown=None, smoothing=None, threshold=None, invert_x=None):
        """Обновление настроек"""
        if sensitivity is not None:
            self.sensitivity = sensitivity
            print(f"Чувствительность обновлена: {sensitivity}x")
        if cooldown is not None:
            self.cooldown = cooldown
            print(f"Задержка обновлена: {cooldown} сек")
        if smoothing is not None:
            self.smoothing_factor = smoothing
        if threshold is not None:
            self.confidence_threshold = threshold
            print(f"Порог уверенности обновлён: {threshold * 100}%")
        if invert_x is not None:
            self.invert_x = invert_x
            # Строку с print удалили — инверсия не выводится в консоль

# Для тестирования модуля отдельно
if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.start()

    print("\n" + "=" * 60)
    print("СИСТЕМА ЗАПУЩЕНА!")
    print(f"Порог уверенности: {CONFIDENCE_THRESHOLD * 100}%")
    print("Нажмите ESC для выхода")
    print("=" * 60 + "\n")

    fps = 0
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        frame = recognizer.get_frame()
        if frame is None:
            break

        result = recognizer.process_frame(frame)

        # Вывод информации на кадр
        display = result['frame_with_marks']
        h, w, _ = display.shape

        cv2.rectangle(display, (5, 5), (500, 130), (0, 0, 0), -1)

        if result['hand_detected'] and result['gesture_id'] != -1:
            cv2.putText(display, f"Gesture: {result['gesture_name']}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {result['confidence']:.1f}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Command: {result['command']}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display, "No hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(display, f"FPS: {fps}", (w - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Press ESC to exit", (w - 200, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.imshow("Gesture Control", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    recognizer.stop()