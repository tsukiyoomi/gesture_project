# gui_app.py
import customtkinter as ctk
import threading
import cv2
from PIL import Image, ImageTk
import time
import json
import os

from config import *
from gesture_recognition import GestureRecognizer

# Словарь для перевода названий жестов на русский
GESTURE_NAMES_RU = {
    "fist": "Кулак",
    "index_up": "Указательный палец",
    "two_fingers": "Два пальца",
    "three_fingers": "Три пальца",
    "four_fingers": "Четыре пальца",
    "open_palm": "Открытая ладонь",
    "thumbs_up": "Большой палец вверх",
    "finger_gun": "Пистолет",
    "rock": "Коза",
    "ok": "OK",
    "none": "Нет",
    "unknown": "Неизвестно"
}

# Словарь для перевода команд на русский
COMMANDS_RU = {
    "left_click": "Левый клик",
    "right_click": "Правый клик",
    "double_click": "Двойной клик",
    "scroll_up": "Скролл вверх",
    "scroll_down": "Скролл вниз",
    "move_mouse": "Движение мыши",
    "back": "Назад",
    "play_pause": "Play/Pause",
    "enter": "Enter",
    "none": "Нет",
    "cooldown": "Ожидание",
    "blocked": "Заблокировано"
}

SETTINGS_FILE = "settings.json"


class GestureGUI:
    def __init__(self):
        # Настройка окна
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Управление жестами v2.0")
        self.root.geometry("1150x750")
        self.root.minsize(950, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Инициализация распознавателя
        self.recognizer = GestureRecognizer()

        # Переменные состояния
        self.is_running = False
        self.video_thread = None
        self.current_frame = None
        self.invert_x = False
        self.gesture_counts = {i: 0 for i in range(10)}  # счётчик жестов

        # Загрузка настроек
        self.load_settings()

        # Создание интерфейса
        self._create_widgets()

        # Виджет для видео
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # FPS
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

        # Статус
        self.update_status("Готов")

    def _create_widgets(self):
        """Создание элементов интерфейса"""
        # Основной контейнер
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Левая панель — видео
        self.video_frame = ctk.CTkFrame(self.main_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        # Правая панель — вкладки
        self.right_panel = ctk.CTkFrame(self.main_frame, width=380)
        self.right_panel.pack(side="right", fill="both", padx=10, pady=10)

        # Создание вкладок
        self.tabview = ctk.CTkTabview(self.right_panel)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        # Добавление вкладок
        self.tabview.add("Основное")
        self.tabview.add("Настройки")
        self.tabview.add("Справка")

        # ========== ВКЛАДКА "ОСНОВНОЕ" ==========
        self.main_tab = self.tabview.tab("Основное")

        # Карточка жеста (тёмная, без зелёного)
        self.gesture_card = ctk.CTkFrame(self.main_tab, corner_radius=10, border_width=1, border_color="#3a3a3a")
        self.gesture_card.pack(fill="x", padx=15, pady=10)

        self.gesture_label = ctk.CTkLabel(
            self.gesture_card,
            text="ТЕКУЩИЙ ЖЕСТ",
            font=("Arial", 12, "bold"),
            text_color="#a0a0a0"
        )
        self.gesture_label.pack(pady=(10, 0))

        ctk.CTkFrame(self.gesture_card, height=1, fg_color="#3a3a3a").pack(fill="x", padx=20, pady=5)

        self.gesture_value = ctk.CTkLabel(
            self.gesture_card,
            text="Нет",
            font=("Arial", 26, "bold"),
            text_color="#d0d0d0",
            wraplength=300
        )
        self.gesture_value.pack(pady=10)

        # Карточка уверенности (оставляем, респект!)
        self.confidence_card = ctk.CTkFrame(self.main_tab, corner_radius=10, border_width=1, border_color="#3a5a8a")
        self.confidence_card.pack(fill="x", padx=15, pady=5)

        self.confidence_label = ctk.CTkLabel(
            self.confidence_card,
            text="УВЕРЕННОСТЬ",
            font=("Arial", 11, "bold"),
            text_color="#5a8aba"
        )
        self.confidence_label.pack(pady=(8, 0))

        self.confidence_value = ctk.CTkLabel(
            self.confidence_card,
            text="0%",
            font=("Arial", 20, "bold"),
            text_color="#f0c050"
        )
        self.confidence_value.pack(pady=5)

        self.confidence_progress = ctk.CTkProgressBar(self.confidence_card, width=200, height=8)
        self.confidence_progress.set(0)
        self.confidence_progress.pack(pady=(0, 10))

        # Карточка команды
        self.command_card = ctk.CTkFrame(self.main_tab, corner_radius=10, border_width=1, border_color="#8a5a3a")
        self.command_card.pack(fill="x", padx=15, pady=5)

        self.command_title = ctk.CTkLabel(
            self.command_card,
            text="ВЫПОЛНЯЕМАЯ КОМАНДА",
            font=("Arial", 11, "bold"),
            text_color="#c08a5a"
        )
        self.command_title.pack(pady=(8, 0))

        self.command_value = ctk.CTkLabel(
            self.command_card,
            text="Нет",
            font=("Arial", 14, "bold"),
            text_color="#e0a060"
        )
        self.command_value.pack(pady=10)

        # Статус
        self.status_frame = ctk.CTkFrame(self.main_tab, corner_radius=10, border_width=1, border_color="#555555")
        self.status_frame.pack(fill="x", padx=15, pady=10)

        self.status_text = ctk.CTkLabel(
            self.status_frame,
            text="Статус: Готов",
            font=("Arial", 13),
            text_color="green"
        )
        self.status_text.pack(pady=8)

        # Кнопки управления
        self.button_frame = ctk.CTkFrame(self.main_tab, fg_color="transparent")
        self.button_frame.pack(fill="x", padx=15, pady=15)

        self.start_btn = ctk.CTkButton(
            self.button_frame,
            text="Старт",
            command=self.start_recognition,
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="#2a6d4c",
            hover_color="#1e4f38",
            state="normal"
        )
        self.start_btn.pack(side="left", expand=True, padx=5)

        self.stop_btn = ctk.CTkButton(
            self.button_frame,
            text="Стоп",
            command=self.stop_recognition,
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="#8a3a3a",
            hover_color="#6e2e2e",
            state="disabled"
        )
        self.stop_btn.pack(side="left", expand=True, padx=5)

        # Статистика
        self.stats_frame = ctk.CTkFrame(self.main_tab, corner_radius=10, border_width=1, border_color="#2a2a2a")
        self.stats_frame.pack(fill="x", padx=15, pady=10)

        self.fps_label = ctk.CTkLabel(
            self.stats_frame,
            text="FPS: 0",
            font=("Arial", 12)
        )
        self.fps_label.pack(pady=3)

        self.frames_label = ctk.CTkLabel(
            self.stats_frame,
            text="Обработано кадров: 0",
            font=("Arial", 12)
        )
        self.frames_label.pack(pady=3)

        self.gesture_counts_label = ctk.CTkLabel(
            self.stats_frame,
            text="Жестов за сессию: 0",
            font=("Arial", 12)
        )
        self.gesture_counts_label.pack(pady=3)

        # ========== ВКЛАДКА "НАСТРОЙКИ" ==========
        self.settings_tab = self.tabview.tab("Настройки")

        self.settings_title = ctk.CTkLabel(
            self.settings_tab,
            text="НАСТРОЙКИ УПРАВЛЕНИЯ",
            font=("Arial", 16, "bold")
        )
        self.settings_title.pack(pady=20)

        # Чувствительность мыши
        self._add_slider(self.settings_tab, "Чувствительность мыши", 1.0, 3.0, self.sensitivity_val,
                         self.on_sensitivity_change)

        # Задержка между жестами
        self._add_slider(self.settings_tab, "Задержка между жестами (сек)", 0.3, 2.0, self.cooldown_val,
                         self.on_cooldown_change)

        # Порог уверенности
        self._add_slider(self.settings_tab, "Порог уверенности (%)", 50, 95, self.threshold_val,
                         self.on_threshold_change)

        # Инверсия движения мыши по оси X
        self.invert_frame = ctk.CTkFrame(self.settings_tab)
        self.invert_frame.pack(fill="x", padx=15, pady=10)

        self.invert_label = ctk.CTkLabel(
            self.invert_frame,
            text="Инверсия движения мыши",
            font=("Arial", 13)
        )
        self.invert_label.pack(pady=(10, 0))

        self.invert_desc = ctk.CTkLabel(
            self.invert_frame,
            text="Исправить зеркальное отображение камеры",
            font=("Arial", 10),
            text_color="gray"
        )
        self.invert_desc.pack()

        self.invert_switch = ctk.CTkSwitch(
            self.invert_frame,
            text="Инвертировать по оси X",
            command=self.on_invert_toggle
        )
        self.invert_switch.pack(pady=10)
        if self.invert_x:
            self.invert_switch.select()
        else:
            self.invert_switch.deselect()

        # Выбор камеры
        self.camera_frame = ctk.CTkFrame(self.settings_tab)
        self.camera_frame.pack(fill="x", padx=15, pady=10)

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="Выбор камеры",
            font=("Arial", 13)
        )
        self.camera_label.pack(pady=(10, 5))

        self.camera_combo = ctk.CTkComboBox(
            self.camera_frame,
            values=["0", "1", "2"],
            command=self.on_camera_change,
            width=150
        )
        self.camera_combo.set(str(self.camera_index))
        self.camera_combo.pack(pady=5)

        # Кнопка сброса настроек
        self.reset_btn = ctk.CTkButton(
            self.settings_tab,
            text="Сбросить настройки по умолчанию",
            command=self.reset_settings,
            height=35,
            fg_color="#3a3a3a",
            hover_color="#2a2a2a"
        )
        self.reset_btn.pack(pady=20)

        # ========== ВКЛАДКА "СПРАВКА" ==========
        self.help_tab = self.tabview.tab("Справка")

        help_text = """
УПРАВЛЕНИЕ ЖЕСТАМИ

КАРТА ЖЕСТОВ

Открытая ладонь (5) → Перемещение курсора мыши
Кулак (0) → Левый клик
Указательный палец (1) → Левый клик (альтернатива)
Два пальца (2) → Правый клик
Три пальца (3) → Скролл вверх
Четыре пальца (4) → Скролл вниз
Большой палец вверх (6) → Двойной клик
Пистолет (7) → Назад (Alt+Left)
Коза (8) → Play/Pause
OK (9) → Enter

<br><br>
КАК ПРАВИЛЬНО ПОКАЗЫВАТЬ ЖЕСТЫ

Условия для всех жестов:
  • Рука в кадре целиком, ладонью к камере
  • Пальцы разомкнуты, видны промежутки
  • Достаточное освещение

<br><br>
ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ

1. Нажмите "Старт" для запуска камеры
2. Покажите руку в кадре
3. Выполните нужный жест
4. Система выполнит команду

<br><br>
НАСТРОЙКИ

• Чувствительность мыши — скорость курсора
• Задержка между жестами — пауза после команды
• Порог уверенности — минимальный % для срабатывания
• Инверсия движения — исправляет зеркальное отображение

<br><br>
УПРАВЛЕНИЕ ПРОГРАММОЙ

Старт / Стоп — запуск и остановка
ESC — выход из программы

<br><br>
ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ

Точность модели: 98.9%
Частота кадров: 20-30 FPS
Распознаваемых жестов: 10
"""

        self.help_textbox = ctk.CTkTextbox(self.help_tab, wrap="word", font=("Arial", 13))
        self.help_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        # Вставка с заменой <br><br> на пустые строки
        formatted_help = help_text.replace("<br><br>", "\n\n")
        self.help_textbox.insert("0.0", formatted_help)
        self.help_textbox.configure(state="disabled")

    def _add_slider(self, parent, label, min_val, max_val, default, callback):
        """Добавление ползунка настроек"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(frame, text=label, font=("Arial", 13)).pack()
        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, command=callback)
        slider.set(default)
        slider.pack(fill="x", padx=10, pady=5)

        value_label = ctk.CTkLabel(frame, text=self._format_value(label, default), font=("Arial", 11))
        value_label.pack(pady=(0, 5))

        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('%',
                                                                                                         '').lower()
        setattr(self, f"{safe_label}_slider", slider)
        setattr(self, f"{safe_label}_value", value_label)

    def _format_value(self, label, value):
        """Форматирование значения для отображения"""
        if "процент" in label or "%" in label:
            return f"{int(value)}%"
        elif "сек" in label:
            return f"{value:.1f} сек"
        else:
            return f"{value:.1f}"

    def update_status(self, message, is_error=False):
        """Обновление статуса"""
        if message == "Готов":
            self.status_text.configure(text=f"Статус: {message}", text_color="green")
        elif message == "Работает...":
            self.status_text.configure(text=f"Статус: {message}", text_color="#2a6d4c")
        elif message == "Остановлено":
            self.status_text.configure(text=f"Статус: {message}", text_color="red")
        else:
            color = "red" if is_error else "yellow"
            self.status_text.configure(text=f"Статус: {message}", text_color=color)

    def load_settings(self):
        """Загрузка настроек из файла"""
        default_settings = {
            "sensitivity": MOUSE_SENSITIVITY,
            "cooldown": GESTURE_COOLDOWN,
            "threshold": CONFIDENCE_THRESHOLD * 100,
            "invert_x": False,
            "camera_index": CAMERA_INDEX
        }
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                self.sensitivity_val = saved.get("sensitivity", default_settings["sensitivity"])
                self.cooldown_val = saved.get("cooldown", default_settings["cooldown"])
                self.threshold_val = saved.get("threshold", default_settings["threshold"])
                self.invert_x = saved.get("invert_x", default_settings["invert_x"])
                self.camera_index = saved.get("camera_index", default_settings["camera_index"])
        except FileNotFoundError:
            self.sensitivity_val = default_settings["sensitivity"]
            self.cooldown_val = default_settings["cooldown"]
            self.threshold_val = default_settings["threshold"]
            self.invert_x = default_settings["invert_x"]
            self.camera_index = default_settings["camera_index"]

        # Применяем к recognizer
        self.recognizer.update_settings(
            sensitivity=self.sensitivity_val,
            cooldown=self.cooldown_val,
            threshold=self.threshold_val / 100,
            invert_x=self.invert_x
        )

    def save_settings(self):
        """Сохранение настроек в файл"""
        settings = {
            "sensitivity": self.sensitivity_val,
            "cooldown": self.cooldown_val,
            "threshold": self.threshold_val,
            "invert_x": self.invert_x,
            "camera_index": self.camera_index
        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)

    def reset_settings(self):
        """Сброс настроек по умолчанию"""
        self.sensitivity_val = MOUSE_SENSITIVITY
        self.cooldown_val = GESTURE_COOLDOWN
        self.threshold_val = CONFIDENCE_THRESHOLD * 100
        self.invert_x = False
        self.camera_index = CAMERA_INDEX

        # Обновляем UI
        self.чувствительность_мыши_slider.set(self.sensitivity_val)
        self.чувствительность_мыши_value.configure(text=f"{self.sensitivity_val:.1f}")
        self.задержка_между_жестами_сек_slider.set(self.cooldown_val)
        self.задержка_между_жестами_сек_value.configure(text=f"{self.cooldown_val:.1f} сек")
        self.порог_уверенности_slider.set(self.threshold_val)
        self.порог_уверенности_value.configure(text=f"{int(self.threshold_val)}%")

        self.invert_switch.deselect()
        self.camera_combo.set(str(self.camera_index))

        # Применяем к recognizer
        self.recognizer.update_settings(
            sensitivity=self.sensitivity_val,
            cooldown=self.cooldown_val,
            threshold=self.threshold_val / 100,
            invert_x=self.invert_x
        )

        self.update_status("Настройки сброшены", is_error=False)
        print("Настройки сброшены к значениям по умолчанию")

    def on_sensitivity_change(self, value):
        """Изменение чувствительности"""
        self.sensitivity_val = float(value)
        self.recognizer.update_settings(sensitivity=self.sensitivity_val)
        self.чувствительность_мыши_value.configure(text=f"{self.sensitivity_val:.1f}")
        self.save_settings()

    def on_cooldown_change(self, value):
        """Изменение задержки между жестами"""
        self.cooldown_val = float(value)
        self.recognizer.update_settings(cooldown=self.cooldown_val)
        self.задержка_между_жестами_сек_value.configure(text=f"{self.cooldown_val:.1f} сек")
        self.save_settings()

    def on_threshold_change(self, value):
        """Изменение порога уверенности"""
        self.threshold_val = float(value)
        self.recognizer.update_settings(threshold=self.threshold_val / 100)
        self.порог_уверенности_value.configure(text=f"{int(self.threshold_val)}%")
        self.save_settings()

    def on_invert_toggle(self):
        """Включение/выключение инверсии движения по X"""
        self.invert_x = self.invert_switch.get()
        self.recognizer.update_settings(invert_x=self.invert_x)
        self.save_settings()

    def on_camera_change(self, value):
        """Изменение индекса камеры"""
        self.camera_index = int(value)
        self.save_settings()
        self.update_status("Камера изменится при следующем запуске", is_error=False)

    def _translate_gesture(self, gesture_name):
        """Перевод названия жеста на русский"""
        return GESTURE_NAMES_RU.get(gesture_name, gesture_name)

    def _translate_command(self, command):
        """Перевод команды на русский"""
        return COMMANDS_RU.get(command, command)

    def start_recognition(self):
        """Запуск распознавания"""
        try:
            # Обновляем индекс камеры в recognizer
            self.recognizer.camera_index = self.camera_index
            self.recognizer.start()
            self.is_running = True

            # Блокируем кнопку Старт (визуально)
            self.start_btn.configure(state="disabled", fg_color="#1a4a3a", text_color="#6a6a6a")
            self.stop_btn.configure(state="normal", fg_color="#8a3a3a")

            self.update_status("Работает...")

            # Сброс счётчика жестов
            self.gesture_counts = {i: 0 for i in range(10)}

            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()

        except Exception as e:
            self.update_status(f"Ошибка: {e}", is_error=True)

    def stop_recognition(self):
        """Остановка распознавания"""
        self.is_running = False

        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.stop()

        # Блокируем кнопку Стоп, разблокируем Старт
        self.start_btn.configure(state="normal", fg_color="#2a6d4c", text_color="white")
        self.stop_btn.configure(state="disabled", fg_color="#5a2a2a", text_color="#a0a0a0")

        self.update_status("Остановлено")

        # Очистка видео и информации
        self.video_label.configure(image="")
        self.video_label.image = None
        self.gesture_value.configure(text="Нет")
        self.confidence_value.configure(text="0%")
        self.confidence_progress.set(0)
        self.command_value.configure(text="Нет")

    def _video_loop(self):
        """Основной цикл обработки видео"""
        frame_count = 0
        total_gestures = 0

        while self.is_running:
            frame = self.recognizer.get_frame()
            if frame is None:
                break

            # Передаём флаг инверсии в recognizer через настройки
            self.recognizer.update_settings(invert_x=self.invert_x)

            result = self.recognizer.process_frame(frame)

            # Обновление счётчика жестов
            if result['hand_detected'] and result['gesture_id'] != -1 and result['gesture_id'] in self.gesture_counts:
                self.gesture_counts[result['gesture_id']] += 1
                total_gestures = sum(self.gesture_counts.values())
                self.gesture_counts_label.configure(text=f"Жестов за сессию: {total_gestures}")

            # Обновление GUI
            if result['hand_detected'] and result['gesture_id'] != -1:
                gesture_name_ru = self._translate_gesture(result['gesture_name'])
                command_ru = self._translate_command(result['command'])

                # Без капса для движения мыши, с капсом для остальных
                if result['gesture_id'] == 5:
                    display_name = gesture_name_ru
                else:
                    display_name = gesture_name_ru.upper()

                self.gesture_value.configure(text=display_name)
                self.confidence_value.configure(text=f"{result['confidence']:.1f}%")
                self.confidence_progress.set(result['confidence'] / 100)
                self.command_value.configure(text=command_ru)

                if result['confidence'] < CONFIDENCE_THRESHOLD * 100:
                    self.gesture_value.configure(text_color="#e0a060")
                    self.confidence_value.configure(text_color="#e0a060")
                else:
                    self.gesture_value.configure(text_color="#d0d0d0")
                    self.confidence_value.configure(text_color="#f0c050")
            else:
                self.gesture_value.configure(text="Нет", text_color="#d0d0d0")
                self.confidence_value.configure(text="0%")
                self.confidence_progress.set(0)
                self.command_value.configure(text="Нет")

            # Конвертация для отображения
            frame_rgb = cv2.cvtColor(result['frame_with_marks'], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            # FPS
            frame_count += 1
            self.frames_label.configure(text=f"Обработано кадров: {frame_count}")

            if time.time() - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
                self.fps_label.configure(text=f"FPS: {self.fps}")

            self.fps_counter += 1

    def on_close(self):
        """Обработка закрытия окна"""
        self.is_running = False
        self.save_settings()
        if hasattr(self, 'recognizer') and self.recognizer:
            self.recognizer.stop()
        self.root.destroy()

    def run(self):
        """Запуск приложения"""
        self.root.mainloop()