# main.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui_app import GestureGUI


def main():
    print("=" * 60)
    print("ЗАПУСК СИСТЕМЫ УПРАВЛЕНИЯ ЖЕСТАМИ")
    print("Версия 2.0")
    print("Разработчик: Маликова А.В.")
    print("=" * 60)

    app = GestureGUI()
    app.run()


if __name__ == "__main__":
    main()