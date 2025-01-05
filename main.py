import tkinter as tk
from core.PneumoniaDetectorApp import PneumoniaDetectorApp

def main():
    root = tk.Tk()
    root.title("Détecteur de Pneumonie")
    window_width = 600
    window_height = 400

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    app = PneumoniaDetectorApp(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()