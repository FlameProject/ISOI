# main.py
import tkinter as tk
from gui import OCRAppEnhanced

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRAppEnhanced(root)
    root.mainloop()
