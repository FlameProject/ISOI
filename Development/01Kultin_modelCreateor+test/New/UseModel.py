# ============================================
#   USE MODEL — Упрощённый, стабильный OCR 24×24
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import cv2
import random

# -----------------------------
# CONFIGURATION
# -----------------------------

class Config:
    img_size = 24
    target_h = 16
    font_size = 20
    chars = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "0123456789"
        ".,:()"
    )

cfg = Config()


# -----------------------------
# MODEL (тот же что в CreateModel)
# -----------------------------

class BetterCNN(nn.Module):
    def __init__(self, num_classes=len(cfg.chars)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# IMAGE PREPROCESSOR
# -----------------------------

def preprocess_symbol(arr):
    """Приводим символ к 24×24 так же, как в CreateModel"""
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    _, bin_img = cv2.threshold(arr, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(bin_img) > 127:
        bin_img = 255 - bin_img

    pts = cv2.findNonZero(bin_img)
    if pts is None:
        return None

    x, y, w, h = cv2.boundingRect(pts)
    crop = bin_img[y:y+h, x:x+w]

    scale = cfg.target_h / h
    new_w = max(3, int(w * scale))
    crop = cv2.resize(crop, (new_w, cfg.target_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((cfg.img_size, cfg.img_size), dtype=np.uint8)
    x0 = (cfg.img_size - new_w) // 2
    y0 = (cfg.img_size - cfg.target_h) // 2
    canvas[y0:y0+cfg.target_h, x0:x0+new_w] = crop

    t = torch.tensor(canvas, dtype=torch.float32) / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0).unsqueeze(0), canvas


# -----------------------------
# DRAW SYMBOL
# -----------------------------

def draw_char(ch, x_offset, y_offset, size=20, rotation=0):
    """Отрисовка символа с X/Y смещениями и вращением"""

    img = Image.new("L", (24, 24), 0)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        font = ImageFont.load_default()

    # Получаем размер буквы
    bbox = font.getbbox(ch)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Центрирование + смещение
    x = (24 - w) // 2 + x_offset
    y = (24 - h) // 2 + y_offset

    draw.text((x, y), ch, font=font, fill=255)

    if rotation != 0:
        img = img.rotate(rotation, fillcolor=0)

    return np.array(img)


# -----------------------------
# LOAD MODEL
# -----------------------------

def load_model(path="ocr24_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(path, map_location=device)

    model = BetterCNN(len(cfg.chars)).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, device


# -----------------------------
# PREDICT
# -----------------------------

def predict(model, device, arr):
    t, canvas = preprocess_symbol(arr)
    if t is None:
        return "?", 0.0, arr

    t = t.to(device)

    with torch.no_grad():
        out = model(t)
        prob = F.softmax(out, dim=1)[0]
        idx = torch.argmax(prob).item()

    return cfg.chars[idx], prob[idx].item(), canvas


# -----------------------------
# INTERACTIVE GUI
# -----------------------------

class Tester:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.char = "А"
        self.x = 0
        self.y = 0
        self.rot = 0
        self.font = 20

    def update(self, _=None):
        img = draw_char(self.char, self.x, self.y, self.font, self.rot)
        pred, conf, canvas = predict(self.model, self.device, img)

        self.ax_img.imshow(img, cmap="gray")
        self.ax_img.set_title(f"Symbol: {self.char}")

        self.ax_pred.imshow(canvas, cmap="gray")
        self.ax_pred.set_title(f"Pred: {pred} ({conf:.2f})")

        plt.draw()

    def run(self):
        fig = plt.figure(figsize=(10, 5))
        self.ax_img = fig.add_subplot(1, 2, 1)
        self.ax_pred = fig.add_subplot(1, 2, 2)

        ax_char = plt.axes([0.1, 0.02, 0.1, 0.05])
        box = TextBox(ax_char, "Char", initial=self.char)
        box.on_submit(lambda t: setattr(self, "char", t[:1]) or self.update())

        ax_x = plt.axes([0.25, 0.02, 0.2, 0.05])
        sx = Slider(ax_x, "X", -8, 8, valinit=0)
        sx.on_changed(lambda v: setattr(self, "x", int(v)) or self.update())

        ax_y = plt.axes([0.48, 0.02, 0.2, 0.05])
        sy = Slider(ax_y, "Y", -8, 8, valinit=0)
        sy.on_changed(lambda v: setattr(self, "y", int(v)) or self.update())

        ax_rot = plt.axes([0.71, 0.02, 0.2, 0.05])
        sr = Slider(ax_rot, "Rot", -15, 15, valinit=0)
        sr.on_changed(lambda v: setattr(self, "rot", int(v)) or self.update())

        self.update()
        plt.show()


if __name__ == "__main__":
    model, device = load_model()
    Tester(model, device).run()
