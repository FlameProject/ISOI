# recognition_new.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# ================== КОНФИГУРАЦИЯ ==================
class OCRConfig:
    img_size = 24
    
    # Символы (совпадают с тренировочной моделью)
    chars = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "0123456789"
        ".,:()"
    )
    num_classes = len(chars)
    target_height = 16

# ================== МОДЕЛЬ ==================
class BetterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 24->12

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 12->6

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3)  # всегда 3x3
        )

        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================== ПРЕПРОЦЕССОР ==================
# ================== ПРЕПРОЦЕССОР ==================
class Preprocessor:
    """Подготовка символов для модели, как на обучении (толстые линии, сверху слева)"""

    @staticmethod
    def prepare_char(char_img, config):
        if char_img is None or char_img.size == 0:
            return Preprocessor._create_empty_tensor(config)

        if not isinstance(char_img, np.ndarray):
            char_img = np.array(char_img, dtype=np.uint8)

        # если цветное → в градации серого
        if len(char_img.shape) > 2:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        # бинаризация: только 0/255
        _, binary = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        # инверсия: символ = 255, фон = 0
        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = 255 - binary

        # обрезаем до символа
        points = cv2.findNonZero(binary)
        if points is None or len(points) < 3:
            return Preprocessor._create_empty_tensor(config)
        x, y, w, h = cv2.boundingRect(points)
        char_cropped = binary[y:y+h, x:x+w]

        # масштабируем максимально по холсту
        scale = min((config.img_size - 2) / w, (config.img_size - 2) / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(char_cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # создаём квадратный холст 24x24
        square = np.zeros((config.img_size, config.img_size), dtype=np.uint8)
        # размещаем сверху слева
        square[0:new_h, 0:new_w] = resized

        # нормализация в тензор
        tensor = torch.from_numpy(square).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)  # 1x24x24

        return tensor, square

    @staticmethod
    def _create_empty_tensor(config):
        square = np.zeros((config.img_size, config.img_size), dtype=np.uint8)
        tensor = torch.from_numpy(square).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)
        return tensor, square



# ================== ЗАГРУЗКА МОДЕЛИ ==================
def load_model(model_path='ocr24_model.pth'):
    print(f"Загрузка модели: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    config = OCRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BetterCNN(config.num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Поддержка старых и новых форматов checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, device, config

# ================== РАСПОЗНАВАНИЕ ==================
def recognize_characters(model, device, config, char_images):
    recognized_chars = []
    confidences = []
    processed_images = []
    alternative_chars = []

    with torch.no_grad():
        for char_img in char_images:
            tensor_img, square_img = Preprocessor.prepare_char(char_img, config)
            tensor_img = tensor_img.unsqueeze(0).to(device)  # batch=1

            output = model(tensor_img)
            probs = torch.nn.functional.softmax(output, dim=1)
            top3_probs, top3_idx = torch.topk(probs, 3)

            main_idx = top3_idx[0][0].item()
            main_conf = top3_probs[0][0].item()
            main_char = config.chars[main_idx] if main_idx < len(config.chars) else '?'

            alternatives = [(config.chars[top3_idx[0][i].item()], top3_probs[0][i].item())
                            for i in range(1, 3) if top3_idx[0][i].item() < len(config.chars)]

            recognized_chars.append(main_char)
            confidences.append(main_conf)
            processed_images.append(square_img)
            alternative_chars.append(alternatives)

    text = ''.join(recognized_chars)
    return text, recognized_chars, confidences, processed_images, alternative_chars
