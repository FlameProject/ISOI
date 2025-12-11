import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2


# ================== КОНФИГУРАЦИЯ ==================
class OCRConfig:
    img_size = 24
    target_height = 16
    chars = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "0123456789"
        ".,:()"
    )
    num_classes = len(chars)

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
def preprocess_char(char_img):
    """
    Преобразует PIL Image или numpy array в нормализованный тензор для PyTorch
    """
    if isinstance(char_img, Image.Image):
        char_img = np.array(char_img, dtype=np.uint8)
    elif isinstance(char_img, torch.Tensor):
        return char_img  # уже тензор

    tensor = torch.from_numpy(char_img).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0)  # 1x24x24
    return tensor

# ================== ДАТАСЕТ ==================
class SyntheticCharDataset(Dataset):
    def __init__(self, config, num_samples=5000, font_path=None, max_shift=2):
        self.config = config
        self.chars = config.chars
        self.num_samples = num_samples
        self.font_path = font_path or "arial.ttf"
        self.max_shift = max_shift  # максимальное смещение в пикселях

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        char = random.choice(self.chars)
        img = self._render_char(char)
        tensor_img = preprocess_char(img)
        label = self.chars.index(char)
        return tensor_img, label

    def _render_char(self, char):
        img = Image.new("L", (self.config.img_size, self.config.img_size), 0)
        draw = ImageDraw.Draw(img)

        # случайный размер шрифта
        min_size = int(self.config.target_height * 0.8)
        max_size = self.config.target_height
        font_size = random.randint(min_size, max_size)
        font = ImageFont.truetype(self.font_path, font_size)

        # bbox символа с учётом невидимых отступов
        bbox = draw.textbbox((0, 0), char, font=font)
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0

        # создаём временный холст для символа
        char_img = Image.new("L", (w, h), 0)
        char_draw = ImageDraw.Draw(char_img)
        # рисуем символ с учетом bbox
        char_draw.text((-x0, -y0), char, fill=255, font=font)

        # масштабирование по большей стороне к максимально доступной области
        scale = min((self.config.img_size - 2)/w, (self.config.img_size - 2)/h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        char_img = char_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # бинаризация и тонкие линии (1 пиксель)
        arr = np.array(char_img)
        arr = np.where(arr > 127, 255, 0).astype(np.uint8)
        # можно применить морфологию, чтобы линии стали тоньше
        kernel = np.ones((1,1), np.uint8)
        arr = cv2.erode(arr, kernel, iterations=1)

        # размещение на холсте сверху слева с небольшим случайным смещением
        square = np.zeros((self.config.img_size, self.config.img_size), dtype=np.uint8)
        max_x_shift = min(self.max_shift, self.config.img_size - new_w)
        max_y_shift = min(self.max_shift, self.config.img_size - new_h)
        x_offset = random.randint(0, max_x_shift)
        y_offset = random.randint(0, max_y_shift)
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = arr

        return Image.fromarray(square)




# ================== ОБУЧЕНИЕ ==================
def train():
    config = OCRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterCNN(config.num_classes).to(device)

    dataset = SyntheticCharDataset(config, num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataset)
        accuracy = correct / total * 100
        print(f"Эпоха {epoch+1}/{epochs} - loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")

        # показываем 5 случайных символов после эпохи
        model.eval()
        samples = [dataset[i][0] for i in random.sample(range(len(dataset)), 5)]
        with torch.no_grad():
            plt.figure(figsize=(10,2))
            for i, img in enumerate(samples):
                output = model(img.unsqueeze(0).to(device))
                pred_idx = torch.argmax(output).item()
                pred = config.chars[pred_idx]
                plt.subplot(1,5,i+1)
                plt.imshow(img.squeeze(0), cmap='gray')
                plt.title(pred)
                plt.axis('off')
            plt.show()

    torch.save({'model': model.state_dict()}, "ocr24_trained.pth")
    print("Модель сохранена: ocr24_trained.pth")

if __name__ == "__main__":
    train()
