import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('Data\\op.jpg')
if image is None:
    raise FileNotFoundError("Не удалось загрузить изображение. Проверьте путь: Data\\op.jpg")

# 1. Конвертация в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Удаление неравномерного освещения
# Используем большое ядро для получения фона
bg = cv2.medianBlur(gray, 71)  # Увеличил ядро для более плавного фона
# Нормализация: делим оригинал на фон
normalized = cv2.divide(gray, bg, scale=255)

# 3. Улучшение контраста
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Увеличил clipLimit
enhanced = clahe.apply(normalized)

blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 6. Инверсия если нужно (текст должен быть черным на белом фоне)
if np.mean(binary) > 127:
    binary = cv2.bitwise_not(binary)

# 7. Морфологические операции для очистки
# Сначала удаляем мелкий шум
kernel_open = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=3)

# Закрытие для соединения разрывов в символах
kernel_close = np.ones((1, 1), np.uint8)  # Немного больше ядро
final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Первое фото (оригинал)
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Оригинал')
axes[0].axis('off')

# Второе фото (результат)
axes[1].imshow(binary, cmap='gray')
axes[1].set_title('После обработки')
axes[1].axis('off')

plt.tight_layout()
plt.show()