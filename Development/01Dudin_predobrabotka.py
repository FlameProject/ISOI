import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Data\\op.jpg')
if image is None:
    raise FileNotFoundError("Не удалось загрузить изображение. Проверьте путь: Data\\op.jpg")

# конвертация в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Удаление неравномерного освещения
bg = cv2.medianBlur(gray, 71)
# Нормализация
normalized = cv2.divide(gray, bg, scale=255)

# Улучшение контраста
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
enhanced = clahe.apply(normalized)

blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Инверсия если нужно
if np.mean(binary) > 127:
    binary = cv2.bitwise_not(binary)

# удаляем мелкий шум
kernel_open = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

# закрытие для соединения разрывов в символах
kernel_close = np.ones((1, 1), np.uint8)
final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Оригинал')
axes[0].axis('off')

axes[1].imshow(final, cmap='gray')
axes[1].set_title('После обработки')
axes[1].axis('off')

plt.tight_layout()
plt.show()
output_filename = 'Data\\op_predobrabotka.png'
cv2.imwrite(output_filename, final)