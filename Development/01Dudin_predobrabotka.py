import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Data/5282775041139478422_121.jpg')

# конвертация в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bg = cv2.medianBlur(gray, 77)
gray = cv2.divide(gray, bg, scale=255)

# Улучшение контраста перед бинаризацией
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# бинаризация
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


if np.mean(binary) > 127:  # Если больше белого чем черного
    binary = cv2.bitwise_not(binary)


# только очень мелкие точки удаляем
kernel_tiny = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny, iterations=1)

# закрытие для соединения разрывов в буквах
kernel_close = np.ones((2, 1), np.uint8)
final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Показываем только оригинал и результат после удаления шумов
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('До')
axes[0].axis('off')

axes[1].imshow(final, cmap='gray')
axes[1].set_title('После')
axes[1].axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite("Data/ep_predobrabotka.png", final)