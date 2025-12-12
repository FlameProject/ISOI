# preprocessing.py
import cv2
import numpy as np

def advanced_preprocessing_improved(image_path, show_steps=False):
    """МАКСИМАЛЬНО ПРОСТАЯ предобработка для черного текста на белом фоне"""
    print("🎯 ЗАПУСК ПРОСТОЙ ПРЕДОБРАБОТКИ...")
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    original = image.copy()
    
    # 1. Конвертация в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. ПРОСТОЕ УДАЛЕНИЕ ФОНА (если есть неравномерное освещение)
    print("   Простое выравнивание фона...")
    # Очень маленький blur, чтобы не потерять детали
    bg = cv2.medianBlur(gray, 15)
    normalized = cv2.divide(gray, bg, scale=255)
    
    # 3. ЛЁГКОЕ УЛУЧШЕНИЕ КОНТРАСТА (если нужно)
    print("   Лёгкое улучшение контраста...")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Очень мягкий
    enhanced = clahe.apply(normalized)
    
    # 4. ПРОСТАЯ БИНАРИЗАЦИЯ ОЦУ
    print("   Простая бинаризация Оцу...")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. АВТОМАТИЧЕСКАЯ ИНВЕРСИЯ (чтобы текст был белым на черном)
    print("   Автоматическая инверсия...")
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    # Если белых пикселей больше - это фон, нужно инвертировать
    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)
        print(f"   Инвертировано (было белых/черных: {white_pixels:,}/{black_pixels:,})")
    else:
        print(f"   Не инвертировано (белых/черных: {white_pixels:,}/{black_pixels:,})")
    
    # 6. МИНИМАЛЬНАЯ ОЧИСТКА (только единичные пиксели)
    print("   Минимальная очистка...")
    # Убираем одиночные белые пиксели на черном фоне
    kernel_clean = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    # 7. НЕМНОГО УСИЛИВАЕМ ТЕКСТ (если он слишком тонкий)
    print("   Усиление текста...")
    kernel_strengthen = np.ones((1, 1), np.uint8)
    final = cv2.dilate(cleaned, kernel_strengthen, iterations=1)
    
    # 8. ПРОСТАЯ ПРОВЕРКА - если у нас почти нет текста, возможно, инверсия была неправильной
    text_pixels = np.sum(final == 255)
    total_pixels = final.shape[0] * final.shape[1]
    text_ratio = text_pixels / total_pixels
    
    print(f"✅ ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print(f"   Размер: {final.shape[1]}x{final.shape[0]}")
    print(f"   Текст (белый): {text_pixels:,} пикселей ({text_ratio:.1%})")
    print(f"   Фон (черный): {total_pixels - text_pixels:,} пикселей")
    
    if text_ratio < 0.01:  # Меньше 1% текста - что-то не так
        print(f"   ⚠️  ОЧЕНЬ МАЛО ТЕКСТА! Возможно, нужно инвертировать?")
        # Пробуем инвертировать обратно
        final = cv2.bitwise_not(final)
        print(f"   Инвертировали обратно")
    
    if show_steps:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        steps = [
            ("1. Оригинал", original),
            ("2. Серое", gray),
            ("3. Улучшенное", enhanced),
            ("4. Бинарное", binary),
            ("5. Очищенное", cleaned),
            ("6. Финальное", final)
        ]
        
        for i, (title, img) in enumerate(steps):
            ax = axes[i//3, i%3]
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return original, final