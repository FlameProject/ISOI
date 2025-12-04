import cv2
import numpy as np
import os
from PIL import Image


# 1. ЗАГРУЗКА ИЗОБРАЖЕНИЯ
def load_image(image_path):
    """Загрузка изображения с проверкой"""
    img = cv2.imread(image_path)
    if img is None:
        # Пробуем найти файл
        print(f"❌ Файл не найден: {image_path}")
        print("\n📁 Доступные файлы в папке:")
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                print(f"  - {file}")

        # Запрашиваем правильное имя
        new_path = input("\nВведи правильное имя файла: ").strip()
        if not os.path.exists(new_path):
            print("❌ Файл не существует!")
            exit()
        img = cv2.imread(new_path)

    print(f"✅ Изображение загружено: {img.shape[1]}x{img.shape[0]}")
    return img


# 2. ПРОСТАЯ БИНАРИЗАЦИЯ ДЛЯ ТЕКСТА
def binarize_image(img):
    """Преобразуем в черно-белое для поиска символов"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Автоматическое определение: темный текст на светлом фоне или наоборот
    mean_intensity = np.mean(gray)

    if mean_intensity > 127:  # Светлый фон
        # Текст темный, нужно инвертировать
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Темный фон
        # Текст светлый
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


# 3. ПОИСК ОТДЕЛЬНЫХ СИМВОЛОВ
def find_characters(binary_img, min_width=5, min_height=10):
    """Находит bounding boxes отдельных символов"""
    # Находим контуры
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Фильтруем слишком маленькие объекты
        if w < min_width or h < min_height:
            continue

        # Фильтруем слишком большие объекты (скорее всего не символ)
        if w > binary_img.shape[1] * 0.5 or h > binary_img.shape[0] * 0.5:
            continue

        boxes.append((x, y, w, h))

    # Сортируем слева направо, сверху вниз
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))

    return boxes


# 4. РАЗДЕЛЕНИЕ СЛИПШИХСЯ СИМВОЛОВ
def split_connected_characters(boxes, binary_img, max_width_ratio=1.5):
    """Пытается разделить слишком широкие bounding boxes"""
    split_boxes = []

    for x, y, w, h in boxes:
        # Если символ слишком широкий для своей высоты
        if w > h * max_width_ratio:
            # Вырезаем область из бинарного изображения
            roi = binary_img[y:y + h, x:x + w]

            # Проекция по горизонтали (сколько белых пикселей в каждом столбце)
            projection = np.sum(roi == 255, axis=0)

            # Находим "провалы" в проекции - возможные места разделения
            threshold = np.max(projection) * 0.1
            valleys = np.where(projection < threshold)[0]

            if len(valleys) > 1:
                # Разделяем на несколько символов
                split_points = [0]

                for i in range(1, len(valleys)):
                    if valleys[i] - valleys[i - 1] > 1:
                        split_points.append((valleys[i - 1] + valleys[i]) // 2)

                split_points.append(w)

                # Создаем новые bounding boxes
                for i in range(len(split_points) - 1):
                    new_x = x + split_points[i]
                    new_w = split_points[i + 1] - split_points[i]
                    if new_w > 5:  # Минимальная ширина
                        split_boxes.append((new_x, y, new_w, h))
            else:
                split_boxes.append((x, y, w, h))
        else:
            split_boxes.append((x, y, w, h))

    return split_boxes


# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
def draw_boxes(img, boxes, output_path="boxes_result.png"):
    """Рисует bounding boxes на изображении"""
    result = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, str(i + 1), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(output_path, result)
    print(f"📊 Результат с bounding boxes сохранен как '{output_path}'")

    # Показываем изображение
    try:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        img_pil.show()
    except:
        pass

    return result


# 6. СОХРАНЕНИЕ КАЖДОГО СИМВОЛА ОТДЕЛЬНО
def save_characters(img, boxes, output_dir="characters"):
    """Сохраняет каждый символ как отдельный файл"""
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for i, (x, y, w, h) in enumerate(boxes, 1):
        # Добавляем небольшой отступ вокруг символа
        padding = 3
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        # Вырезаем символ
        char_img = img[y1:y2, x1:x2]

        # Сохраняем
        filename = f"char_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)

        # Конвертируем и сохраняем
        Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)).save(filepath)
        saved_files.append(filepath)

        print(f"  ✓ {filename} ({char_img.shape[1]}x{char_img.shape[0]})")

    return saved_files


# ОСНОВНАЯ ФУНКЦИЯ
def main():
    print("=" * 50)
    print("🎯 ВЫРЕЗАНИЕ ОТДЕЛЬНЫХ СИМВОЛОВ")
    print("=" * 50)

    # Имя файла
    image_file = "2.png"

    # 1. Загружаем изображение
    print("\n1. Загрузка изображения...")
    img = load_image(image_file)

    # 2. Бинаризация
    print("\n2. Бинаризация...")
    binary = binarize_image(img)

    # 3. Поиск символов
    print("\n3. Поиск символов...")
    boxes = find_characters(binary, min_width=3, min_height=8)
    print(f"   Найдено контуров: {len(boxes)}")

    # 4. Разделение слипшихся символов
    print("\n4. Проверка на слипшиеся символы...")
    boxes = split_connected_characters(boxes, binary, max_width_ratio=1.3)
    print(f"   После разделения: {len(boxes)} символов")

    if len(boxes) == 0:
        print("❌ Символы не найдены! Попробуй другую картинку.")
        return

    # 5. Визуализация
    print("\n5. Визуализация результатов...")
    result_img = draw_boxes(img, boxes)

    # 6. Сохранение каждого символа
    print(f"\n6. Сохранение {len(boxes)} символов...")
    saved = save_characters(img, boxes)

    print("\n" + "=" * 50)
    print(f"✅ ГОТОВО! Сохранено {len(saved)} символов в папку 'characters'")
    print("=" * 50)

    # 7. Показываем статистику
    print("\n📊 СТАТИСТИКА:")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        print(f"  Символ {i:2d}: позиция ({x:4d},{y:4d}), размер {w:3d}x{h:3d}")


# ЗАПУСК
if __name__ == "__main__":
    main()
    input("\nНажми Enter для выхода...")