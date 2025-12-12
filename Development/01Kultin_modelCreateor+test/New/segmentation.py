# segmentation.py
import cv2
import numpy as np

def segment_characters(binary_img, debug_mode=False):
    """Упрощенная сегментация символов"""
    print("\n✂️ СЕГМЕНТАЦИЯ СИМВОЛОВ...")
    
    img = binary_img.copy()
    
    # 1. Убедимся что текст БЕЛЫЙ на ЧЕРНОМ фоне
    white_px = np.sum(img == 255)
    black_px = np.sum(img == 0)
    
    if white_px > black_px:
        img = cv2.bitwise_not(img)
        if debug_mode:
            print(f"   Инвертировано (было белых/черных: {white_px:,}/{black_px:,})")
    
    # 2. Находим строки
    h, w = img.shape
    horizontal_proj = np.sum(img == 255, axis=1)
    
    lines = []
    in_line = False
    start = 0
    
    for y in range(h):
        has_text = horizontal_proj[y] > 0
        
        if has_text and not in_line:
            start = y
            in_line = True
        elif not has_text and in_line:
            end = y - 1
            if (end - start) >= 3:  # Минимум 3 пикселя высотой
                lines.append({'y1': start, 'y2': end})
            in_line = False
    
    if in_line:
        end = h - 1
        if (end - start) >= 3:
            lines.append({'y1': start, 'y2': end})
    
    if debug_mode:
        print(f"   Найдено строк: {len(lines)}")
    
    # 3. Для каждой строки находим символы
    all_boxes = []
    
    for line_idx, line in enumerate(lines):
        y1, y2 = line['y1'], line['y2']
        line_img = img[y1:y2+1, :]
        line_h = y2 - y1 + 1
        
        # Вертикальная проекция для этой строки
        vertical_proj = np.sum(line_img == 255, axis=0)
        empty_cols = vertical_proj == 0
        
        # Находим символы в строке
        char_start = None
        char_boxes = []
        
        for x in range(w):
            if empty_cols[x]:  # Пустой столбец
                if char_start is not None:  # Мы внутри символа
                    char_end = x - 1
                    char_width = char_end - char_start + 1
                    
                    if char_width >= 2:  # Минимум 2 пикселя
                        char_boxes.append((char_start, y1, char_width, line_h))
                    
                    char_start = None
            else:  # Есть текст
                if char_start is None:
                    char_start = x
        
        # Последний символ
        if char_start is not None:
            char_end = w - 1
            char_width = char_end - char_start + 1
            
            if char_width >= 2:
                char_boxes.append((char_start, y1, char_width, line_h))
        
        # Фильтруем слишком большие
        if char_boxes:
            widths = [w for _, _, w, _ in char_boxes]
            avg_width = np.mean(widths) if widths else 20
            
            filtered_char_boxes = []
            for box in char_boxes:
                x, y, w_box, h_box = box
                if (2 <= w_box <= avg_width * 3 and 
                    3 <= h_box <= line_h * 1.2):
                    filtered_char_boxes.append(box)
            
            char_boxes = filtered_char_boxes
        
        all_boxes.extend(char_boxes)
        
        if debug_mode:
            print(f"   Строка {line_idx}: {len(char_boxes)} символов")
    
    # 4. Финальная фильтрация и сортировка
    filtered_boxes = []
    for box in all_boxes:
        x, y, w, h = box
        if w >= 2 and h >= 3:
            filtered_boxes.append(box)
    
    # Сортируем
    filtered_boxes.sort(key=lambda b: (b[1], b[0]))
    
    if debug_mode:
        print(f"   Всего символов: {len(filtered_boxes)}")
        if filtered_boxes:
            widths = [w for _, _, w, _ in filtered_boxes]
            heights = [h for _, _, _, h in filtered_boxes]
            print(f"   Средний размер: {np.mean(widths):.1f}x{np.mean(heights):.1f} px")
    
    print(f"✅ СЕГМЕНТАЦИЯ ЗАВЕРШЕНА: {len(filtered_boxes)} символов")
    return filtered_boxes

# Удалены дублирующие функции:
# - segment_characters_simple
# - segment_characters_enhanced
# - segment_characters_by_lines
# - find_char_height
# - filter_boxes_by_size