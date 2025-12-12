# segmentation.py
import cv2
import numpy as np
from utils import extract_char_images_with_padding

def segment_characters_smart(binary_img, original_img, debug_mode=False):
    """Умная сегментация: строки → символы с проверкой на реальные символы"""
    print("\n🎯 ЗАПУСК УМНОЙ СЕГМЕНТАЦИИ...")
    
    # 1. ПОДГОТОВКА
    img = binary_img.copy()
    
    # Убедимся что текст белый на черном фоне
    white_px = np.sum(img == 255)
    black_px = np.sum(img == 0)
    
    if white_px > black_px * 1.5:
        img = cv2.bitwise_not(img)
        if debug_mode:
            print("   Инвертировано: текст белый на черном фоне")
    
    # 2. НАХОДИМ СТРОКИ (по горизонтальным проекциям)
    horizontal_proj = np.sum(img == 255, axis=1)
    text_rows = horizontal_proj > 0
    
    # Группируем строки
    lines = []
    in_line = False
    line_start = 0
    
    for i, has_text in enumerate(text_rows):
        if has_text and not in_line:
            line_start = i
            in_line = True
        elif not has_text and in_line:
            line_end = i - 1
            line_height = line_end - line_start + 1
            
            if line_height >= 5:  # Минимальная высота строки
                # Находим границы текста в строке
                line_img = img[line_start:line_end+1, :]
                vertical_proj = np.sum(line_img == 255, axis=0)
                text_cols = vertical_proj > 0
                
                if np.any(text_cols):
                    x_indices = np.where(text_cols)[0]
                    x_left = x_indices[0]
                    x_right = x_indices[-1]
                    
                    lines.append({
                        'y': line_start,
                        'x': x_left,
                        'h': line_height,
                        'w': x_right - x_left + 1,
                        'img': line_img[:, x_left:x_right+1]
                    })
            
            in_line = False
    
    if in_line:  # Последняя строка
        line_end = len(text_rows) - 1
        line_height = line_end - line_start + 1
        
        if line_height >= 5:
            line_img = img[line_start:line_end+1, :]
            vertical_proj = np.sum(line_img == 255, axis=0)
            text_cols = vertical_proj > 0
            
            if np.any(text_cols):
                x_indices = np.where(text_cols)[0]
                x_left = x_indices[0]
                x_right = x_indices[-1]
                
                lines.append({
                    'y': line_start,
                    'x': x_left,
                    'h': line_height,
                    'w': x_right - x_left + 1,
                    'img': line_img[:, x_left:x_right+1]
                })
    
    if debug_mode:
        print(f"   Найдено строк: {len(lines)}")
    
    # 3. ДЛЯ КАЖДОЙ СТРОКИ НАХОДИМ СИМВОЛЫ
    all_boxes = []
    
    for line_idx, line in enumerate(lines):
        if debug_mode:
            print(f"\n   Строка {line_idx}: {line['w']}x{line['h']}")
        
        line_boxes = segment_line_smart(line['img'], line['x'], line['y'], debug_mode)
        all_boxes.extend(line_boxes)
        
        if debug_mode:
            print(f"     Найдено символов: {len(line_boxes)}")
    
    # 4. ФИЛЬТРАЦИЯ И СОРТИРОВКА
    all_boxes = filter_boxes_by_size(all_boxes, debug_mode)
    
    # Сортируем: сначала по строкам (y), потом по позиции в строке (x)
    if all_boxes:
        # Группируем по строкам (близкие y - одна строка)
        boxes_by_line = {}
        for box in all_boxes:
            x, y, w, h = box
            line_key = round(y / 10) * 10  # Группируем по y с шагом 10px
            
            if line_key not in boxes_by_line:
                boxes_by_line[line_key] = []
            boxes_by_line[line_key].append(box)
        
        # Сортируем каждую строку по x и объединяем
        sorted_boxes = []
        for line_key in sorted(boxes_by_line.keys()):
            line_boxes = sorted(boxes_by_line[line_key], key=lambda b: b[0])
            sorted_boxes.extend(line_boxes)
        
        all_boxes = sorted_boxes
    
    if debug_mode:
        print(f"\n✅ СЕГМЕНТАЦИЯ ЗАВЕРШЕНА")
        print(f"   Всего символов: {len(all_boxes)}")
        
        if all_boxes:
            widths = [w for _, _, w, _ in all_boxes]
            heights = [h for _, _, _, h in all_boxes]
            
            print(f"   Средний размер: {np.mean(widths):.1f}x{np.mean(heights):.1f} px")
            print(f"   Разброс ширины: {np.min(widths)}-{np.max(widths)} px")
    
    return all_boxes

def segment_line_smart(line_img, offset_x, offset_y, debug_mode=False):
    """Умная сегментация символов в строке"""
    h, w = line_img.shape
    
    if w < 5:
        return []
    
    # 1. ВЕРТИКАЛЬНЫЕ ПРОЕКЦИИ
    vertical_proj = np.sum(line_img == 255, axis=0)
    max_proj = np.max(vertical_proj)
    
    if max_proj == 0:
        return []
    
    if debug_mode:
        print(f"     Длина строки: {w}px, макс проекция: {max_proj}")
    
    # 2. АНАЛИЗ ПРОЕКЦИЙ - находим РЕАЛЬНЫЕ промежутки
    # Нормализуем проекции
    normalized = vertical_proj / max_proj if max_proj > 0 else vertical_proj
    
    # Ищем устойчивые промежутки (где несколько подряд идущих колонок пустые)
    gap_threshold = 0.1  # 10% от максимума
    min_gap_width = 2    # Минимальная ширина промежутка
    
    # Находим все промежутки
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, proj_value in enumerate(normalized):
        if proj_value < gap_threshold and not in_gap:
            gap_start = i
            in_gap = True
        elif proj_value >= gap_threshold and in_gap:
            gap_end = i - 1
            gap_width = gap_end - gap_start + 1
            
            if gap_width >= min_gap_width:
                gaps.append((gap_start, gap_end, gap_width))
            
            in_gap = False
    
    # Последний промежуток
    if in_gap:
        gap_end = w - 1
        gap_width = gap_end - gap_start + 1
        if gap_width >= min_gap_width:
            gaps.append((gap_start, gap_end, gap_width))
    
    if debug_mode:
        print(f"     Найдено промежутков: {len(gaps)}")
    
    # 3. РАЗДЕЛЯЕМ СТРОКУ НА СИМВОЛЫ ПО ПРОМЕЖУТКАМ
    boxes = []
    
    if not gaps:  # Нет промежутков - вся строка один символ?
        # Но проверяем ширину
        if w > 50:  # Слишком широко для одного символа
            # Принудительно ищем минимумы в проекциях
            char_parts = split_by_minima(line_img, 0, 0, debug_mode)
            for part in char_parts:
                x_part, w_part = part
                # Находим высоту
                char_roi = line_img[:, x_part:x_part+w_part]
                y_top, char_h = find_char_height(char_roi)
                
                if char_h >= 5 and w_part >= 3:
                    boxes.append((
                        offset_x + x_part,
                        offset_y + y_top,
                        w_part,
                        char_h
                    ))
        else:
            # Один символ
            y_top, char_h = find_char_height(line_img)
            boxes.append((
                offset_x,
                offset_y + y_top,
                w,
                char_h
            ))
    else:
        # Разделяем по промежуткам
        # Создаем точки разделения
        split_points = [0]
        for gap_start, gap_end, _ in gaps:
            split_point = (gap_start + gap_end) // 2
            split_points.append(split_point)
        split_points.append(w)
        
        # Объединяем слишком близкие точки разделения
        filtered_splits = [split_points[0]]
        for i in range(1, len(split_points)-1):
            if split_points[i] - filtered_splits[-1] >= 5:  # Минимум 5px между символами
                filtered_splits.append(split_points[i])
        filtered_splits.append(split_points[-1])
        
        # Создаем символы
        for i in range(len(filtered_splits)-1):
            char_start = filtered_splits[i]
            char_end = filtered_splits[i+1]
            char_width = char_end - char_start
            
            if char_width >= 3:  # Минимальная ширина символа
                # Вырезаем символ
                char_roi = line_img[:, char_start:char_end]
                
                # Проверяем, не состоит ли он из нескольких частей
                char_vertical_proj = np.sum(char_roi == 255, axis=0)
                char_max_proj = np.max(char_vertical_proj)
                
                # Если внутри символа есть глубокие провалы
                if char_max_proj > 0:
                    char_normalized = char_vertical_proj / char_max_proj
                    
                    # Ищем внутренние провалы (глубокие)
                    deep_valleys = np.sum(char_normalized < 0.3)
                    
                    if deep_valleys >= 3 and char_width > 15:
                        # Возможно несколько символов
                        sub_parts = split_by_minima(char_roi, char_start, 0, debug_mode)
                        for part in sub_parts:
                            x_part, w_part = part
                            sub_roi = line_img[:, x_part:x_part+w_part]
                            y_top, char_h = find_char_height(sub_roi)
                            
                            if char_h >= 5 and w_part >= 3:
                                boxes.append((
                                    offset_x + x_part,
                                    offset_y + y_top,
                                    w_part,
                                    char_h
                                ))
                    else:
                        # Один символ
                        y_top, char_h = find_char_height(char_roi)
                        
                        if char_h >= 5:
                            boxes.append((
                                offset_x + char_start,
                                offset_y + y_top,
                                char_width,
                                char_h
                            ))
    
    return boxes

def find_char_height(char_roi):
    """Находит верх и высоту символа"""
    h, w = char_roi.shape
    if h == 0 or w == 0:
        return 0, 0
    
    horizontal_proj = np.sum(char_roi == 255, axis=1)
    text_rows = horizontal_proj > 0
    
    if np.any(text_rows):
        y_indices = np.where(text_rows)[0]
        y_top = y_indices[0]
        char_h = y_indices[-1] - y_top + 1
        return y_top, char_h
    
    return 0, 0

def split_by_minima(char_roi, offset_x, offset_y, debug_mode=False):
    """Разделяет символ по локальным минимумам в проекциях"""
    h, w = char_roi.shape
    
    if w < 10:
        return [(offset_x, w)]
    
    vertical_proj = np.sum(char_roi == 255, axis=0)
    max_proj = np.max(vertical_proj)
    
    if max_proj == 0:
        return [(offset_x, w)]
    
    # Находим локальные минимумы
    minima = []
    
    for i in range(1, w-1):
        # Проверяем что это минимум
        if vertical_proj[i] <= vertical_proj[i-1] and vertical_proj[i] <= vertical_proj[i+1]:
            # Проверяем глубину (должен быть достаточно глубоким)
            depth_ratio = vertical_proj[i] / max_proj if max_proj > 0 else 0
            
            if depth_ratio < 0.4:  # Глубокий минимум (менее 40% от максимума)
                minima.append(i)
    
    if not minima:
        return [(offset_x, w)]
    
    if debug_mode:
        print(f"       Найдено минимумов: {len(minima)}")
    
    # Сортируем и фильтруем слишком близкие минимумы
    minima.sort()
    filtered_minima = []
    
    if minima:
        filtered_minima.append(minima[0])
        for i in range(1, len(minima)):
            if minima[i] - filtered_minima[-1] >= 5:  # Минимум 5px между разделениями
                filtered_minima.append(minima[i])
    
    # Создаем части
    parts = []
    start = 0
    
    for min_pos in filtered_minima:
        part_width = min_pos - start
        if part_width >= 3:
            parts.append((offset_x + start, part_width))
        start = min_pos
    
    # Последняя часть
    last_width = w - start
    if last_width >= 3:
        parts.append((offset_x + start, last_width))
    
    return parts if parts else [(offset_x, w)]

def filter_boxes_by_size(boxes, debug_mode=False):
    """Фильтрует боксы по размеру и удаляет пересекающиеся"""
    if not boxes:
        return boxes
    
    # 1. Собираем статистику по размерам
    widths = [w for _, _, w, _ in boxes]
    heights = [h for _, _, _, h in boxes]
    
    if not widths or not heights:
        return boxes
    
    median_width = np.median(widths)
    median_height = np.median(heights)
    
    if debug_mode:
        print(f"   Медианный размер: {median_width:.1f}x{median_height:.1f}")
    
    # 2. Фильтруем по размеру
    filtered = []
    
    for box in boxes:
        x, y, w, h = box
        
        # Проверяем соотношения
        width_ratio = w / median_width if median_width > 0 else 1
        height_ratio = h / median_height if median_height > 0 else 1
        
        # Допустимые диапазоны
        if (0.3 < width_ratio < 3.0 and 
            0.4 < height_ratio < 2.5 and
            w >= 3 and h >= 5):
            filtered.append(box)
        elif debug_mode:
            print(f"   Отфильтрован: {w}x{h} (w_ratio={width_ratio:.2f}, h_ratio={height_ratio:.2f})")
    
    # 3. Удаляем пересекающиеся
    final_boxes = []
    
    for i, (x1, y1, w1, h1) in enumerate(filtered):
        overlap = False
        
        for x2, y2, w2, h2 in final_boxes:
            # Проверяем пересечение
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            
            # Если значительное пересечение
            if x_overlap > min(w1, w2) * 0.4 and y_overlap > min(h1, h2) * 0.6:
                overlap = True
                break
        
        if not overlap:
            final_boxes.append((x1, y1, w1, h1))
    
    return final_boxes

# Функции для совместимости
def segment_characters_enhanced(binary_img, original_img, debug_mode=False):
    return segment_characters_smart(binary_img, original_img, debug_mode)

def segment_characters_simple(binary_img, original_img, debug_mode=False):
    return segment_characters_smart(binary_img, original_img, debug_mode)

def segment_characters_by_lines(binary_img, original_img, debug_mode=False):
    return segment_characters_smart(binary_img, original_img, debug_mode)

# Остальные функции
def split_wide_character(binary_img, x, y, w, h):
    roi = binary_img[y:y+h, x:x+w]
    parts = split_by_minima(roi, 0, 0, debug_mode=False)
    
    result = []
    for x_part, w_part in parts:
        result.append((x + x_part, y, w_part, h))
    
    return result if result else [(x, y, w, h)]

def check_if_needs_split(roi):
    h, w = roi.shape
    if w < 15:
        return False
    
    vertical_proj = np.sum(roi == 255, axis=0)
    max_proj = np.max(vertical_proj)
    
    if max_proj == 0:
        return False
    
    normalized = vertical_proj / max_proj
    minima = []
    
    for i in range(1, w-1):
        if normalized[i] <= normalized[i-1] and normalized[i] <= normalized[i+1]:
            if normalized[i] < 0.4:
                minima.append(i)
    
    return len(minima) >= 1

def split_by_projection(roi, offset_x, offset_y):
    parts = split_by_minima(roi, 0, 0, debug_mode=False)
    
    result = []
    for x_part, w_part in parts:
        result.append((offset_x + x_part, offset_y, w_part, roi.shape[0]))
    
    return result if result else [(offset_x, offset_y, roi.shape[1], roi.shape[0])]

def remove_nested_boxes(boxes):
    return filter_boxes_by_size(boxes, debug_mode=False)