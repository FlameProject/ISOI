# utils.py
import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """Объединение пересекающихся bounding boxes"""
    if not boxes:
        return boxes
    
    # Сортируем по X координате
    boxes = sorted(boxes, key=lambda b: b[0])
    
    merged = []
    current_box = boxes[0]
    
    for box in boxes[1:]:
        x1, y1, w1, h1 = current_box
        x2, y2, w2, h2 = box
        
        # Проверяем пересечение
        x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        
        overlap_area = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Если пересечение значительное
        if overlap_area > 0 and (overlap_area/area1 > overlap_threshold or 
                                overlap_area/area2 > overlap_threshold):
            # Объединяем боксы
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1+w1, x2+w2) - new_x
            new_h = max(y1+h1, y2+h2) - new_y
            current_box = (new_x, new_y, new_w, new_h)
        else:
            merged.append(current_box)
            current_box = box
    
    merged.append(current_box)
    return merged

def extract_char_images_with_padding(original_img, boxes, padding_ratio=0.2):
    """Извлечение символов с адаптивным паддингом"""
    char_images = []
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Адаптивный паддинг (процент от размера символа)
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(original_img.shape[1], x + w + pad_x)
        y2 = min(original_img.shape[0], y + h + pad_y)
        
        # Вырезаем символ
        char_region = original_img[y1:y2, x1:x2]
        
        # Конвертируем в градации серого если нужно
        if len(char_region.shape) == 3:
            char_gray = cv2.cvtColor(char_region, cv2.COLOR_BGR2GRAY)
        else:
            char_gray = char_region
        
        # Нормализуем контраст для этого символа
        char_normalized = cv2.normalize(char_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        char_images.append((char_normalized, (x, y, w, h, x1, y1, x2, y2)))
    
    return char_images

def split_connected_characters_advanced(roi, offset_x, offset_y, avg_height):
    """Улучшенное разделение слипшихся символов"""
    boxes = []
    
    # Проекция по вертикали
    vertical_proj = np.sum(roi == 255, axis=0)
    
    # Находим "провалы" в проекции
    min_val = np.min(vertical_proj)
    threshold = min_val * 2 if min_val > 0 else 1
    
    # Находим места, где проекция ниже порога
    below_threshold = vertical_proj < threshold
    
    # Находим сегменты (группы пикселей)
    segments = []
    in_segment = False
    start = 0
    
    for i, val in enumerate(below_threshold):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i-1))
            in_segment = False
    
    if in_segment:
        segments.append((start, len(below_threshold)-1))
    
    # Если найдены сегменты для разделения
    if segments:
        split_points = [0]
        for start, end in segments:
            if end - start > 2:  # Минимальная ширина разделителя
                split_points.append((start + end) // 2)
        split_points.append(roi.shape[1])
        
        # Создаем боксы для каждого символа
        for i in range(len(split_points)-1):
            char_start = split_points[i]
            char_end = split_points[i+1]
            char_width = char_end - char_start
            
            if char_width > 3:  # Минимальная ширина символа
                boxes.append((
                    offset_x + char_start,
                    offset_y,
                    char_width,
                    roi.shape[0]
                ))
    else:
        # Не смогли разделить - возвращаем как один символ
        boxes.append((offset_x, offset_y, roi.shape[1], roi.shape[0]))
    
    return boxes

# utils.py (дополнение)
def split_connected_characters_advanced(roi, offset_x, offset_y, avg_height):
    """Альтернативное разделение слипшихся символов (для backward compatibility)"""
    # Просто вызываем новую функцию
    return split_connected_characters(roi, offset_x, offset_y)