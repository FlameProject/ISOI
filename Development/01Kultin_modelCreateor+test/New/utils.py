# utils.py
import numpy as np
import cv2

def extract_char_images_with_padding(original_img, boxes, padding_ratio=0.1):
    """Извлечение символов с паддингом"""
    char_images = []
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Адаптивный паддинг
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
        
        char_images.append(char_gray)
    
    return char_images

# Удалены избыточные функции:
# - merge_overlapping_boxes
# - split_connected_characters_advanced (две версии)