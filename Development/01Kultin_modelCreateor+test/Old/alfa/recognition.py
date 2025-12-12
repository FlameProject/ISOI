# recognition.py (полная версия)
import torch
import numpy as np
import cv2
from model import Config

def prepare_for_model_enhanced(char_img, target_size=24, enhance_contrast=True):
    """Улучшенная подготовка изображения символа с центрированием"""
    h, w = char_img.shape
    
    # 1. Усиление контраста для символа
    if enhance_contrast:
        if np.std(char_img) > 8:  # Если есть хоть какой-то контраст
            # Нормализуем
            char_normalized = cv2.normalize(char_img, None, 0, 255, cv2.NORM_MINMAX)
            
            # CLAHE для маленьких изображений
            if h > 10 and w > 10:  # Только для достаточно больших символов
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                char_img = clahe.apply(char_normalized)
            else:
                char_img = char_normalized
    
    # 2. Автоматическая бинаризация символа
    if len(np.unique(char_img)) > 2:
        _, binary = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        char_binary = binary
    else:
        char_binary = char_img
    
    # 3. Инвертируем если текст темный на светлом фоне
    if np.mean(char_binary) > 127:
        char_binary = 255 - char_binary
    
    # 4. Находим границы символа
    points = cv2.findNonZero(char_binary)
    if points is not None:
        x, y, w_char, h_char = cv2.boundingRect(points)
        
        # Вырезаем символ с небольшим отступом
        margin = 1
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + w_char + margin)
        y2 = min(h, y + h_char + margin)
        
        char_cropped = char_binary[y1:y2, x1:x2]
        h_crop, w_crop = char_cropped.shape
        
        # 5. Ресайз с сохранением пропорций
        scale = min(target_size / w_crop, (target_size * 0.85) / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        if new_w > 0 and new_h > 0:
            resized = cv2.resize(char_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = char_cropped
            new_w, new_h = w_crop, h_crop
    else:
        # Если символ пустой
        resized = np.zeros((target_size, target_size), dtype=np.uint8)
        new_w, new_h = target_size, target_size
    
    # 6. Создаем квадратное изображение с центрированием
    square = np.zeros((target_size, target_size), dtype=np.uint8)
    
    if points is not None:
        # Центрируем по горизонтали
        x_offset = (target_size - new_w) // 2
        # Смещаем немного вниз
        y_offset = target_size - new_h - 2
        
        # Проверяем границы
        y_offset = max(2, min(y_offset, target_size - new_h - 2))
        
        if y_offset + new_h <= target_size and x_offset + new_w <= target_size:
            square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            # Если не влезает, просто по центру
            y_offset = (target_size - new_h) // 2
            square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 7. Легкое сглаживание
    if np.std(square) > 10:
        square = cv2.GaussianBlur(square, (1, 1), 0.5)
    
    # 8. Нормализация для модели
    tensor_img = torch.from_numpy(square).float() / 255.0
    tensor_img = (tensor_img - 0.5) / 0.5
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)  # [1, 1, 24, 24]
    
    return tensor_img, square

def recognize_characters_enhanced(model, device, config, char_images):
    """Улучшенное распознавание символов"""
    print("🧠 ЗАПУСК РАСПОЗНАВАНИЯ...")
    
    recognized_chars = []
    confidences = []
    processed_images = []
    alternative_chars = []
    
    with torch.no_grad():
        for i, (char_img, bbox_info) in enumerate(char_images):
            # Подготовка изображения
            tensor_img, processed_img = prepare_for_model_enhanced(char_img, enhance_contrast=True)
            tensor_img = tensor_img.to(device)
            
            # Распознавание
            output = model(tensor_img)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Получаем топ-3 варианта
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            
            # Основной символ
            main_char_idx = top3_indices[0][0].item()
            main_confidence = top3_probs[0][0].item()
            
            if main_char_idx < len(config.chars):
                main_char = config.chars[main_char_idx]
            else:
                main_char = '?'
            
            # Альтернативные варианты
            alternatives = []
            for j in range(1, 3):
                alt_idx = top3_indices[0][j].item()
                alt_prob = top3_probs[0][j].item()
                if alt_idx < len(config.chars):
                    alternatives.append((config.chars[alt_idx], alt_prob))
            
            recognized_chars.append(main_char)
            confidences.append(main_confidence)
            processed_images.append(processed_img)
            alternative_chars.append(alternatives)
            
            # Прогресс
            if (i + 1) % 20 == 0 or i == 0 or i == len(char_images)-1:
                print(f"   {i+1}/{len(char_images)}: '{main_char}' ({main_confidence:.2%})")
    
    # Формируем текст
    text = reconstruct_text_with_spacing(
        recognized_chars, confidences, [b[1] for b in char_images])
    
    print("✅ РАСПОЗНАВАНИЕ ЗАВЕРШЕНО")
    print(f"   Всего символов: {len(recognized_chars)}")
    print(f"   Средняя уверенность: {np.mean(confidences):.2%}")
    
    if len(text) <= 100:
        print(f"   Текст: '{text}'")
    else:
        print(f"   Текст (первые 100 символов): '{text[:100]}...'")
    
    return text, recognized_chars, confidences, processed_images, alternative_chars

def reconstruct_text_with_spacing(chars, confidences, bboxes):
    """Восстановление текста с пробелами"""
    if not chars:
        return ""
    
    text_parts = []
    current_word = []
    
    for i, (char, confidence, bbox_info) in enumerate(zip(chars, confidences, bboxes)):
        if i == 0:
            current_word.append(char)
        else:
            # Проверяем расстояние между символами
            prev_x, prev_y, prev_w, prev_h = bboxes[i-1][:4]
            curr_x, curr_y, curr_w, curr_h = bbox_info[:4]
            
            distance = curr_x - (prev_x + prev_w)
            avg_height = (prev_h + curr_h) / 2
            
            # Если большое расстояние - вероятно пробел
            if distance > avg_height * 0.5:
                text_parts.append(''.join(current_word))
                text_parts.append(' ')
                current_word = [char]
            else:
                current_word.append(char)
    
    # Добавляем последнее слово
    if current_word:
        text_parts.append(''.join(current_word))
    
    return ''.join(text_parts)