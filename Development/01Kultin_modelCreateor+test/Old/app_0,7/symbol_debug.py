# symbol_debug.py - для отладки символов
import torch
import numpy as np
import cv2
from recognition import load_model, Preprocessor, OCRConfig

def test_symbols():
    """Тестирование распознавания символов"""
    print("🧪 ТЕСТИРОВАНИЕ СИМВОЛОВ")
    print("=" * 50)
    
    try:
        # Загружаем модель
        model, device, config = load_model()
        
        print(f"\n📋 СИМВОЛЫ В МОДЕЛИ ({len(config.chars)}):")
        for i, char in enumerate(config.chars):
            print(f"  {i:3d}: '{char}'")
        
        # Тестируем основные символы
        test_symbols = ['А', 'Б', 'В', 'а', 'б', 'в', '1', '2', '3']
        
        print(f"\n🧪 ТЕСТ РАСПОЗНАВАНИЯ:")
        for test_char in test_symbols:
            # Создаем изображение символа
            img = np.zeros((100, 100), dtype=np.uint8)
            cv2.putText(img, test_char, (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
            
            # Подготавливаем
            tensor_img, processed_img = Preprocessor.prepare_char(img, config)
            
            # Распознаем
            with torch.no_grad():
                tensor_img = tensor_img.to(device)
                output = model(tensor_img)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Топ-3 варианта
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                idx = top3_indices[0][0].item()
                char = config.chars[idx] if idx < len(config.chars) else '?'
                prob = top3_probs[0][0].item()
                
                print(f"  '{test_char}' -> '{char}' ({prob:.1%})")
                
                if char == '?':
                    print(f"    ⚠️  Распознано как '?' (индекс {idx})")
                    print(f"    Всего символов в модели: {len(config.chars)}")
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_symbols()