# interactive_symbol_tester_24x24_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.patches as patches
import warnings

# Отключаем предупреждения о шрифтах
warnings.filterwarnings('ignore')

# Конфигурация модели 24x24
class Config:
    img_size = 24
    russian_upper = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    russian_lower = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    digits = '0123456789'
    punctuation = '.,:()'
    
    chars = russian_upper + russian_lower + digits + punctuation
    num_classes = len(chars)
    font_path = "arial.ttf"

# Архитектура модели
class SimpleCharRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCharRecognizer, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class InteractiveSymbolTester24x24:
    def __init__(self, model_path='best_char_model_24x24.pth'):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка модели
        self.model = SimpleCharRecognizer(num_classes=self.config.num_classes)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Модель 24x24 загружена!")
            print(f"📊 Точность модели: {checkpoint.get('accuracy', 'N/A')}%")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("💡 Создана новая модель (для тестирования)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Параметры символа
        self.current_char = 'А'
        self.x_offset = 0
        self.y_offset = 0
        self.font_size = 18
        self.rotation = 0
        
        # Графические элементы
        self.fig = None
        self.ax_main = None
        self.ax_preview = None
        self.sliders = []
        
        print(f"🎮 Интерактивный тестер символов 24x24 запущен!")
        print(f"📊 Поддерживается {self.config.num_classes} символов")

    def create_symbol_image_simple(self):
        """ПРОСТОЙ метод создания символа - гарантированно работает"""
        img_size = self.config.img_size
        img = Image.new('L', (img_size, img_size), 0)  # Черный фон
        
        # Создаем простой белый квадрат для теста
        draw = ImageDraw.Draw(img)
        
        # 🔥 ДИАГНОСТИКА: Сначала проверим базовую функциональность
        
        # Вариант 1: Просто рисуем прямоугольник
        if self.current_char == 'TEST_RECT':
            draw.rectangle([5, 5, 19, 19], fill=255)
            return img
        
        # Вариант 2: Рисуем символ простым методом
        try:
            # Пробуем загрузить шрифт
            try:
                font = ImageFont.truetype("arial.ttf", self.font_size)
                print(f"✅ Шрифт Arial загружен: {self.font_size}px")
            except:
                font = ImageFont.load_default()
                print("⚠️  Используется системный шрифт по умолчанию")
            
            # Простое позиционирование по центру
            text_x = 6 + self.x_offset
            text_y = 4 + self.y_offset
            
            # Рисуем символ
            draw.text((text_x, text_y), self.current_char, fill=255, font=font)
            print(f"✅ Нарисован символ: '{self.current_char}' в позиции ({text_x}, {text_y})")
            
        except Exception as e:
            print(f"❌ Ошибка рисования символа: {e}")
            # Fallback: рисуем крестик
            draw.line([2, 2, 22, 22], fill=255, width=2)
            draw.line([22, 2, 2, 22], fill=255, width=2)
        
        # Применяем вращение
        if abs(self.rotation) > 1:
            img = img.rotate(self.rotation, resample=Image.BICUBIC, expand=False, fillcolor=0)
        
        return img

    def create_symbol_image_advanced(self):
        """УЛУЧШЕННЫЙ метод с отладкой"""
        img_size = self.config.img_size
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        
        print(f"🔍 ОТЛАДКА: Символ='{self.current_char}', Шрифт={self.font_size}px")
        
        # Тест 1: Проверяем отображение простых фигур
        if self.current_char == 'TEST':
            # Рисуем тестовый паттерн
            draw.rectangle([2, 2, 10, 10], fill=255)  # Левый верхний квадрат
            draw.rectangle([14, 2, 22, 10], fill=255)  # Правый верхний квадрат  
            draw.rectangle([2, 14, 10, 22], fill=255)  # Левый нижний квадрат
            draw.rectangle([14, 14, 22, 22], fill=255)  # Правый нижний квадрат
            return img
        
        # Тест 2: Пробуем разные методы отрисовки
        try:
            # Метод 1: Прямой вызов с системным шрифтом
            font = ImageFont.load_default()
            draw.text((6, 4), self.current_char, fill=255, font=font)
            print("✅ Метод 1: Системный шрифт")
            
            # Метод 2: Пробуем создать изображение символа отдельно
            char_img = Image.new('L', (20, 20), 0)
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((2, 2), self.current_char, fill=255, font=font)
            
            # Вставляем в основное изображение
            img.paste(char_img, (2 + self.x_offset, 2 + self.y_offset))
            print("✅ Метод 2: Отдельное изображение символа")
            
        except Exception as e:
            print(f"❌ Ошибка улучшенного метода: {e}")
            # Резервный метод: рисуем рамку
            draw.rectangle([1, 1, 23, 23], outline=255, width=1)
            draw.text((8, 8), "?", fill=255)
        
        if abs(self.rotation) > 1:
            img = img.rotate(self.rotation, resample=Image.BICUBIC, expand=False, fillcolor=0)
            
        return img

    def create_symbol_image(self):
        """Основной метод создания символа"""
        # Пробуем простой метод
        return self.create_symbol_image_simple()

    def predict_symbol(self, img):
        """Предсказание символа моделью"""
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_char = self.config.chars[predicted.item()]
            confidence_value = confidence.item()
            
            # Топ-3 предсказания
            all_probs = probabilities[0].cpu().numpy()
            top3_indices = np.argsort(all_probs)[-3:][::-1]
            top3_predictions = [(self.config.chars[idx], all_probs[idx]) for idx in top3_indices]
        
        return predicted_char, confidence_value, top3_predictions

    def update_display(self, event=None):
        """Обновление отображения"""
        try:
            print(f"\n🔄 ОБНОВЛЕНИЕ: символ='{self.current_char}', смещение=({self.x_offset},{self.y_offset})")
            
            # Создаем символ с текущими параметрами
            img = self.create_symbol_image()
            
            # Конвертируем в numpy для отображения
            img_array = np.array(img)
            print(f"📊 Изображение: shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}")
            
            # Очищаем основную область
            self.ax_main.clear()
            
            # Отображаем символ
            self.ax_main.imshow(img_array, cmap='gray', vmin=0, vmax=255)
            
            # Добавляем сетку
            img_size = self.config.img_size
            for i in range(0, img_size, 4):
                self.ax_main.axhline(i, color='red', alpha=0.3, linewidth=0.5)
                self.ax_main.axvline(i, color='red', alpha=0.3, linewidth=0.5)
            
            # Центральные линии
            self.ax_main.axhline(img_size//2, color='yellow', alpha=0.8, linewidth=1)
            self.ax_main.axvline(img_size//2, color='yellow', alpha=0.8, linewidth=1)
            
            self.ax_main.set_title(f'Символ: "{self.current_char}" | Шрифт: {self.font_size}px', 
                                 fontsize=14, color='white', pad=10)
            self.ax_main.set_facecolor('black')
            self.ax_main.tick_params(colors='white')
            self.ax_main.set_xlim(0, img_size)
            self.ax_main.set_ylim(img_size, 0)
            
            # Предсказание моделью
            predicted_char, confidence, top3 = self.predict_symbol(img)
            
            # Отображаем результат предсказания
            result_text = f'Модель: "{predicted_char}"\nУверенность: {confidence:.1%}'
            color = 'lime' if confidence > 0.7 else 'yellow' if confidence > 0.3 else 'red'
            
            self.ax_main.text(0.02, 0.98, result_text, transform=self.ax_main.transAxes,
                             fontsize=11, verticalalignment='top', color='white',
                             bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            
            # Топ-3 предсказания
            top3_text = "Топ-3:\n" + "\n".join([f"'{char}': {prob:.1%}" for char, prob in top3])
            self.ax_main.text(0.98, 0.98, top3_text, transform=self.ax_main.transAxes,
                             fontsize=9, verticalalignment='top', horizontalalignment='right', color='white',
                             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
            
            # Отображаем текущие параметры
            params_text = (f"Смещение: ({self.x_offset}, {self.y_offset})\n"
                          f"Размер шрифта: {self.font_size}px\n"
                          f"Вращение: {self.rotation}°")
            
            self.ax_main.text(0.02, 0.02, params_text, transform=self.ax_main.transAxes,
                             fontsize=9, verticalalignment='bottom', color='white',
                             bbox=dict(boxstyle='round', facecolor='purple', alpha=0.7))
            
            # Обновляем превью
            self.ax_preview.clear()
            preview_img = img.resize((96, 96), Image.Resampling.LANCZOS)
            self.ax_preview.imshow(preview_img, cmap='gray')
            self.ax_preview.set_title('Превью (увеличено)', fontsize=10, color='white')
            self.ax_preview.set_facecolor('black')
            self.ax_preview.axis('off')
            
            self.fig.canvas.draw_idle()
            print("✅ Отображение обновлено")
            
        except Exception as e:
            print(f"❌ Ошибка в update_display: {e}")
            import traceback
            traceback.print_exc()

    def on_char_change(self, text):
        """Обработчик изменения символа"""
        if text and text in self.config.chars:
            self.current_char = text
            print(f"🔤 СИМВОЛ ИЗМЕНЕН НА: '{self.current_char}'")
            self.update_display()
        else:
            print(f"⚠️  Неверный символ: '{text}'")

    def on_font_size_change(self, text):
        """Обработчик изменения размера шрифта"""
        try:
            size = int(text)
            if 8 <= size <= 30:
                self.font_size = size
                print(f"🔤 РАЗМЕР ШРИФТА ИЗМЕНЕН НА: {size}px")
                self.update_display()
            else:
                print(f"⚠️  Неверный размер шрифта: {size}")
        except ValueError:
            print(f"⚠️  Неверный формат размера шрифта: '{text}'")

    def create_interactive_interface(self):
        """Создание интерактивного интерфейса"""
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        
        # Основная область для отображения символа
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2, facecolor='black')
        
        # Область превью
        self.ax_preview = plt.subplot2grid((3, 4), (0, 3), facecolor='black')
        
        # Настройка расположения слайдеров
        slider_height = 0.03
        slider_width = 0.12
        start_x = 0.05
        start_y = 0.25
        
        # Текстовое поле для символа
        ax_char = plt.axes([start_x, start_y + 0.15, slider_width, 0.04], facecolor='gray')
        char_box = TextBox(ax_char, 'Символ: ', initial=self.current_char, color='white', hovercolor='darkgray')
        char_box.on_submit(self.on_char_change)
        
        # Текстовое поле для размера шрифта
        ax_font_size = plt.axes([start_x + slider_width + 0.02, start_y + 0.15, slider_width, 0.04], facecolor='gray')
        font_size_box = TextBox(ax_font_size, 'Шрифт (px): ', initial=str(self.font_size), color='white', hovercolor='darkgray')
        font_size_box.on_submit(self.on_font_size_change)
        
        # Слайдеры - первый столбец
        ax_x = plt.axes([start_x, start_y + 0.1, slider_width, slider_height], facecolor='lightblue')
        x_slider = Slider(ax_x, 'Смещение X', -8, 8, valinit=self.x_offset, valstep=1, color='blue')
        x_slider.on_changed(lambda val: self.slider_update('x', val))
        
        # Слайдеры - второй столбец
        second_col_x = start_x + slider_width + 0.02
        
        ax_y = plt.axes([second_col_x, start_y + 0.1, slider_width, slider_height], facecolor='lightcoral')
        y_slider = Slider(ax_y, 'Смещение Y', -8, 8, valinit=self.y_offset, valstep=1, color='red')
        y_slider.on_changed(lambda val: self.slider_update('y', val))
        
        ax_rotate = plt.axes([second_col_x, start_y + 0.05, slider_width, slider_height], facecolor='lightcyan')
        rotate_slider = Slider(ax_rotate, 'Вращение', -15, 15, valinit=self.rotation, color='teal')
        rotate_slider.on_changed(lambda val: self.slider_update('rotate', val))
        
        # Кнопки - третий столбец
        buttons_x = second_col_x + slider_width + 0.02
        
        ax_reset = plt.axes([buttons_x, start_y + 0.1, 0.1, 0.04], facecolor='lightgray')
        reset_button = Button(ax_reset, 'Сброс', hovercolor='gray')
        reset_button.on_clicked(self.reset_parameters)
        
        ax_random = plt.axes([buttons_x, start_y + 0.05, 0.1, 0.04], facecolor='lightgray')
        random_button = Button(ax_random, 'Случайный', hovercolor='gray')
        random_button.on_clicked(self.random_symbol)
        
        # Тестовые кнопки
        ax_test1 = plt.axes([buttons_x, start_y, 0.1, 0.04], facecolor='lightgreen')
        test1_button = Button(ax_test1, 'Тест А', hovercolor='gray')
        test1_button.on_clicked(lambda x: self.test_char('А'))
        
        ax_test2 = plt.axes([buttons_x + 0.11, start_y, 0.1, 0.04], facecolor='lightgreen')
        test2_button = Button(ax_test2, 'Тест 1', hovercolor='gray')
        test2_button.on_clicked(lambda x: self.test_char('1'))
        
        # Сохраняем слайдеры
        self.sliders = {
            'x': x_slider, 'y': y_slider, 'rotate': rotate_slider,
            'char': char_box, 'font_size': font_size_box
        }
        
        # Информация
        info_text = f"МОДЕЛЬ 24x24 | {self.config.num_classes} символов | ДИАГНОСТИЧЕСКИЙ РЕЖИМ"
        self.fig.text(0.5, 0.02, info_text, fontsize=10, ha='center', color='white',
                     bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))
        
        self.fig.patch.set_facecolor('black')
        plt.subplots_adjust(bottom=0.35)
        
        # Первоначальное обновление
        print("🎯 НАЧАЛО РАБОТЫ - ТЕСТОВЫЙ РЕЖИМ")
        self.update_display()

    def slider_update(self, param, value):
        """Обновление параметров при изменении слайдеров"""
        if param == 'x':
            self.x_offset = int(value)
        elif param == 'y':
            self.y_offset = int(value)
        elif param == 'rotate':
            self.rotation = value
        
        print(f"🔧 Параметр {param} изменен на: {value}")
        self.update_display()

    def reset_parameters(self, event):
        """Сброс всех параметров"""
        self.x_offset = 0
        self.y_offset = 0
        self.font_size = 18
        self.rotation = 0
        
        for param, slider in self.sliders.items():
            if param not in ['char', 'font_size']:
                slider.set_val(getattr(self, param))
        
        self.sliders['font_size'].set_val(str(self.font_size))
        print("🔄 Параметры сброшены")
        self.update_display()

    def random_symbol(self, event):
        """Выбор случайного символа"""
        self.current_char = random.choice(self.config.chars)
        self.sliders['char'].set_val(self.current_char)
        print(f"🎲 СЛУЧАЙНЫЙ СИМВОЛ: '{self.current_char}'")
        self.update_display()

    def test_char(self, char):
        """Тест конкретного символа"""
        self.current_char = char
        self.sliders['char'].set_val(char)
        print(f"🧪 ТЕСТ СИМВОЛА: '{char}'")
        self.update_display()

    def run(self):
        """Запуск интерфейса"""
        self.create_interactive_interface()
        plt.show()

def main():
    # Проверяем наличие модели
    model_files = [
        'best_char_model_24x24.pth',
        'final_char_model_24x24.pth',
        'simple_char_model_24x24_epoch3.pth',
        'simple_char_model_24x24_epoch2.pth', 
        'simple_char_model_24x24_epoch1.pth',
    ]
    
    found_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            found_model = model_file
            break
    
    if found_model:
        print(f"✅ Найдена модель: {found_model}")
        tester = InteractiveSymbolTester24x24(found_model)
        tester.run()
    else:
        print("❌ Модели не найдены!")
        # Все равно запускаем для теста
        tester = InteractiveSymbolTester24x24()
        tester.run()

if __name__ == "__main__":
    main()
