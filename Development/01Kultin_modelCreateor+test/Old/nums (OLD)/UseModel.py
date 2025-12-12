# enhanced_digit_app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torchvision.transforms as transforms
import cv2
from scipy import ndimage
import time
import os

# Используем ТОЧНО ТУ ЖЕ архитектуру что и при обучении
class EnhancedDigitRecognizer(nn.Module):
    def __init__(self):
        super(EnhancedDigitRecognizer, self).__init__()
        
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Третий сверточный блок
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Четвертый сверточный блок для лучшего извлечения признаков
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Полносвязные слои с большей емкостью
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Первый блок
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Второй блок
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Третий блок
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Четвертый блок
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = self.dropout1(x)
        
        # Полносвязные слои
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class EnhancedDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Улучшенное распознавание цифр с AI-анализом")
        self.root.geometry("1000x800")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.model = self.load_model()
        self.prediction_history = []
        
        # Трансформы идентичные обучению
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Для рисования - ЧЕРНЫЙ фон, БЕЛЫЕ цифры
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.line_width = 20
        
        # Настройки рисования
        self.drawing_enabled = True
        self.brush_size = 20
        
        self.setup_ui()
        
    def load_model(self):
        """Загрузка улучшенной модели"""
        model_files = ['enhanced_digit_model.pth', 'digit_model.pth']
        model = None
        
        for model_file in model_files:
            try:
                if not os.path.exists(model_file):
                    print(f"❌ Файл {model_file} не найден")
                    continue
                    
                print(f"🔄 Пробуем загрузить модель из {model_file}...")
                model = EnhancedDigitRecognizer()  # Используем правильную архитектуру
                checkpoint = torch.load(model_file, map_location=self.device)
                
                # Проверяем разные форматы сохранения модели
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Загружен model_state_dict")
                else:
                    # Пробуем загрузить напрямую
                    try:
                        model.load_state_dict(checkpoint)
                        print("✅ Загружены прямые веса модели")
                    except:
                        # Пробуем загрузить с исправлением ключей
                        new_state_dict = {}
                        for k, v in checkpoint.items():
                            if k.startswith('module.'):
                                k = k[7:]  # Убираем 'module.' если модель была сохранена с DataParallel
                            new_state_dict[k] = v
                        model.load_state_dict(new_state_dict)
                        print("✅ Загружены веса с исправлением ключей")
                    
                model.to(self.device)
                model.eval()
                print(f"✅ Модель загружена из {model_file}!")
                
                # Покажем точность модели если есть в checkpoint
                if 'test_accuracy' in checkpoint:
                    print(f"🎯 Точность модели: {checkpoint['test_accuracy']:.2f}%")
                
                # Проверяем, что модель работает
                test_input = torch.randn(1, 1, 28, 28).to(self.device)
                with torch.no_grad():
                    test_output = model(test_input)
                print(f"🧪 Тест пройден: выходная форма {test_output.shape}")
                    
                return model
            except Exception as e:
                print(f"❌ Ошибка загрузки {model_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Если не удалось загрузить улучшенную модель, пробуем создать простую
        messagebox.showwarning("Предупреждение", 
            "Не удалось загрузить сохраненную модель!\n\n"
            "Создана новая неподготовленная модель.\n"
            "Сначала обучите модель с помощью CreateModel.py")
        
        # Создаем новую модель
        model = EnhancedDigitRecognizer().to(self.device)
        model.eval()
        return model
    
    def setup_ui(self):
        """Создание улучшенного интерфейса"""
        # Создаем меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить изображение", command=self.load_image)
        file_menu.add_command(label="Сохранить рисунок", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Инструменты", menu=tools_menu)
        tools_menu.add_command(label="Анализ модели", command=self.show_model_analysis)
        tools_menu.add_command(label="История предсказаний", command=self.show_prediction_history)
        tools_menu.add_command(label="Тест confusion 3-9", command=self.test_3_9_confusion)
        tools_menu.add_command(label="Проверить модель", command=self.debug_model)
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Верхняя панель - рисование и управление
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Левая часть - рисование
        draw_frame = ttk.LabelFrame(top_frame, 
                                   text="🎨 Рисуйте БЕЛЫЕ цифры на ЧЕРНОМ фоне", 
                                   padding="10")
        draw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        canvas_frame = ttk.Frame(draw_frame)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, bg='black', cursor="crosshair")
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)
        
        # Правая часть - инструменты
        tools_frame = ttk.LabelFrame(top_frame, text="⚙️ Инструменты", padding="10")
        tools_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Регулятор толщины кисти
        ttk.Label(tools_frame, text="Толщина кисти:").pack(anchor=tk.W)
        self.brush_var = tk.IntVar(value=20)
        brush_scale = ttk.Scale(tools_frame, from_=5, to=40, variable=self.brush_var,
                               command=self.update_brush_size, orient=tk.HORIZONTAL)
        brush_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(tools_frame, textvariable=self.brush_var).pack()
        
        # Кнопки предобработки
        ttk.Button(tools_frame, text="Улучшить контраст", 
                  command=self.enhance_contrast).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="Центрировать цифру", 
                  command=self.center_digit).pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="Применить размытие", 
                  command=self.apply_blur).pack(fill=tk.X, pady=2)
        
        # Основные кнопки
        button_frame = ttk.Frame(draw_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="🔍 Распознать", 
                  command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🧹 Очистить", 
                  command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Авто-тест", 
                  command=self.auto_test).pack(side=tk.LEFT, padx=5)
        
        # Нижняя панель - результаты
        result_frame = ttk.LabelFrame(main_frame, text="📊 Результаты распознавания", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 1: Визуализация
        vis_frame = ttk.Frame(self.notebook)
        self.notebook.add(vis_frame, text="Визуализация")
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, vis_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 2: Уверенность
        confidence_frame = ttk.Frame(self.notebook)
        self.notebook.add(confidence_frame, text="Уверенность")
        
        self.confidence_fig, self.confidence_ax = plt.subplots(figsize=(10, 6))
        self.confidence_canvas = FigureCanvasTkAgg(self.confidence_fig, confidence_frame)
        self.confidence_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 3: Анализ
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="AI Анализ")
        
        self.analysis_text = tk.Text(analysis_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(analysis_frame, command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def debug_model(self):
        """Отладочная информация о модели"""
        if self.model:
            info = f"🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ О МОДЕЛИ:\n\n"
            info += f"• Тип модели: {type(self.model).__name__}\n"
            info += f"• Устройство: {self.device}\n"
            info += f"• Параметры: {sum(p.numel() for p in self.model.parameters()):,}\n"
            
            # Проверяем прямолинейность модели
            try:
                test_input = torch.randn(1, 1, 28, 28).to(self.device)
                with torch.no_grad():
                    test_output = self.model(test_input)
                info += f"• Тест пройден: ✓\n"
                info += f"• Выходная форма: {test_output.shape}\n"
            except Exception as e:
                info += f"• Тест не пройден: {e}\n"
            
            messagebox.showinfo("Отладка модели", info)
        else:
            messagebox.showerror("Ошибка", "Модель не загружена!")
    
    def update_brush_size(self, value):
        self.brush_size = int(float(value))
        self.line_width = self.brush_size
    
    def start_draw(self, event):
        if self.drawing_enabled:
            self.last_x = event.x
            self.last_y = event.y
    
    def draw_line(self, event):
        if self.drawing_enabled and self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.line_width, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], 
                          fill=255, width=self.line_width)
            self.last_x = event.x
            self.last_y = event.y
    
    def reset_draw(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.status_var.set("Холст очищен")
        self.update_visualizations()
    
    def enhance_contrast(self):
        """Улучшение контраста изображения"""
        if np.array(self.image).max() > 0:
            img_array = np.array(self.image)
            img_array = cv2.equalizeHist(img_array)
            self.image = Image.fromarray(img_array)
            self.redraw_canvas()
            self.status_var.set("Контраст улучшен")
    
    def center_digit(self):
        """Центрирование цифры на изображении"""
        if np.array(self.image).max() > 0:
            img_array = np.array(self.image)
            
            # Находим bounding box цифры
            coords = np.column_stack(np.where(img_array > 0))
            if len(coords) > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                
                # Вычисляем смещение для центрирования
                center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
                digit_center_x = (x0 + x1) // 2
                digit_center_y = (y0 + y1) // 2
                
                shift_x = center_x - digit_center_x
                shift_y = center_y - digit_center_y
                
                # Сдвигаем изображение
                shifted = ndimage.shift(img_array, [shift_y, shift_x], mode='constant', cval=0)
                self.image = Image.fromarray(shifted.astype(np.uint8))
                self.redraw_canvas()
                self.status_var.set("Цифра центрирована")
    
    def apply_blur(self):
        """Применение размытия Гаусса"""
        if np.array(self.image).max() > 0:
            img_array = np.array(self.image)
            blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
            self.image = Image.fromarray(blurred)
            self.redraw_canvas()
            self.status_var.set("Применено размытие")
    
    def redraw_canvas(self):
        """Перерисовывает canvas на основе текущего изображения"""
        self.canvas.delete("all")
        img_temp = self.image.copy()
        img_temp = img_temp.resize((280, 280), Image.LANCZOS)
        
        # Конвертируем в формат для tkinter
        img_tk = ImageTk.PhotoImage(img_temp)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Сохраняем ссылку
    
    def preprocess_image(self):
        """Предобработка изображения с дополнительными возможностями"""
        img_28x28 = self.image.resize((28, 28), Image.LANCZOS)
        img_tensor = self.transform(img_28x28).unsqueeze(0)
        return img_tensor.to(self.device), np.array(img_28x28)
    
    def predict_digit(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        if np.array(self.image).max() == 0:
            messagebox.showwarning("Предупреждение", "Сначала нарисуйте цифру!")
            return
        
        try:
            start_time = time.time()
            img_tensor, img_array = self.preprocess_image()
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            digit = predicted.item()
            conf_value = confidence.item() * 100
            inference_time = (time.time() - start_time) * 1000
            
            # Сохраняем в историю
            self.prediction_history.append({
                'digit': digit,
                'confidence': conf_value,
                'time': inference_time,
                'timestamp': time.time()
            })
            
            all_probs = probabilities.cpu().numpy()[0] * 100
            
            self.update_visualizations(img_array, digit, conf_value, all_probs, inference_time)
            self.update_confidence_chart(all_probs, digit)
            self.generate_analysis(digit, conf_value, all_probs, inference_time)
            
            self.status_var.set(f"Распознано: {digit} ({conf_value:.1f}%) за {inference_time:.1f}мс")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_visualizations(self, img_array=None, digit=None, confidence=None, probs=None, inference_time=None):
        """Обновление визуализаций"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        if img_array is not None:
            # 1. Ваше изображение
            self.ax1.imshow(img_array, cmap='gray')
            self.ax1.set_title("📷 Ваше изображение (28x28)")
            self.ax1.axis('off')
            
            # 2. Нормализованное
            img_normalized = (img_array - 0.1307) / 0.3081
            self.ax2.imshow(img_normalized, cmap='gray')
            self.ax2.set_title("⚡ После нормализации")
            self.ax2.axis('off')
            
            # 3. Heatmap активаций
            self.ax3.imshow(img_array, cmap='hot', alpha=0.7)
            self.ax3.set_title("🔥 Heatmap интенсивности")
            self.ax3.axis('off')
            
            # 4. Информация
            self.ax4.axis('off')
            info_text = f"🤖 РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:\n\n"
            info_text += f"🔢 Цифра: {digit}\n"
            info_text += f"🎯 Уверенность: {confidence:.1f}%\n"
            info_text += f"⏱ Время: {inference_time:.1f}мс\n\n"
            
            if confidence > 95:
                info_text += "✅ ВЫСОКАЯ УВЕРЕННОСТЬ"
            elif confidence > 80:
                info_text += "👍 ХОРОШАЯ УВЕРЕННОСТЬ"
            elif confidence > 60:
                info_text += "⚠️  СРЕДНЯЯ УВЕРЕННОСТЬ"
            else:
                info_text += "❌ НИЗКАЯ УВЕРЕННОСТЬ"
                
            # Анализ confusion 3-9
            if digit in [3, 9]:
                prob_3 = probs[3]
                prob_9 = probs[9]
                if abs(prob_3 - prob_9) < 20:
                    info_text += f"\n\n🔍 Внимание: цифра {digit} имеет схожие\nвероятности с {'3' if digit == 9 else '9'}"
            
            self.ax4.text(0.1, 0.9, info_text, transform=self.ax4.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.canvas_plot.draw()
    
    def update_confidence_chart(self, probs, predicted_digit):
        """Обновление графика уверенности"""
        self.confidence_ax.clear()
        
        digits = list(range(10))
        colors = ['red' if i == predicted_digit else 'skyblue' for i in range(10)]
        
        bars = self.confidence_ax.bar(digits, probs, color=colors, alpha=0.7)
        self.confidence_ax.set_title("📊 Вероятности распознавания цифр", fontsize=14)
        self.confidence_ax.set_xlabel("Цифра")
        self.confidence_ax.set_ylabel("Вероятность (%)")
        self.confidence_ax.grid(True, alpha=0.3)
        self.confidence_ax.set_ylim(0, 100)
        
        # Добавляем значения на столбцы
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            self.confidence_ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
        
        self.confidence_canvas.draw()
    
    def generate_analysis(self, digit, confidence, probs, inference_time):
        """Генерация AI-анализа"""
        analysis = "🤖 AI-АНАЛИЗ РАСПОЗНАВАНИЯ:\n\n"
        analysis += f"• Распознанная цифра: {digit}\n"
        analysis += f"• Уровень уверенности: {confidence:.1f}%\n"
        analysis += f"• Время обработки: {inference_time:.1f} мс\n\n"
        
        # Анализ уверенности
        if confidence > 95:
            analysis += "✅ Отличное распознавание! Модель очень уверена.\n"
        elif confidence > 80:
            analysis += "👍 Хорошее распознавание. Модель уверена в результате.\n"
        elif confidence > 60:
            analysis += "⚠️  Средняя уверенность. Рассмотрите перерисовку.\n"
        else:
            analysis += "❌ Низкая уверенность. Попробуйте нарисовать четче.\n"
        
        # Анализ альтернативных вариантов
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) > 1:
            second_best = sorted_probs[1]
            if second_best[1] > 20:  # Если второй вариант имеет значительную вероятность
                analysis += f"\n🔍 Альтернативный вариант: {second_best[0]} ({second_best[1]:.1f}%)\n"
        
        # Специфический анализ для 3 и 9
        if digit in [3, 9]:
            other_digit = 9 if digit == 3 else 3
            analysis += f"\n🎯 Особое внимание: цифры {digit} и {other_digit} часто путаются\n"
            analysis += f"   • Вероятность {digit}: {probs[digit]:.1f}%\n"
            analysis += f"   • Вероятность {other_digit}: {probs[other_digit]:.1f}%\n"
            analysis += f"   • Разница: {abs(probs[digit] - probs[other_digit]):.1f}%\n"
        
        # Рекомендации
        analysis += "\n💡 Рекомендации:\n"
        if confidence < 80:
            analysis += "• Попробуйте нарисовать цифру четче и крупнее\n"
            analysis += "• Используйте инструмент 'Центрировать цифру'\n"
            analysis += "• Убедитесь, что цифра не касается краев\n"
        else:
            analysis += "• Отличное качество рисунка!\n"
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, analysis)
    
    def load_image(self):
        """Загрузка изображения из файла"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            try:
                loaded_image = Image.open(file_path).convert('L')
                loaded_image = loaded_image.resize((280, 280), Image.LANCZOS)
                self.image = loaded_image
                self.redraw_canvas()
                self.status_var.set(f"Изображение загружено: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
    
    def save_image(self):
        """Сохранение рисунка"""
        if np.array(self.image).max() > 0:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                self.image.save(file_path)
                self.status_var.set(f"Изображение сохранено: {file_path}")
    
    def auto_test(self):
        """Автоматический тест на нескольких примерах"""
        test_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        results = []
        
        for digit in test_digits:
            self.clear_canvas()
            messagebox.showinfo("Авто-тест", 
                              f"Нарисуйте цифру {digit} для тестирования")
            self.predict_digit()
            if self.prediction_history:
                results.append(self.prediction_history[-1])
        
        # Анализ результатов
        if results:
            correct = sum(1 for r in results if r['digit'] == test_digits[results.index(r)])
            accuracy = correct / len(results) * 100
            
            messagebox.showinfo("Результаты авто-теста",
                              f"Точность: {accuracy:.1f}%\n"
                              f"Правильно: {correct}/{len(results)}")
    
    def show_model_analysis(self):
        """Показать анализ модели"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            analysis = f"📊 АНАЛИЗ МОДЕЛИ:\n\n"
            analysis += f"• Архитектура: {type(self.model).__name__}\n"
            analysis += f"• Параметры: {total_params:,}\n"
            analysis += f"• Устройство: {self.device}\n"
            analysis += f"• История предсказаний: {len(self.prediction_history)} записей\n"
            
            if self.prediction_history:
                avg_confidence = np.mean([r['confidence'] for r in self.prediction_history])
                avg_time = np.mean([r['time'] for r in self.prediction_history])
                analysis += f"• Средняя уверенность: {avg_confidence:.1f}%\n"
                analysis += f"• Среднее время: {avg_time:.1f} мс\n"
            
            messagebox.showinfo("Анализ модели", analysis)
    
    def show_prediction_history(self):
        """Показать историю предсказаний"""
        if not self.prediction_history:
            messagebox.showinfo("История", "История предсказаний пуста")
            return
        
        history_text = "📈 ИСТОРИЯ ПРЕДСКАЗАНИЙ:\n\n"
        for i, pred in enumerate(self.prediction_history[-10:], 1):  # Последние 10
            history_text += f"{i}. Цифра: {pred['digit']} | Уверенность: {pred['confidence']:.1f}% | Время: {pred['time']:.1f}мс\n"
        
        messagebox.showinfo("История предсказаний", history_text)
    
    def test_3_9_confusion(self):
        """Специальный тест для анализа confusion между 3 и 9"""
        messagebox.showinfo("Тест 3-9", 
                          "Нарисуйте цифру 3 или 9 для анализа специфической проблемы confusion")
        self.predict_digit()

# Добавляем импорт для redraw_canvas
from PIL import ImageTk

def main():
    root = tk.Tk()
    app = EnhancedDigitApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
