# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import cv2

from model import load_model
from preprocessing import advanced_preprocessing_improved
from segmentation import segment_characters_enhanced
from utils import extract_char_images_with_padding
from recognition import recognize_characters_enhanced

class OCRAppEnhanced:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Распознавание текста - УЛУЧШЕННАЯ ВЕРСИЯ")
        self.root.geometry("1400x900")
        
        # ИНИЦИАЛИЗИРУЕМ ПЕРЕМЕННЫЕ ПЕРЕД ИСПОЛЬЗОВАНИЕМ
        self.log_messages = []  # ДОБАВЛЕНО ПЕРВЫМ!
        
        # Переменные
        self.original_image = None
        self.preprocessed_image = None
        self.char_images = []
        self.recognized_text = ""
        self.model = None
        self.device = None
        self.config = None
        self.boxes = []
        self.alternative_chars = []
        
        # Параметры обработки
        self.processing_params = {
            'preprocess_blur': 51,
            'preprocess_clip_limit': 3.0,
            'segment_min_area': 10,
            'padding_ratio': 0.2
        }
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Загружаем модель ПОСЛЕ создания интерфейса
        self.load_model()
        
        # Логируем старт
        self.log("=" * 70)
        self.log("🎯 OCR СИСТЕМА РАСПОЗНАВАНИЯ ТЕКСТА - УЛУЧШЕННАЯ ВЕРСИЯ")
        self.log("=" * 70)
        self.log("✅ Приложение запущено")
    
    def log(self, message):
        """Логирование сообщений"""
        self.log_messages.append(message)
        print(message)
        
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
    
    def load_model(self):
        """Загрузка модели при запуске"""
        try:
            self.model, self.device, self.config = load_model()
            self.log("✅ Модель загружена успешно")
        except Exception as e:
            self.log(f"⚠️  Модель не загружена: {e}")
    
    def create_widgets(self):
        """Создание виджетов интерфейса"""
        # Основной контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== ПАНЕЛЬ УПРАВЛЕНИЯ =====
        control_frame = ttk.LabelFrame(main_container, text="УПРАВЛЕНИЕ", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Первый ряд кнопок
        btn_row1 = ttk.Frame(control_frame)
        btn_row1.pack(fill=tk.X, pady=5)
        
        self.btn_open = ttk.Button(btn_row1, text="1. 📁 Открыть изображение", 
                                  command=self.open_image, width=22)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_prepre = ttk.Button(btn_row1, text="2. 🎯 Улучшенная предобработка", 
                                    command=self.prepreprocess_image, 
                                    state='disabled', width=25)
        self.btn_prepre.pack(side=tk.LEFT, padx=5)
        
        self.btn_segment = ttk.Button(btn_row1, text="3. ✂️ Улучшенная сегментация", 
                                     command=self.segment_image, 
                                     state='disabled', width=25)
        self.btn_segment.pack(side=tk.LEFT, padx=5)
        
        self.btn_recognize = ttk.Button(btn_row1, text="4. 🧠 Распознать текст", 
                                       command=self.recognize, 
                                       state='disabled', width=22)
        self.btn_recognize.pack(side=tk.LEFT, padx=5)
        
        # Второй ряд кнопок
        btn_row2 = ttk.Frame(control_frame)
        btn_row2.pack(fill=tk.X, pady=5)
        
        self.btn_copy = ttk.Button(btn_row2, text="📋 Копировать текст", 
                                  command=self.copy_text, width=20)
        self.btn_copy.pack(side=tk.LEFT, padx=5)
        
        self.btn_debug = ttk.Button(btn_row2, text="🔧 Показать отладку", 
                                   command=self.show_debug_info, width=20)
        self.btn_debug.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = ttk.Button(btn_row2, text="🔄 Сбросить всё", 
                                   command=self.reset_processing, width=20)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        self.btn_manual = ttk.Button(btn_row2, text="✏️ Ручная коррекция", 
                                    command=self.manual_correction, width=20)
        self.btn_manual.pack(side=tk.LEFT, padx=5)
        
        # ===== ОБЛАСТЬ ДЛЯ ИЗОБРАЖЕНИЙ И ТЕКСТА =====
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - изображения
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Правая панель - текст и лог
        right_panel = ttk.Frame(content_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)
        
        # ===== ИЗОБРАЖЕНИЯ =====
        self.image_notebook = ttk.Notebook(left_panel)
        self.image_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка с оригиналом
        self.original_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.original_tab, text="Оригинал")
        
        # Вкладка с предобработкой
        self.preprocessed_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.preprocessed_tab, text="Предобработка")
        
        # Вкладка с сегментацией
        self.segmented_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.segmented_tab, text="Сегментация")
        
        # Вкладка с символами
        self.characters_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.characters_tab, text="Символы")
        
        # ===== ТЕКСТ И ЛОГ =====
        # Текстовая область
        text_frame = ttk.LabelFrame(right_panel, text="РАСПОЗНАННЫЙ ТЕКСТ", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_widget = scrolledtext.ScrolledText(text_frame, height=15, 
                                                    font=('Courier New', 11))
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Лог
        log_frame = ttk.LabelFrame(right_panel, text="ЛОГ ОБРАБОТКИ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, 
                                                 font=('Courier New', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Статистика
        stats_frame = ttk.Frame(right_panel)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_var = tk.StringVar(value="Ожидание обработки...")
        ttk.Label(stats_frame, textvariable=self.stats_var, 
                 font=('Arial', 9)).pack()
        
        # Статусная строка
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе. Откройте изображение.")
        ttk.Label(main_container, textvariable=self.status_var, 
                 relief=tk.SUNKEN, anchor=tk.W).pack(
                     fill=tk.X, pady=(10, 0))
        
        # Инициализация вкладок
        self.init_tabs()
    
    def init_tabs(self):
        """Инициализация вкладок изображений"""
        for tab in [self.original_tab, self.preprocessed_tab, 
                   self.segmented_tab, self.characters_tab]:
            label = ttk.Label(tab, text="Изображение не загружено", 
                             font=('Arial', 12), foreground='gray')
            label.pack(expand=True)
    
    def display_image_on_tab(self, image, title, tab):
        """Отображение изображения на указанной вкладке"""
        # Очищаем вкладку
        for widget in tab.winfo_children():
            widget.destroy()
        
        if image is None:
            label = ttk.Label(tab, text=title, 
                             font=('Arial', 12), foreground='gray')
            label.pack(expand=True)
            return
        
        # Проверяем, что это numpy массив
        if not isinstance(image, np.ndarray):
            label = ttk.Label(tab, text=f"{title} (ошибка: не изображение)", 
                             font=('Arial', 12), foreground='red')
            label.pack(expand=True)
            self.log(f"⚠️  Ошибка отображения: передан не numpy массив для {title}")
            return
        
        try:
            # Конвертируем для matplotlib
            if len(image.shape) == 3:
                # Цветное изображение
                if image.shape[2] == 3:  # BGR
                    display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:  # BGRA
                    display_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    display_img = display_img[:, :, :3]  # Убираем альфа-канал
                else:
                    display_img = image[:, :, :3]  # Берем только первые 3 канала
                cmap = None
            elif len(image.shape) == 2:
                # Grayscale изображение
                display_img = image
                cmap = 'gray'
            else:
                label = ttk.Label(tab, text=f"{title} (неподдерживаемый формат)", 
                                 font=('Arial', 12), foreground='red')
                label.pack(expand=True)
                return
            
            # Создаем фигуру
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(display_img, cmap=cmap)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Встраиваем в tkinter
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохраняем ссылку на canvas
            tab.canvas = canvas
            
        except Exception as e:
            label = ttk.Label(tab, text=f"{title} (ошибка отображения: {str(e)})", 
                             font=('Arial', 12), foreground='red')
            label.pack(expand=True)
            self.log(f"❌ Ошибка отображения {title}: {e}")
    
    def open_image(self):
        """Открытие изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение с текстом",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Загрузка: {os.path.basename(file_path)}")
                self.log(f"\n📁 ЗАГРУЗКА ИЗОБРАЖЕНИЯ: {file_path}")
                
                # Сохраняем путь
                self.image_path = file_path
                
                # Загружаем оригинал
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Не удалось загрузить изображение")
                
                # Показываем оригинал
                self.display_image_on_tab(
                    self.original_image, 
                    f"ОРИГИНАЛ: {os.path.basename(file_path)}", 
                    self.original_tab
                )
                
                # Активируем кнопку предобработки
                self.btn_prepre['state'] = 'normal'
                self.btn_open['state'] = 'disabled'
                
                info = f"Размер: {self.original_image.shape[1]}x{self.original_image.shape[0]}, "
                info += f"Каналы: {self.original_image.shape[2]}"
                self.log(f"✅ Изображение загружено: {info}")
                self.status_var.set(f"Загружено: {os.path.basename(file_path)}")
                
                # Переключаем на вкладку оригинального изображения
                self.image_notebook.select(self.original_tab)
                
            except Exception as e:
                error_msg = f"Ошибка загрузки изображения: {e}"
                self.log(f"❌ {error_msg}")
                messagebox.showerror("Ошибка", error_msg)
                self.status_var.set("Ошибка загрузки")
    
    def prepreprocess_image(self):
        """Улучшенная предобработка"""
        if not hasattr(self, 'image_path'):
            return
        
        try:
            self.status_var.set("Улучшенная предобработка...")
            self.log("\n🎯 ЗАПУСК УЛУЧШЕННОЙ ПРЕДОБРАБОТКИ")
            
            # Выполняем улучшенную предобработку
            original, preprocessed = advanced_preprocessing_improved(self.image_path, show_steps=False)
            
            # Сохраняем результат
            self.original_image = original
            self.preprocessed_image = preprocessed
            
            # Показываем результат
            self.display_image_on_tab(
                self.preprocessed_image, 
                "ПОСЛЕ УЛУЧШЕННОЙ ПРЕДОБРАБОТКИ", 
                self.preprocessed_tab
            )
            
            # Активируем кнопку сегментации
            self.btn_segment['state'] = 'normal'
            self.btn_prepre['state'] = 'disabled'
            
            self.log("✅ Предобработка завершена")
            if isinstance(self.preprocessed_image, np.ndarray):
                white_px = np.sum(self.preprocessed_image == 255)
                total_px = self.preprocessed_image.size
                self.log(f"   Белый текст на черном фоне: {white_px:,} пикселей ({white_px/total_px:.1%})")
            
            self.status_var.set("Предобработка завершена. Готово к сегментации.")
            
            # Переключаем на вкладку предобработанного изображения
            self.image_notebook.select(self.preprocessed_tab)
            
        except Exception as e:
            error_msg = f"Ошибка предобработки: {e}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Ошибка", error_msg)
            self.status_var.set("Ошибка предобработки")
    
    def segment_image(self):
        """Улучшенная сегментация символов"""
        if self.preprocessed_image is None:
            messagebox.showwarning("Внимание", "Сначала выполните предобработку!")
            return
        
        try:
            self.status_var.set("Улучшенная сегментация...")
            self.log("\n✂️ ЗАПУСК УЛУЧШЕННОЙ СЕГМЕНТАЦИИ")
            
            # ПРОВЕРКА: убедимся, что preprocessed_image - это numpy array
            if not isinstance(self.preprocessed_image, np.ndarray):
                self.log("❌ Ошибка: предобработанное изображение не является numpy массивом")
                messagebox.showerror("Ошибка", "Предобработанное изображение не является numpy массивом")
                return
            
            # Используем улучшенную сегментацию
            self.boxes = segment_characters_enhanced(
                self.preprocessed_image, 
                self.original_image,
                debug_mode=True
            )
            
            if not self.boxes:
                self.log("⚠️  Символы не найдены! Попробуйте другие параметры.")
                messagebox.showwarning("Внимание", "Символы не найдены!")
                return
            
            # СОЗДАЕМ КОПИЮ оригинального изображения для отрисовки
            if self.original_image is not None and isinstance(self.original_image, np.ndarray):
                result_img = self.original_image.copy()
            else:
                # Если нет оригинального, используем предобработанное
                if isinstance(self.preprocessed_image, np.ndarray):
                    # Конвертируем в цветное если нужно
                    if len(self.preprocessed_image.shape) == 2:
                        result_img = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_GRAY2BGR)
                    else:
                        result_img = self.preprocessed_image.copy()
                else:
                    self.log("❌ Нет изображения для отрисовки")
                    return
            
            # Рисуем bounding boxes
            for i, (x, y, w, h) in enumerate(self.boxes):
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_img, str(i+1), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Отображаем результат
            self.display_image_on_tab(
                result_img, 
                f"СЕГМЕНТАЦИЯ: найдено {len(self.boxes)} символов", 
                self.segmented_tab
            )
            
            # Извлекаем изображения символов
            self.char_images = extract_char_images_with_padding(
                self.preprocessed_image,  # Используем предобработанное изображение!
                self.boxes, 
                padding_ratio=self.processing_params['padding_ratio']
            )
            
            # Активируем кнопку распознавания
            self.btn_recognize['state'] = 'normal'
            self.btn_segment['state'] = 'disabled'
            
            # Показываем символы
            self.show_characters_grid()
            
            # Обновляем статистику
            stats = f"Символов: {len(self.boxes)} | "
            if self.boxes:
                widths = [w for _, _, w, _ in self.boxes]
                heights = [h for _, _, _, h in self.boxes]  # ← Исправлено: self.boxes вместо final_boxes
                stats += f"Размеры: {np.mean(widths):.1f}x{np.mean(heights):.1f} px"

            
            self.stats_var.set(stats)
            self.log(f"✅ Сегментация завершена: {len(self.boxes)} символов")
            self.status_var.set(f"Найдено {len(self.boxes)} символов. Готово к распознаванию.")
            
            # Переключаем на вкладку сегментированного изображения
            self.image_notebook.select(self.segmented_tab)
            
        except Exception as e:
            error_msg = f"Ошибка сегментации: {e}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Ошибка", error_msg)
            self.status_var.set("Ошибка сегментации")
    
    def show_characters_grid(self, cols=12):
        """Показ символов в виде сетки"""
        if not self.char_images:
            return
        
        # Очищаем вкладку
        for widget in self.characters_tab.winfo_children():
            widget.destroy()
        
        n = len(self.char_images)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.0, rows*1.2))
        fig.suptitle(f"ВЫДЕЛЕННЫЕ СИМВОЛЫ ({n} шт.)", fontsize=14, fontweight='bold')
        
        # Если только одна строка
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            
            if i < n:
                ax = axes[row, col] if rows > 1 else axes[col]
                char_img, _ = self.char_images[i]
                
                # Проверяем, что char_img - это numpy массив
                if isinstance(char_img, np.ndarray):
                    ax.imshow(char_img, cmap='gray')
                else:
                    # Если не массив, создаем черное изображение
                    ax.imshow(np.zeros((24, 24), dtype=np.uint8), cmap='gray')
                
                ax.set_title(f"{i+1}", fontsize=8, fontweight='bold')
                ax.axis('off')
            else:
                if rows > 1:
                    axes[row, col].axis('off')
                else:
                    axes[col].axis('off')
        
        plt.tight_layout()
        
        # Встраиваем в tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.characters_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем ссылку
        self.characters_tab.canvas = canvas
    
    def recognize(self):
        """Распознавание текста"""
        if not self.char_images:
            messagebox.showwarning("Внимание", "Сначала выполните сегментацию!")
            return
        
        if self.model is None:
            messagebox.showwarning("Внимание", "Модель не загружена!")
            return
        
        try:
            self.status_var.set("Улучшенное распознавание...")
            self.log("\n🧠 ЗАПУСК УЛУЧШЕННОГО РАСПОЗНАВАНИЯ")
            
            # Распознавание с альтернативами
            text, chars, confs, processed_imgs, alternatives = \
                recognize_characters_enhanced(
                    self.model, self.device, self.config, self.char_images)
            
            # Сохраняем результаты
            self.recognized_text = text
            self.recognized_chars = chars
            self.confidences = confs
            self.alternative_chars = alternatives
            
            # Отображаем в текстовом поле
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, text)
            
            # Показываем символы с распознаванием
            self.show_recognized_characters_grid(chars, confs, processed_imgs, alternatives)
            
            # Обновляем статистику
            avg_conf = np.mean(confs) if confs else 0
            low_conf_count = sum(1 for c in confs if c < 0.5)
            
            stats = (f"Символов: {len(chars)} | "
                    f"Уверенность: {avg_conf:.2%} | "
                    f"Низкая уверенность: {low_conf_count}")
            
            self.stats_var.set(stats)
            self.log(f"✅ Распознавание завершено: {len(chars)} символов")
            self.log(f"   Средняя уверенность: {avg_conf:.2%}")
            
            if low_conf_count > 0:
                self.log(f"   ⚠️  Символов с низкой уверенностью (<50%): {low_conf_count}")
            
            # Активируем кнопку копирования
            self.btn_copy['state'] = 'normal'
            self.btn_recognize['state'] = 'disabled'
            
            self.status_var.set(f"Распознано {len(chars)} символов. Средняя уверенность: {avg_conf:.2%}")
            
            # Переключаем на вкладку символов
            self.image_notebook.select(self.characters_tab)
            
        except Exception as e:
            error_msg = f"Ошибка распознавания: {e}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Ошибка", error_msg)
            self.status_var.set("Ошибка распознавания")
    
    def show_recognized_characters_grid(self, chars, confs, images, alternatives, cols=12):
        """Показ распознанных символов с уверенностью"""
        if not images:
            return
        
        # Очищаем вкладку
        for widget in self.characters_tab.winfo_children():
            widget.destroy()
        
        n = len(images)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.0, rows*1.3))
        fig.suptitle(f"РАСПОЗНАННЫЕ СИМВОЛЫ ({n} шт.)", fontsize=14, fontweight='bold')
        
        # Если только одна строка
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            
            if i < n:
                ax = axes[row, col] if rows > 1 else axes[col]
                
                # Проверяем, что image - это numpy массив
                if isinstance(images[i], np.ndarray):
                    ax.imshow(images[i], cmap='gray')
                else:
                    # Если не массив, создаем черное изображение
                    ax.imshow(np.zeros((24, 24), dtype=np.uint8), cmap='gray')
                
                # Цвет в зависимости от уверенности
                if confs[i] > 0.8:
                    color = 'green'
                elif confs[i] > 0.5:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Отображаем основной символ и уверенность
                title = f"{chars[i]}\n{confs[i]:.2%}"
                
                # Добавляем альтернативы если уверенность низкая
                if confs[i] < 0.7 and alternatives[i]:
                    alt_text = "/".join([f"{alt[0]}" for alt in alternatives[i][:2]])
                    title += f"\n({alt_text})"
                
                ax.set_title(title, fontsize=8, color=color, fontweight='bold')
                ax.axis('off')
            else:
                if rows > 1:
                    axes[row, col].axis('off')
                else:
                    axes[col].axis('off')
        
        plt.tight_layout()
        
        # Встраиваем в tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.characters_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем ссылку
        self.characters_tab.canvas = canvas
    
    def copy_text(self):
        """Копирование текста в буфер обмена"""
        if self.recognized_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.recognized_text)
            
            self.log("\n📋 Текст скопирован в буфер обмена")
            self.status_var.set("Текст скопирован в буфер обмена")
            
            # Предложение сохранить
            response = messagebox.askyesno("Сохранение", 
                                          "Текст скопирован. Хотите также сохранить в файл?")
            if response:
                self.save_text_to_file()
        else:
            messagebox.showwarning("Внимание", "Нет текста для копирования")
    
    def save_text_to_file(self):
        """Сохранение текста в файл"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.recognized_text)
                
                self.log(f"💾 Текст сохранен в: {file_path}")
                messagebox.showinfo("Сохранено", f"Текст сохранен в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
    
    def reset_processing(self):
        """Сброс всей обработки"""
        self.log("\n🔄 СБРОС ВСЕЙ ОБРАБОТКИ")
        
        self.preprocessed_image = None
        self.char_images = []
        self.recognized_text = ""
        self.boxes = []
        self.alternative_chars = []
        
        # Сбрасываем кнопки
        self.btn_prepre['state'] = 'disabled'
        self.btn_segment['state'] = 'disabled'
        self.btn_recognize['state'] = 'disabled'
        self.btn_copy['state'] = 'disabled'
        self.btn_open['state'] = 'normal'
        
        # Очищаем текстовое поле
        self.text_widget.delete(1.0, tk.END)
        self.stats_var.set("Ожидание обработки...")
        self.status_var.set("Готов к загрузке нового изображения")
        
        # Очищаем вкладки изображений
        self.init_tabs()
        
        # Переключаем на вкладку оригинального изображения
        self.image_notebook.select(self.original_tab)
        
        self.log("✅ Состояние сброшено. Готово к новой обработке.")
    
    def show_debug_info(self):
        """Показать отладочную информацию"""
        debug_info = []
        debug_info.append("=" * 50)
        debug_info.append("ОТЛАДОЧНАЯ ИНФОРМАЦИЯ")
        debug_info.append("=" * 50)
        
        debug_info.append(f"\n📊 СТАТУС МОДЕЛИ:")
        debug_info.append(f"  Модель загружена: {'Да' if self.model is not None else 'Нет'}")
        if self.model:
            debug_info.append(f"  Устройство: {self.device}")
            debug_info.append(f"  Классов в модели: {self.config.num_classes if self.config else 'N/A'}")
        
        debug_info.append(f"\n📁 ДАННЫЕ ИЗОБРАЖЕНИЯ:")
        if hasattr(self, 'original_image'):
            if isinstance(self.original_image, np.ndarray):
                debug_info.append(f"  Оригинальное: {self.original_image.shape}")
            else:
                debug_info.append(f"  Оригинальное: Не numpy массив (тип: {type(self.original_image)})")
        else:
            debug_info.append(f"  Оригинальное: Не загружено")
        
        if self.preprocessed_image is not None:
            if isinstance(self.preprocessed_image, np.ndarray):
                debug_info.append(f"  Предобработанное: {self.preprocessed_image.shape}")
                white_px = np.sum(self.preprocessed_image == 255)
                black_px = np.sum(self.preprocessed_image == 0)
                total_px = white_px + black_px
                if total_px > 0:
                    debug_info.append(f"  Белых пикселей: {white_px:,} ({white_px/total_px:.1%})")
                    debug_info.append(f"  Черных пикселей: {black_px:,} ({black_px/total_px:.1%})")
            else:
                debug_info.append(f"  Предобработанное: Не numpy массив (тип: {type(self.preprocessed_image)})")
        
        debug_info.append(f"\n✂️ СЕГМЕНТАЦИЯ:")
        debug_info.append(f"  Найдено символов: {len(self.boxes)}")
        if self.boxes:
            widths = [w for _, _, w, _ in self.boxes]
            heights = [h for _, _, _, h in self.boxes]
            debug_info.append(f"  Средняя ширина: {np.mean(widths):.1f} px")
            debug_info.append(f"  Средняя высота: {np.mean(heights):.1f} px")
            debug_info.append(f"  Min-Max ширина: {np.min(widths)}-{np.max(widths)} px")
            debug_info.append(f"  Min-Max высота: {np.min(heights)}-{np.max(heights)} px")
        
        debug_info.append(f"\n🧠 РАСПОЗНАВАНИЕ:")
        if hasattr(self, 'recognized_chars'):
            debug_info.append(f"  Распознано символов: {len(self.recognized_chars)}")
            if hasattr(self, 'confidences') and self.confidences:
                avg_conf = np.mean(self.confidences)
                low_conf = sum(1 for c in self.confidences if c < 0.5)
                debug_info.append(f"  Средняя уверенность: {avg_conf:.2%}")
                debug_info.append(f"  Низкая уверенность (<50%): {low_conf}")
        
        if hasattr(self, 'recognized_text') and self.recognized_text:
            debug_info.append(f"\n📝 ТЕКСТ ({len(self.recognized_text)} символов):")
            debug_info.append(f"  '{self.recognized_text[:100]}...'" 
                           if len(self.recognized_text) > 100 else f"  '{self.recognized_text}'")
        
        debug_info.append(f"\n⚙️ ПАРАМЕТРЫ ОБРАБОТКИ:")
        for key, value in self.processing_params.items():
            debug_info.append(f"  {key}: {value}")
        
        # Показываем в отдельном окне
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Отладочная информация")
        debug_window.geometry("600x700")
        
        text_widget = scrolledtext.ScrolledText(debug_window, width=70, height=40)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for line in debug_info:
            text_widget.insert(tk.END, line + "\n")
        
        text_widget.config(state='disabled')
    
    def manual_correction(self):
        """Ручная коррекция распознанного текста"""
        if not hasattr(self, 'recognized_text') or not self.recognized_text:
            messagebox.showwarning("Внимание", "Сначала выполните распознавание!")
            return
        
        correction_window = tk.Toplevel(self.root)
        correction_window.title("Ручная коррекция текста")
        correction_window.geometry("800x600")
        
        # Текущий текст
        ttk.Label(correction_window, text="Текущий текст:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        current_text = scrolledtext.ScrolledText(correction_window, height=10)
        current_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        current_text.insert(1.0, self.recognized_text)
        current_text.config(state='disabled')
        
        # Поле для редактирования
        ttk.Label(correction_window, text="Исправленный текст:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        corrected_text = scrolledtext.ScrolledText(correction_window, height=10)
        corrected_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        corrected_text.insert(1.0, self.recognized_text)
        
        # Кнопки
        btn_frame = ttk.Frame(correction_window)
        btn_frame.pack(pady=10)
        
        def apply_correction():
            new_text = corrected_text.get(1.0, tk.END).strip()
            self.recognized_text = new_text
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, new_text)
            self.log("\n✏️  Текст откорректирован вручную")
            correction_window.destroy()
        
        ttk.Button(btn_frame, text="Применить исправления", 
                  command=apply_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", 
                  command=correction_window.destroy).pack(side=tk.LEFT, padx=5)
