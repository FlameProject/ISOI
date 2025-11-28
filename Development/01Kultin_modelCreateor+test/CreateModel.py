# simple_char_recognizer_24x24_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import torchvision.transforms as transforms

# Простая конфигурация
class Config:
    img_size = 24
    chars = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' + \
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' + \
            '0123456789' + '.,:()'
    num_classes = len(chars)
    batch_size = 128
    epochs = 30
    learning_rate = 0.001

# Простой датасет
class SimpleCharsDataset(Dataset):
    def __init__(self, num_samples=10000, is_train=True):
        self.num_samples = num_samples
        self.is_train = is_train
        self.config = Config()
        
        # Создаем простой шрифт
        self.font_sizes = [16, 18, 20]
        self.fonts = {}
        for size in self.font_sizes:
            try:
                self.fonts[size] = ImageFont.truetype("arial.ttf", size)
            except:
                self.fonts[size] = ImageFont.load_default()
                print(f"⚠️  Используется системный шрифт для размера {size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Случайный символ
        char = random.choice(self.config.chars)
        font_size = random.choice(self.font_sizes)
        font = self.fonts[font_size]
        
        # Создаем изображение
        img = Image.new('L', (self.config.img_size, self.config.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Простое центрирование
        text_x = (self.config.img_size - 10) // 2
        text_y = (self.config.img_size - 10) // 2
        
        draw.text((text_x, text_y), char, fill=255, font=font)
        
        # Простые аугментации для тренировки
        if self.is_train and random.random() > 0.5:
            # Случайное смещение
            x_shift = random.randint(-2, 2)
            y_shift = random.randint(-2, 2)
            new_img = Image.new('L', img.size, 0)
            new_img.paste(img, (x_shift, y_shift))
            img = new_img
        
        # Преобразуем в тензор
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor)
        
        label = self.config.chars.index(char)
        return img_tensor, label

# Простая модель
class SimpleCharRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCharRecognizer, self).__init__()
        
        self.features = nn.Sequential(
            # 24x24 -> 12x12
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 12x12 -> 6x6
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 6x6 -> 3x3
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

def train_simple_model():
    print("🚀 ЗАПУСК ПРОСТОГО ОБУЧЕНИЯ 24x24")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Устройство: {device}")
    
    # Создаем датасеты
    print("📁 Создаем данные...")
    train_dataset = SimpleCharsDataset(num_samples=5000, is_train=True)
    test_dataset = SimpleCharsDataset(num_samples=1000, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Модель
    model = SimpleCharRecognizer(config.num_classes).to(device)
    
    # Считаем параметры
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Параметры модели: {total_params:,}")
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    print(f"\n🎯 Начинаем обучение на {config.epochs} эпох...")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"📚 Обучающие примеры: {len(train_dataset)}")
    print(f"🧪 Тестовые примеры: {len(test_dataset)}")
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    
    for epoch in range(config.epochs):
        # Тренировка
        model.train()
        total_loss = 0
        batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 10 == 0:
                print(f'    Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / batches
        train_losses.append(avg_loss)
        
        # Валидация
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'✅ Эпоха {epoch+1}/{config.epochs}:')
        print(f'   📉 Loss: {avg_loss:.4f}')
        print(f'   📈 Accuracy: {accuracy:.2f}%')
        print(f'   ✅ Правильно: {correct}/{total}')

    # Сохраняем финальную модель
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': avg_loss,
        'config': config.__dict__
    }, 'final_char_model_24x24.pth')
    print(f"💾 Финальная модель сохранена как: final_char_model_24x24.pth")
    
    # Показываем список сохраненных моделей
    print(f"\n📁 Сохраненные модели:")
    for file in os.listdir('.'):
        if file.endswith('.pth') and '24x24' in file:
            file_size = os.path.getsize(file) // 1024
            print(f"   - {file} ({file_size} KB)")
    
    # Простой график
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title('Потери при обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'g-')
    plt.title('Точность на тесте')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_training_results.png', dpi=100)
    plt.show()
    
    return best_accuracy

def test_model_quick():
    """Быстрый тест модели"""
    print("\n🧪 БЫСТРЫЙ ТЕСТ МОДЕЛИ")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Простая модель для теста
    model = SimpleCharRecognizer(config.num_classes).to(device)
    
    # Тестовые данные
    test_dataset = SimpleCharsDataset(num_samples=100, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"📊 Точность на 100 примерах: {accuracy:.2f}%")
    print(f"✅ Правильно: {correct}/{total}")
    
    return accuracy

def check_existing_models():
    """Проверяем существующие модели"""
    print("🔍 Поиск существующих моделей...")
    model_files = []
    
    for file in os.listdir('.'):
        if file.endswith('.pth'):
            model_files.append(file)
            file_size = os.path.getsize(file) // 1024
            print(f"   📁 {file} ({file_size} KB)")
    
    return model_files

if __name__ == "__main__":
    # Устанавливаем случайные семена для воспроизводимости
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    try:
        print("=" * 50)
        print("🎯 ПРОСТОЙ РАСПОЗНАТЕЛЬ СИМВОЛОВ 24x24")
        print("=" * 50)
        
        # Проверяем существующие модели
        existing_models = check_existing_models()
        
        if existing_models:
            print(f"\n✅ Найдено {len(existing_models)} моделей")
            response = input("🔄 Хотите обучить новую модель? (y/n): ")
            if response.lower() != 'y':
                print("🚫 Обучение отменено")
                exit()
        
        # Сначала быстрый тест
        test_accuracy = test_model_quick()
        
        print("\n🔧 Начинаем обучение...")
        final_accuracy = train_simple_model()
        
        print(f"\n🎊 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {final_accuracy:.2f}%")
        
        # Показываем инструкцию для тестера
        print(f"\n📋 ИНСТРУКЦИЯ:")
        print(f"1. Запустите тестер: python interactive_symbol_tester_24x24_fixed.py")
        print(f"2. Он автоматически найдет сохраненные модели")
        print(f"3. Используйте слайдеры для тестирования разных символов")
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
