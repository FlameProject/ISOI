# reliable_enhanced_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import urllib.request
import gzip
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Убедимся, что данные загружены
def ensure_mnist_data():
    """Проверяет и загружает данные MNIST если нужно"""
    data_path = './data/MNIST/raw'
    os.makedirs(data_path, exist_ok=True)
    
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    
    all_files_exist = all(os.path.exists(os.path.join(data_path, f)) for f in files)
    
    if not all_files_exist:
        print("📥 Загружаем данные MNIST...")
        for url, filename in zip(urls, files):
            filepath = os.path.join(data_path, filename)
            if not os.path.exists(filepath):
                print(f"Скачиваем {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(filepath.replace('.gz', ''), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"✅ {filename} загружен и распакован")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {filename}: {e}")
    else:
        print("✅ Данные MNIST уже загружены")

# Улучшенная архитектура с вниманием к проблемным цифрам
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

# Улучшенная аугментация с фокусом на проблемные цифры
class AdvancedAugmentation:
    def __init__(self):
        self.affine_transform = transforms.RandomAffine(
            degrees=10,  # Уменьшил вращение для сохранения ориентации цифр
            translate=(0.08, 0.08),  # Уменьшил смещение
            scale=(0.9, 1.1),  # Более консервативное масштабирование
            shear=8,  # Уменьшил наклон
            fill=0
        )
    
    def __call__(self, img):
        # Применяем аффинные преобразования
        img = self.affine_transform(img)
        
        # Случайное изменение яркости (30% chance) - реже
        if np.random.random() > 0.7:
            factor = np.random.uniform(0.8, 1.2)  # Более узкий диапазон
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Случайное изменение контраста (30% chance)
        if np.random.random() > 0.7:
            factor = np.random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # Случайное размытие (10% chance) - реже
        if np.random.random() > 0.9:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # Меньше размытие
            
        # Случайная эрозия/дилатация для улучшения различий между 3 и 9
        if np.random.random() > 0.8:
            if np.random.random() > 0.5:
                img = img.filter(ImageFilter.MinFilter(3))  # Эрозия
            else:
                img = img.filter(ImageFilter.MaxFilter(3))  # Дилатация
        
        return img

# Классы для трансформаций (должны быть глобальными для multiprocessing)
class TrainTransform:
    def __init__(self):
        self.augmentation = AdvancedAugmentation()
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def __call__(self, img):
        # Применяем аугментацию с вероятностью 80%
        if np.random.random() > 0.2:
            img = self.augmentation(img)
        img = self.base_transform(img)
        return img

class TestTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def __call__(self, img):
        return self.transform(img)

# Функция для анализа ошибок между 3 и 9
def analyze_3_9_confusion(model, test_loader, device):
    """Анализирует и визуализирует confusion между 3 и 9"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Особый фокус на 3 и 9
    confusion_3_9 = cm[3, 9] + cm[9, 3]
    total_3_9 = cm[3].sum() + cm[9].sum() - cm[3, 3] - cm[9, 9]
    confusion_rate = confusion_3_9 / total_3_9 if total_3_9 > 0 else 0
    
    plt.subplot(1, 2, 2)
    classes = ['3-3', '3-9', '9-3', '9-9']
    values = [cm[3, 3], cm[3, 9], cm[9, 3], cm[9, 9]]
    colors = ['green', 'red', 'red', 'green']
    
    bars = plt.bar(classes, values, color=colors)
    plt.title(f'Confusion 3-9: {confusion_rate:.2%}')
    plt.ylabel('Count')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('3_9_confusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🔍 Анализ confusion 3-9:")
    print(f"   3 ошибочно классифицированы как 9: {cm[3, 9]}")
    print(f"   9 ошибочно классифицированы как 3: {cm[9, 3]}")
    print(f"   Общая ошибка между 3 и 9: {confusion_3_9}")
    print(f"   Процент ошибок: {confusion_rate:.2%}")
    
    return confusion_rate

# Функция для визуализации проблемных примеров
def visualize_problem_cases(model, test_dataset, device, num_examples=10):
    """Визуализирует примеры, где модель ошибается между 3 и 9"""
    model.eval()
    problematic_examples = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if len(problematic_examples) >= num_examples * 2:
                break
                
            img, target = test_dataset[i]
            if target not in [3, 9]:
                continue
                
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            if pred != target and pred in [3, 9]:
                problematic_examples.append((img, target, pred))
    
    # Отображаем проблемные примеры
    if problematic_examples:
        plt.figure(figsize=(15, 6))
        for i, (img, target, pred) in enumerate(problematic_examples[:num_examples]):
            plt.subplot(2, 5, i+1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'True: {target}, Pred: {pred}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('problematic_3_9_examples.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_enhanced_model():
    ensure_mnist_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Улучшенное обучение на {device}")
    
    # Используем глобальные классы для трансформаций
    train_transform = TrainTransform()
    test_transform = TestTransform()
    
    # Загружаем датасеты
    try:
        print("📁 Загружаем тренировочные данные...")
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True,
            transform=train_transform
        )
        
        print("📁 Загружаем тестовые данные...")
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True,
            transform=test_transform
        )
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return 0, 0
    
    # DataLoader - убираем pin_memory и num_workers для Windows
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    # Модель
    model = EnhancedDigitRecognizer().to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Архитектура модели:")
    print(f"   - 4 сверточных блока с BatchNorm")
    print(f"   - Global Average Pooling")
    print(f"   - Улучшенная регуляризация")
    print(f"   - Всего параметров: {total_params:,}")
    
    # Улучшенный оптимизатор и планировщик
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing для лучшей обобщающей способности
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Комбинированный планировщик
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=25,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Обучение
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    
    print("\n🎯 Начинаем улучшенное обучение с фокусом на 3 и 9...")
    print("Улучшения в аугментации:")
    print("   - Более консервативные преобразования")
    print("   - Морфологические операции для улучшения различий")
    print("   - Label smoothing для лучшей регуляризации")
    print("   - OneCycle LR scheduling")
    
    for epoch in range(25):  # Увеличил количество эпох
        # Тренировка
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Эпоха {epoch+1}/25 [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.6f}, LR: {current_lr:.6f}')
        
        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Валидация
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Эпоха {epoch+1} завершена:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Accuracy: {test_accuracy:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Сохраняем лучшую модель
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_accuracy': test_accuracy,
                'epoch': epoch,
            }, 'enhanced_digit_model.pth')
            print(f'  💾 Сохранена лучшая модель с точностью {test_accuracy:.2f}%')
    
    print(f"\n✅ Обучение завершено!")
    print(f"🎯 Лучшая точность на тесте: {best_accuracy:.2f}%")
    
    # Анализ confusion между 3 и 9
    print("\n🔍 Проводим анализ confusion между 3 и 9...")
    confusion_rate = analyze_3_9_confusion(model, test_loader, device)
    
    # Визуализация проблемных случаев
    print("\n📊 Визуализируем проблемные примеры...")
    visualize_problem_cases(model, test_dataset, device)
    
    # Графики обучения
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Потери при обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Точность на тестовых данных')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Финальный анализ
    if confusion_rate < 0.01:
        print("🎉 Отлично! Проблема 3-9 практически решена!")
    elif confusion_rate < 0.03:
        print("👍 Хорошо! Confusion между 3 и 9 значительно уменьшился!")
    elif confusion_rate < 0.05:
        print("💪 Неплохо, но есть куда улучшать.")
    else:
        print("🔧 Требуется дополнительная работа над проблемой 3-9.")
    
    return best_accuracy, confusion_rate

if __name__ == "__main__":
    accuracy, confusion_rate = train_enhanced_model()
