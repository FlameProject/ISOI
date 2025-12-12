# model.py
import torch
import torch.nn as nn
import os

class Config:
    img_size = 24
    chars = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' + \
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' + \
            '0123456789' + '.,:()'
    num_classes = len(chars)

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

def load_model(model_path='final_char_model_24x24.pth'):
    """Загрузка обученной модели"""
    print(f"🤖 ЗАГРУЗКА МОДЕЛИ: {model_path}")
    
    if not os.path.exists(model_path):
        # Ищем другие модели
        model_files = [f for f in os.listdir('.') if f.endswith('.pth') and '24x24' in f]
        if not model_files:
            model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not model_files:
            raise FileNotFoundError("Не найдена модель для распознавания!")
        
        model_path = model_files[0]
        print(f"⚠️  Используем модель: {model_path}")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCharRecognizer(config.num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ МОДЕЛЬ ЗАГРУЖЕНА НА: {device}")
    return model, device, config