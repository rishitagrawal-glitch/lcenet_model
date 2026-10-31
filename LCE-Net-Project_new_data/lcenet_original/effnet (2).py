import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ===== 参数设置 =====
data_dir = '../dataset/Potato'
num_classes = 3
batch_size = 32
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 图像增强与预处理 =====
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

aug_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== 加载数据集并划分（固定随机种子） =====
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# 固定随机种子划分
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# 设置不同 transform
train_dataset.dataset.transform = aug_transforms
val_dataset.dataset.transform = data_transforms

# ===== WeightedRandomSampler (过采样) =====
train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
class_sample_count = np.bincount(train_labels)
weights = 1. / class_sample_count
samples_weight = [weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

dataloaders = {
    'Train': DataLoader(train_dataset, batch_size=batch_size, sampler=sampler),
    'Valid': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

# ===== CBAM 注意力模块定义 =====
class Lite_CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = self.sa(torch.cat([max_out, avg_out], dim=1))
        return x * sa

# ===== 定义 EfficientNet-B0 + CBAM 模型 =====
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.cbam = Lite_CBAM(channels=320)
        self.head = self.features[8]  # Conv2d(320 → 1280)
        self.pool = base.avgpool
        self.dropout = base.classifier[0]
        self.classifier = nn.Linear(1280, num_classes)

        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "features.6" in name or "classifier" in name:
                param.requires_grad = True

    def forward(self, x):
        for i in range(8):
            x = self.features[i](x)
        x = self.cbam(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

model = EfficientNet_CBAM(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam([
    {'params': model.features[6].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
])
criterion = nn.CrossEntropyLoss()

def main():
    best_val_acc = 0.0
    best_model_path = 'best_model_cbam.pth'

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in dataloaders['Train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_inputs, val_labels in dataloaders['Valid']:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val_acc = {val_acc:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- Train Loss: {train_loss:.4f} "
              f"- Train Acc: {train_acc:.4f} "
              f"- Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['Valid']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
