import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ================= Configuration =================
DATA_DIR = os.path.expanduser("~/DL_CV_class/HW/HW1/data")
IMG_SIZE = 448
BATCH_SIZE = 8
ACCUM_STEPS = 4
EPOCHS = 200
LR = 1e-4
PATIENCE_LIMIT = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. Data Processing =================
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize
])


# ================= 2. Mixup Function =================
def mixup_data(x, y, alpha=0.2):
    """Return mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class MonsterResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Load ResNet-152 as Backbone (骨幹網路)
        self.backbone = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Custom Classifier Head (自訂分類頭)
        self.custom_head = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.custom_head(x)
        return x


# ================= 3. Core Functions =================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss_a = criterion(outputs, labels_a)
            loss_b = criterion(outputs, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0) * ACCUM_STEPS
        _, pred = outputs.max(1)

        # Accuracy computation under Mixup
        correct_a = pred.eq(labels_a).sum().item()
        correct_b = pred.eq(labels_b).sum().item()
        correct += (lam * correct_a + (1 - lam) * correct_b)
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ================= 4. Main Program =================
if __name__ == "__main__":
    print(f"Using device: {DEVICE} | Resolution: {IMG_SIZE}")

    train_set = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), 
        train_transform
    )
    val_set = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), 
        val_test_transform
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    model = MonsterResNet(num_classes=len(train_set.classes)).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f} M (Limit < 100M)")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.2, 
        patience=3
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()

        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        v_loss, v_acc = validate(
            model, val_loader, criterion
        )

        scheduler.step(v_acc)
        curr_lr = optimizer.param_groups[0]['lr']
        time_elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | LR: {curr_lr:.1e} | "
            f"T-Loss: {t_loss:.4f} | T-Acc: {t_acc:.4f} | "
            f"V-Loss: {v_loss:.4f} | V-Acc: {v_acc:.4f} | "
            f"Time: {time_elapsed:.1f}s"
        )

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(
                model.state_dict(), 
                "hw1_base.pth"
            )
            print(f"New Best: {best_acc:.4f} Saved!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE_LIMIT:
                print("Early stopping triggered!")
                break

    print(f"Done! Best Val Acc: {best_acc:.4f}")