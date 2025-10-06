# src/train_classification.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from model import get_resnet50
from dataset import make_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet50(num_classes=args.num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, classes = make_dataloaders(args.train_dir, args.val_dir,
                                                         batch_size=args.batch_size, size=args.size)
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_acc = correct/total

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{args.epochs} TrainAcc:{train_acc:.4f} ValAcc:{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'resnet50_best.pth'))
    print("Best val acc:", best_acc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--save_dir', default='../models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=6)
    args = parser.parse_args()
    train(args)
