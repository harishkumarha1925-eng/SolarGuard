# src/train_classification.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from model import get_resnet50
from dataset import make_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet50(num_classes=args.num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader, val_loader, classes = make_dataloaders(args.train_dir, args.val_dir,
                                                         batch_size=args.batch_size, size=args.size, num_workers=args.num_workers)
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0; total=0; correct=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_loss = running_loss / total if total>0 else 0
        train_acc = correct / total if total>0 else 0

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        val_acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0

        print(f"Epoch {epoch}/{args.epochs} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'resnet50_best.pth'))
            print(f"Saved best model (val_acc={best_acc:.4f})")

    print("Training finished. Best val acc:", best_acc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--save_dir', default='models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    train(args)
