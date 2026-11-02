import argparse

import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_loaders(data_dir, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.ImageFolder(data_dir, train_tf)
    val_set = datasets.ImageFolder(data_dir, val_tf)

    idx = list(range(len(train_set)))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=train_set.targets, random_state=42
    )

    train_loader = DataLoader(Subset(train_set, train_idx), batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_set, val_idx), batch_size, num_workers=2)
    train_targets = [train_set.targets[i] for i in train_idx]
    return train_loader, val_loader, train_set.classes, train_targets


def class_weights(targets, n_classes):
    counts = torch.bincount(torch.tensor(targets), minlength=n_classes).float()
    return counts.sum() / (n_classes * counts)


def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()
    return preds, labels


def main(data_dir, epochs, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, classes, train_targets = make_loaders(data_dir, 64)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    weights = class_weights(train_targets, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        preds, labels = predict(model, val_loader, device)
        f1 = f1_score(labels, preds, average="macro")
        print(f"epoch {epoch + 1}: loss {total_loss / len(train_loader):.4f}, val macro F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "classes": classes}, out_path)

    model.load_state_dict(torch.load(out_path)["model"])
    preds, labels = predict(model, val_loader, device)
    print(classification_report(labels, preds, target_names=classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="model.pth")
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.out)
