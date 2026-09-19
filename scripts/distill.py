import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MODELS_DIR, build_model
from student_model import StudentCNN

DATA_DIR = MODELS_DIR / "mnist_data"
STUDENT_PATH = MODELS_DIR / "student_distilled.pt"

EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
TEMPERATURE = 4.0
ALPHA = 0.3  # weight on hard-label loss; (1 - ALPHA) goes to the distillation loss


def get_loaders():
    # torchvision MNIST is already 28x28, white stroke on black background,
    # 0-255 -> ToTensor() scales it to [0, 1] -- the same convention preprocess()
    # in main.py produces, so no extra inversion/normalization is needed here.
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


def distillation_loss(student_logits, teacher_logits, labels):
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_teacher = F.softmax(teacher_logits / TEMPERATURE, dim=1)
    soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (TEMPERATURE ** 2)

    return ALPHA * hard_loss + (1 - ALPHA) * soft_loss


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_student(teacher, student, train_loader, test_loader):
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        student.train()
        start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(student, test_loader)
        print(f"epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  test_acc={test_acc:.4f}  ({time.time() - start:.1f}s)")


def main():
    print("Loading MNIST (downloads on first run)...")
    train_loader, test_loader = get_loaders()

    print("Loading frozen teacher model...")
    teacher = build_model()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = StudentCNN()
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,} ({teacher_params / student_params:.0f}x smaller)")

    print("\nDistilling...")
    train_student(teacher, student, train_loader, test_loader)

    teacher_acc = evaluate(teacher, test_loader)
    student_acc = evaluate(student, test_loader)
    print(f"\nFinal teacher test accuracy: {teacher_acc:.4f}")
    print(f"Final student test accuracy: {student_acc:.4f}")

    torch.save(student.state_dict(), STUDENT_PATH)
    print(f"\nSaved distilled student weights to {STUDENT_PATH} ({STUDENT_PATH.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
