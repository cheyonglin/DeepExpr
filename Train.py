import os
import io
import csv
import time
import random
import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thop import profile

from model import pyramid_trans_expr


# ==================== 参数配置 ====================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default="./datasets")#数据集路径
    parser.add_argument("--save-dir", type=str, default="./Result/4")#结果保存
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)#训练轮数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model-type", type=int, default=4)
    parser.add_argument(
        "--resume",
        choices=["last", "best", "none"],
        default="last",
    )

    return parser.parse_args()


# ==================== 随机种子 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================== 数据集 ====================
class MonkeyFaceDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        img_size=224,
        transform=None,
        num_classes=6,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.samples = []

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"图片目录不存在：{self.images_dir}"
            )

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"标签目录不存在：{self.labels_dir}"
            )

        self.transform = (
            transform
            if transform is not None
            else transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        )

        img_files = sorted([
            p for p in self.images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp"
            }
        ])

        for img_path in img_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            with label_path.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, start=1):
                    parts = line.strip().split()

                    if not parts:
                        continue

                    location = f"{label_path}:{line_no}"

                    if len(parts) != 5:
                        raise ValueError(
                            f"{location}：需要5列 TXT 格式标签。"
                        )

                    values = np.asarray(
                        [float(value) for value in parts],
                        dtype=np.float64,
                    )

                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"{location}：标签包含无效数值。"
                        )

                    cls_value, x, y, bw, bh = values

                    if not float(cls_value).is_integer():
                        raise ValueError(
                            f"{location}：类别ID必须为整数。"
                        )

                    cls = int(cls_value)

                    if not 0 <= cls < num_classes:
                        raise ValueError(
                            f"{location}：类别ID必须在 "
                            f"0～{num_classes - 1} 之间。"
                        )

                    if not (
                        0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 < bw <= 1
                        and 0 < bh <= 1
                    ):
                        raise ValueError(
                            f"{location}：坐标需归一化到 [0, 1]，"
                            "且宽高必须大于0。"
                        )

                    self.samples.append(
                        (img_path, cls, (x, y, bw, bh))
                    )

        if not self.samples:
            raise ValueError(
                f"目录中没有可用样本：{self.images_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls, bbox = self.samples[idx]

        img = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if img is None:
            raise RuntimeError(f"无法读取图片：{img_path}")

        height, width = img.shape[:2]
        x, y, bw, bh = bbox

        cx = x * width
        cy = y * height
        bw_pix = bw * width
        bh_pix = bh * height

        x1 = max(0, min(width, int(round(cx - bw_pix / 2))))
        y1 = max(0, min(height, int(round(cy - bh_pix / 2))))
        x2 = max(0, min(width, int(round(cx + bw_pix / 2))))
        y2 = max(0, min(height, int(round(cy + bh_pix / 2))))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"无效裁剪框：{img_path}, bbox={bbox}"
            )

        face = img[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)

        return self.transform(face), cls


# ==================== 权重初始化 ====================
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(
        module,
        (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm),
    ):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ==================== 获取分类输出 ====================
def get_logits(model, imgs):
    outputs = model(imgs)

    # 对应模型返回值：(分类输出, 辅助输出)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    return outputs


# ==================== 模型统计 ====================
def print_model_summary(model, img_size, device):
    total_layers = len(list(model.modules()))
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    total_gradients = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    profile_model = deepcopy(model).to(device)
    profile_model.eval()
    cuda_devices = (
        [device.index if device.index is not None
         else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            dummy_input = torch.randn(
                1, 3, img_size, img_size,
                device=device,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This API is being deprecated",
                    category=UserWarning,
                    module=r"thop\.vision\.calc_func",
                )

                with torch.no_grad(), redirect_stdout(io.StringIO()):
                    macs, _ = profile(
                        profile_model,
                        inputs=(dummy_input,),
                        verbose=False,
                    )
    finally:
        del profile_model
    print(
        f"model summary: {total_layers} layers, "
        f"{total_params / 1e6:.2f}M parameters, "
        f"{total_gradients / 1e6:.2f}M gradients, "
        f"{macs / 1e9:.2f} GFLOPs, "
        f"{macs / 1e6:.1f} MFLOPs"
    )


# ==================== 训练 ====================
def train_one_epoch(
    model, loader, criterion, optimizer, device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = get_logits(model, imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ==================== 验证 ====================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = get_logits(model, imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ==================== 加载检查点 ====================
def load_checkpoint(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


# ==================== 测试 ====================
@torch.no_grad()
def test_model(model, test_loader, device, save_dir):
    model.eval()

    preds_all = []
    labels_all = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)

        outputs = get_logits(model, imgs)
        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.tolist())

    correct = sum(
        pred == target
        for pred, target in zip(preds_all, labels_all)
    )
    test_acc = correct / len(labels_all)
    test_error = 1.0 - test_acc

    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print(f"Test Error Rate: {test_error:.4f}")

    out_pred_file = save_dir / "small_test_preds.txt"

    with out_pred_file.open("w", encoding="utf-8") as f:
        for pred, target in zip(preds_all, labels_all):
            f.write(f"{pred}\t{target}\n")

    print(f"Test predictions saved to {out_pred_file}")


# ==================== 主程序 ====================
def main():
    args = parse_args()

    if (
        args.img_size <= 0
        or args.num_classes <= 0
        or args.batch_size <= 0
        or args.epochs <= 0
        or args.lr <= 0
        or args.num_workers < 0
    ):
        raise ValueError("训练参数无效，请检查输入。")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)

    train_images = data_root / "images" / "train"
    train_labels = data_root / "labels" / "train"
    val_images = data_root / "images" / "val"
    val_labels = data_root / "labels" / "val"
    test_images = data_root / "images" / "test"
    test_labels = data_root / "labels" / "test"

    csv_dir = save_dir / "Train_csv"
    model_dir = save_dir / "model_save"

    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_file = csv_dir / "train_log.csv"
    best_model_path = model_dir / "best.pth"
    last_model_path = model_dir / "last.pth"

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_ds = MonkeyFaceDataset(
        train_images,
        train_labels,
        img_size=args.img_size,
        transform=train_transform,
        num_classes=args.num_classes,
    )

    val_ds = MonkeyFaceDataset(
        val_images,
        val_labels,
        img_size=args.img_size,
        transform=val_transform,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    with redirect_stdout(io.StringIO()):
        model = pyramid_trans_expr(
            img_size=args.img_size,
            num_classes=args.num_classes,
            type=args.model_type,
        )
    model.apply(init_weights)
    model = model.to(device)
    print_model_summary(model, args.img_size, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    start_epoch = 1
    best_val_acc = -1.0
    resumed = False
    resume_path = None
    if args.resume == "last":
        resume_path = last_model_path
    elif args.resume == "best":
        resume_path = best_model_path
    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1

        best_val_acc = max(
            float(checkpoint.get("best_val_acc", -1.0)),
            float(checkpoint.get("val_acc", -1.0)),
        )

        resumed = True

    # ==================== CSV日志 ====================
    fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]

    previous_rows = []

    if resumed and csv_file.exists():
        with csv_file.open(
            "r", newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != fields:
                raise ValueError(
                    f"{csv_file} 表头不匹配，请更换输出目录。"
                )

            previous_rows = [
                row for row in reader
                if int(row["epoch"]) < start_epoch
            ]

    with csv_file.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous_rows)

    evaluation_path = None

    if resumed:
        evaluation_path = (
            best_model_path
            if best_model_path.exists()
            else resume_path
        )

    # ==================== 训练循环 ====================
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }

        torch.save(checkpoint, last_model_path)

        if is_best:
            torch.save(checkpoint, best_model_path)
            evaluation_path = best_model_path

        with csv_file.open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"time={time.time() - t0:.1f}s | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    # ==================== 最佳模型测试 ====================
    if os.path.exists(test_images) and os.path.exists(test_labels):
        if evaluation_path is None:
            raise RuntimeError("没有可用于测试的训练检查点。")

        checkpoint = load_checkpoint(evaluation_path, device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_ds = MonkeyFaceDataset(
            test_images,
            test_labels,
            img_size=args.img_size,
            transform=val_transform,
            num_classes=args.num_classes,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        test_model(
            model,
            test_loader,
            device,
            save_dir,
        )
    else:
        print("\n未找到测试集目录，跳过测试。")


if __name__ == "__main__":
    main()