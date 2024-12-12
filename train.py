import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# import wandb
from models import get_model
from scheduler import CosineAnnealingWithWarmRestartsLR
from dataset import MLDataste

seed = 529
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        training_dataloader,
        validation_dataloader,
        classes,
        output_dir,
        max_epochs: int = 10000,
        early_stopping_patience: int = 10,
        execution_name=None,
        lr: float = 1e-4,
        amp: bool = False,
        ema_decay: float = 0.99,
        ema_update_every: int = 16,
        gradient_accumulation_steps: int = 1,
        checkpoint_path: str = None,
        device = 0,
    ):
        self.epochs = max_epochs

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        print("Device used: ", self.device)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        print(f"lr: {lr}")
        for param_group in self.optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(
            self.optimizer, warmup_steps=128, cycle_steps=1024
        )
        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        self.early_stopping_patience = early_stopping_patience

        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)

        self.best_val_accuracy = 0

        self.execution_name = "model" if execution_name is None else execution_name

        if checkpoint_path:
            self.load(checkpoint_path)

        # wandb.watch(model, log="all")

    def run(self):
        counter = 0  # Counter for epochs with no validation loss improvement

        images, _ = next(iter(self.training_dataloader))
        images = [transforms.ToPILImage()(image) for image in images]

        all_train_loss = []
        all_val_loss = []
        for epoch in range(self.epochs):
            for param_group in self.optimizer.param_groups:
                print(f"Current LR: {param_group['lr']}")
            print("[Epoch: %d/%d]" % (epoch + 1, self.epochs))
            self.visualize_stn()
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()
            print(f'Training Loss: {train_loss}, Training Acc: {train_accuracy} %')
            print(f'Validation Loss: {val_loss}, Validation Acc: {val_accuracy} %')
            if(np.isnan(train_loss)):
                train_loss = 300
            if(np.isnan(val_loss)):
                val_loss = 300
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            # plot_learning_curve(all_train_loss, all_val_loss)
            # Early stopping
            if val_accuracy > self.best_val_accuracy:
                self.save(epoch=epoch+1)
                counter = 0
                self.best_val_accuracy = val_accuracy
            else:
                counter += 1
                print(f"Not improvement for {counter} epoch")
                if counter >= self.early_stopping_patience:
                    print(
                        "Validation loss did not improve for %d epochs. Stopping training."
                        % self.early_stopping_patience
                    )
                    break

    def train_epoch(self):
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.training_dataloader))
        for batch_idx, data in enumerate(self.training_dataloader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step()

            batch_accuracy = (predictions == labels).sum().item() / labels.size(0)

            avg_loss.append(loss.item())
            avg_accuracy.append(batch_accuracy)

            # Update progress bar
            pbar.set_postfix(
                {"loss": np.mean(avg_loss), "acc": np.mean(avg_accuracy) * 100.0}
            )
            pbar.update(1)

        pbar.close()

        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def val_epoch(self):
        self.model.eval()

        avg_loss = []
        predicted_labels = []
        true_labels = []

        pbar = tqdm(
            unit="batch", file=sys.stdout, total=len(self.validation_dataloader)
        )
        for batch_idx, (inputs, labels) in enumerate(self.validation_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            avg_loss.append(loss.item())
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        return np.mean(avg_loss), accuracy * 100.0

    def test_model(self):
        self.ema.eval()

        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.testing_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.testing_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                _, logits = self.ema(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        print("Test Accuracy: %.4f %%" % (accuracy * 100.0))

    def visualize_stn(self):
        self.model.eval()

        batch = torch.utils.data.Subset(val_dataset, range(32))

        # Access the batch data
        batch = torch.stack([batch[i][0] for i in range(len(batch))]).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model.stn(batch)

        to_pil = transforms.ToPILImage()

        grid = to_pil(torchvision.utils.make_grid(batch, nrow=16, padding=4))
        stn_batch = to_pil(torchvision.utils.make_grid(stn_batch, nrow=16, padding=4))

    def save(self, epoch=-1):
        data = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_val_accuracy,
        }

        torch.save(data, str(self.output_directory / f"{epoch}.pt"))
        print(f'save model weight: {str(self.output_directory / f"{epoch}.pt")}')

    def load(self, path):
        data = torch.load(path, map_location=self.device)

        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.scaler.load_state_dict(data["scaler"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.best_val_accuracy = data["best_acc"]


def plot_images():
    # Create a grid of images for visualization
    num_rows = 4
    num_cols = 8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    # Plot the images
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j  # Calculate the corresponding index in the dataset
            image, _ = train_dataset[index]  # Get the image
            axes[i, j].imshow(
                image.permute(1, 2, 0)
            )  # Convert tensor to PIL image format and plot
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("images.png")
    plt.show()

def plot_learning_curve(training_losses, validation_losses=None, save_path="learning_curve.png"):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, training_losses, label="Training Loss", color="blue", marker="o")

    if validation_losses:
        plt.plot(epochs, validation_losses, label="Validation Loss", color="orange", marker="o")

    plt.title("Learning Curve", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmoNeXt on Fer2013")

    parser.add_argument("--dataset-path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Path where the best model will be saved",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable mixed precision training",
    )
    parser.add_argument("--in_22k", action="store_true", default=False)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating the model weights",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading."
        "0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file for resuming training or performing inference",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base", "large", "xlarge"],
        default="tiny",
        help="Choose the size of the model: tiny, small, base, large, or xlarge",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Choose device number",
    )

    opt = parser.parse_args()
    print(opt)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exec_name = f"EmoNeXt_{opt.model_size}_{current_time}"

    # wandb.init(project="EmoNeXt", name=exec_name, anonymous="must")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ]
    )

    aug_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    aug2_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.05, 0.3)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    # train_dataset = datasets.ImageFolder(opt.dataset_path + "/train", train_transform)
    train_dataset = MLDataste(opt.dataset_path + "/train", normal_transform=val_transform, aug_transform=aug_transform, aug2_transform=aug2_transform, mode='train')
    classes = ["Angry", "Disgust", "Fear", "Happy","Neutral", "Sad", "Surprise"]
    train_dataset, val_dataset = \
    random_split(train_dataset, [int(0.96 * len(train_dataset)), len(train_dataset) - int(0.96 * len(train_dataset))])

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(val_dataset))
    # print("Using %d images for testing." % len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    net = get_model(len(classes), opt.model_size, in_22k=opt.in_22k)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")

    Trainer(
        model=net,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        classes=classes,
        execution_name=exec_name,
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp,
        device=opt.device,
    ).run()
