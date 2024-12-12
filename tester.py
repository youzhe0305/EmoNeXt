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
import csv
from dataset import MLDataste

# import wandb
from models import get_model
from scheduler import CosineAnnealingWithWarmRestartsLR

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Tester:
    def __init__(
        self,
        model,
        testing_dataloader,
        classes,
        output_dir,
        max_epochs: int = 10000,
        early_stopping_patience: int = 12,
        execution_name=None,
        lr: float = 1e-4,
        amp: bool = False,
        ema_decay: float = 0.99,
        ema_update_every: int = 16,
        gradient_accumulation_steps: int = 1,
        checkpoint_path: str = None,
    ):
        self.epochs = max_epochs

        self.testing_dataloader = testing_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: ", self.device)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr)
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
        self.load('/home/youzhe0305/NYCU-Intro-ML/hw4/EmoNeXt/model_weight/30.pt')
        self.test_model()

    def test_model(self):
        self.ema.eval()

        # predicted_labels = []
        prediction_all = [['filename', 'label']]
        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.testing_dataloader))
        for batch_idx, (inputs, img_path) in enumerate(self.testing_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.to(self.device)
            inputs = inputs.view(-1, c, h, w)

            with torch.autocast(self.device.type, enabled=self.amp):
                _, logits = self.ema(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)
            prediction_all.append([img_path[0].split('/')[-1].split('.')[0], predictions.item()])

            # predicted_labels.extend(predictions.tolist())

            pbar.update(1)

        pbar.close()

        with open('output12.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(prediction_all)

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

    opt = parser.parse_args()
    print(opt)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exec_name = f"EmoNeXt_{opt.model_size}_{current_time}"

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

    test_dataset = MLDataste(opt.dataset_path + "/test", normal_transform=test_transform)
    classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = get_model(len(classes), opt.model_size, in_22k=opt.in_22k)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")

    Tester(
        model=net,
        testing_dataloader=test_loader,
        classes=classes,
        execution_name=exec_name,
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp,
    ).run()
