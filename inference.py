import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# 1. Import settings and model from your advanced training script
from train_advanced import (
    BATCH_SIZE,
    DATA_DIR,
    MonsterResNet,
    val_test_transform,
)


# 2. Define TestDataset directly here to make the script self-contained
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(Path(root_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path.stem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Retrieve Label Mapping from the training directory
    dummy_train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
    idx_to_class = dummy_train_set.classes

    # Prepare Test DataLoader
    test_dataset = TestDataset(
        os.path.join(DATA_DIR, "test"), val_test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Load the latest MonsterResNet architecture and weights
    model = MonsterResNet(num_classes=len(idx_to_class))
    
    # Ensure the path matches the saved model name from train_advanced.py
    state_dict = torch.load(
        "hw1_advanced.pth", map_location=device
    )

    # Create a new dictionary to remove the 'module.' prefix 
    # (handles models saved via DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load weights into the model
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # Begin inference with TTA (Test-Time Augmentation)
    predictions = []

    with torch.no_grad():
        for inputs, filenames in test_loader:
            inputs = inputs.to(device)

            # --- TTA Implementation ---
            # Generate original, horizontally flipped, and vertically flipped views
            inputs_flipped_h = torch.flip(inputs, dims=[3])
            inputs_flipped_v = torch.flip(inputs, dims=[2])

            with torch.amp.autocast("cuda"):
                # 1. Get model predictions (Logits) for each view
                out_normal = model(inputs)
                out_flipped_h = model(inputs_flipped_h)
                out_flipped_v = model(inputs_flipped_v)

                # 2. Convert Logits to Probabilities using Softmax
                prob_normal = F.softmax(out_normal, dim=1)
                prob_flipped_h = F.softmax(out_flipped_h, dim=1)
                prob_flipped_v = F.softmax(out_flipped_v, dim=1)

                # 3. Average the probabilities for a more stable final prediction
                prob_ensemble = (
                    prob_normal + prob_flipped_h + prob_flipped_v
                ) / 3.0

            # Extract the index of the highest probability class
            _, preds = prob_ensemble.max(1)
            # ---------------------------------------------

            preds = preds.cpu().numpy()

            for filename, pred_idx in zip(filenames, preds):
                real_class_id = idx_to_class[pred_idx]
                predictions.append([filename, real_class_id])

    # Export to the prediction.csv required by CodaBench
    csv_path = "prediction.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(predictions)

    print(f"Inference done! Results saved to {csv_path}")


if __name__ == "__main__":
    main()