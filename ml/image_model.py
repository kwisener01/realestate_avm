"""
Image Model for Property Valuation
Uses CNN (ResNet) to extract features from property images and predict value impact
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import List, Dict, Optional, Union


class PropertyImageDataset(Dataset):
    """Dataset for property images"""

    def __init__(self, image_paths: List[str], labels: Optional[np.ndarray] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image


class ImageModel(nn.Module):
    """
    CNN-based model for property image valuation
    Uses pre-trained ResNet50 with custom regression head
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(ImageModel, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final layer with regression head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.backbone(x).squeeze()

    def extract_features(self, x):
        """Extract features before final regression layer"""
        # Get features from second-to-last layer
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(self.backbone.layer2(
                self.backbone.layer1(self.backbone.maxpool(
                    self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
                ))
            ))
        ))
        return features.view(features.size(0), -1)


class ImageModelWrapper:
    """Wrapper class for training and inference"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = ImageModel()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, image_paths: List[str], labels: np.ndarray,
              epochs: int = 10, batch_size: int = 32,
              learning_rate: float = 0.001, validation_split: float = 0.2) -> Dict:
        """
        Train the image model

        Args:
            image_paths: List of paths to property images
            labels: Target values (property prices)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation

        Returns:
            Dictionary with training metrics
        """
        # Split data
        n = len(image_paths)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - validation_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Normalize labels (important for training stability)
        self.label_mean = train_labels.mean()
        self.label_std = train_labels.std()
        train_labels_norm = (train_labels - self.label_mean) / self.label_std
        val_labels_norm = (val_labels - self.label_mean) / self.label_std

        # Create datasets
        train_dataset = PropertyImageDataset(train_paths, train_labels_norm, self.transform)
        val_dataset = PropertyImageDataset(val_paths, val_labels_norm, self.transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images = images.to(self.model.device)
                targets = targets.to(self.model.device).float()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.model.device)
                    targets = targets.to(self.model.device).float()

                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_dataset)

            # Update learning rate
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Calculate final metrics (denormalized)
        self.model.eval()
        with torch.no_grad():
            val_preds = []
            for images, _ in val_loader:
                images = images.to(self.model.device)
                outputs = self.model(images)
                val_preds.extend(outputs.cpu().numpy())

            val_preds = np.array(val_preds) * self.label_std + self.label_mean
            val_mae = np.mean(np.abs(val_labels - val_preds))
            val_mape = np.mean(np.abs((val_labels - val_preds) / val_labels)) * 100

        return {
            'history': history,
            'val_mae': val_mae,
            'val_mape': val_mape,
            'best_val_loss': best_val_loss
        }

    def predict(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        """
        Make predictions on images

        Args:
            image_paths: Single path or list of paths to images

        Returns:
            Array of predicted price impacts
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for img_path in image_paths:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.model.device)
                output = self.model(image_tensor)
                predictions.append(output.cpu().item())

        predictions = np.array(predictions) * self.label_std + self.label_mean
        return predictions

    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_mean': self.label_mean,
            'label_std': self.label_std
        }, path)

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_mean = checkpoint['label_mean']
        self.label_std = checkpoint['label_std']
        self.model.eval()
