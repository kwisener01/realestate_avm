"""
Text Model for Property Valuation
Uses BERT/transformers to analyze property descriptions and extract value signals
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Dict, Optional


class PropertyTextDataset(Dataset):
    """Dataset for property descriptions"""

    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None,
                 tokenizer=None, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class TextModel(nn.Module):
    """
    BERT-based model for property description analysis
    Uses pre-trained BERT with regression head
    """

    def __init__(self, freeze_bert: bool = True):
        super(TextModel, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Regression
        return self.regressor(pooled_output).squeeze()

    def extract_features(self, input_ids, attention_mask):
        """Extract BERT features"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.pooler_output


class TextModelWrapper:
    """Wrapper class for training and inference"""

    def __init__(self, model_path: Optional[str] = None, max_length: int = 256):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TextModel()
        self.max_length = max_length
        self.label_mean = 0
        self.label_std = 1

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, texts: List[str], labels: np.ndarray,
              epochs: int = 5, batch_size: int = 16,
              learning_rate: float = 2e-5, validation_split: float = 0.2) -> Dict:
        """
        Train the text model

        Args:
            texts: List of property descriptions
            labels: Target values (property prices)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation

        Returns:
            Dictionary with training metrics
        """
        # Split data
        n = len(texts)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - validation_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Normalize labels
        self.label_mean = train_labels.mean()
        self.label_std = train_labels.std()
        train_labels_norm = (train_labels - self.label_mean) / self.label_std
        val_labels_norm = (val_labels - self.label_mean) / self.label_std

        # Create datasets
        train_dataset = PropertyTextDataset(
            train_texts, train_labels_norm, self.tokenizer, self.max_length
        )
        val_dataset = PropertyTextDataset(
            val_texts, val_labels_norm, self.tokenizer, self.max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['label'].to(self.model.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * input_ids.size(0)

            train_loss /= len(train_dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    targets = batch['label'].to(self.model.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * input_ids.size(0)

            val_loss /= len(val_dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            val_preds = []
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                outputs = self.model(input_ids, attention_mask)
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

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on property descriptions

        Args:
            texts: List of property descriptions

        Returns:
            Array of predicted price impacts
        """
        self.model.eval()
        dataset = PropertyTextDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                outputs = self.model(input_ids, attention_mask)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions) * self.label_std + self.label_mean
        return predictions

    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_mean': self.label_mean,
            'label_std': self.label_std,
            'max_length': self.max_length
        }, path)

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_mean = checkpoint['label_mean']
        self.label_std = checkpoint['label_std']
        self.max_length = checkpoint.get('max_length', 256)
        self.model.eval()
