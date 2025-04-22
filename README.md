# goemotions-multilabel-classifier
# Install dependencies
!pip install transformers datasets scikit-learn

# Imports
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("go_emotions")
df_train, df_valid, df_test = map(pd.DataFrame, [dataset["train"], dataset["validation"], dataset["test"]])
label_names = dataset["train"].features["labels"].feature.names

# Binarize labels
mlb = MultiLabelBinarizer(classes=list(range(len(label_names))))
y_train, y_valid, y_test = map(mlb.fit_transform, [df_train["labels"], df_valid["labels"], df_test["labels"]])

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(texts): return tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
train_encodings = tokenize(df_train["text"])
valid_encodings = tokenize(df_valid["text"])
test_encodings = tokenize(df_test["text"])

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = EmotionDataset(train_encodings, y_train)
valid_dataset = EmotionDataset(valid_encodings, y_valid)
test_dataset = EmotionDataset(test_encodings, y_test)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_names), problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Metrics
def compute_metrics(pred):
    preds = (pred.predictions > 0.5).astype(int)
    labels = pred.label_ids
    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "hamming_loss": hamming_loss(labels, preds)
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
print("\nTest set evaluation:")
trainer.evaluate(test_dataset)

# Predict on custom text
def predict_emotions(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).detach().numpy()
    return [{label_names[i]: prob[i] for i in range(len(prob)) if prob[i] > 0.5} for prob in probs]

# Example predictions
sample_texts = ["I love this product!", "I'm really upset with the service."]
predictions = predict_emotions(sample_texts)
for text, pred in zip(sample_texts, predictions):
    print(f"\nText: {text}\nPredicted Emotions: {pred}")

