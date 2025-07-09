import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder

# ======= LOAD AND PREPROCESS DATA =======

df = pd.read_csv("Heuristically_Tagged_Brazil_Paragraphs.csv")
df = df.dropna(subset=["Speech_Paragraph", "Issue", "Framing", "Blame Target", "Tone"])

# Encode categorical variables
issue_encoder = LabelEncoder()
framing_encoder = LabelEncoder()
blame_encoder = LabelEncoder()

df["Issue_Label"] = issue_encoder.fit_transform(df["Issue"])
df["Framing_Label"] = framing_encoder.fit_transform(df["Framing"])
df["Blame_Label"] = blame_encoder.fit_transform(df["Blame Target"])
df["Tone_Label"] = df["Tone"].astype(float) / 5.0  # Normalize tone (0 to 1)

# Tokenize speech paragraphs
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
encodings = tokenizer(df["Speech_Paragraph"].tolist(), truncation=True, padding=True, max_length=256)

# ======= DEFINE DATASET =======

class ParagraphDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        for label_name in self.labels:
            item[label_name] = torch.tensor(self.labels[label_name][idx])
        return item

    def __len__(self):
        return len(self.labels["Issue_Label"])

labels = {
    "Issue_Label": df["Issue_Label"].tolist(),
    "Framing_Label": df["Framing_Label"].tolist(),
    "Blame_Label": df["Blame_Label"].tolist(),
    "Tone_Label": df["Tone_Label"].tolist()
}

dataset = ParagraphDataset(encodings, labels)

# ======= DEFINE MULTITASK BERT MODEL =======

class MultiTaskBERT(nn.Module):
    def __init__(self, num_issues, num_framings, num_blames):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.issue_head = nn.Linear(hidden_size, num_issues)
        self.framing_head = nn.Linear(hidden_size, num_framings)
        self.blame_head = nn.Linear(hidden_size, num_blames)
        self.tone_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return {
            "issue": self.issue_head(cls_output),
            "framing": self.framing_head(cls_output),
            "blame": self.blame_head(cls_output),
            "tone": self.tone_head(cls_output).squeeze(-1)
        }

# ======= DEFINE LOSS FUNCTION =======

def compute_loss(preds, labels):
    issue_loss = F.cross_entropy(preds["issue"], labels["Issue_Label"])
    framing_loss = F.cross_entropy(preds["framing"], labels["Framing_Label"])
    blame_loss = F.cross_entropy(preds["blame"], labels["Blame_Label"])
    tone_loss = F.mse_loss(preds["tone"], labels["Tone_Label"])
    return issue_loss + framing_loss + blame_loss + tone_loss

# ======= TRAINING =======

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskBERT(num_issues=len(issue_encoder.classes_),
                      num_framings=len(framing_encoder.classes_),
                      num_blames=len(blame_encoder.classes_)).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Labels
        labels_batch = {
            "Issue_Label": batch["Issue_Label"].to(device),
            "Framing_Label": batch["Framing_Label"].to(device),
            "Blame_Label": batch["Blame_Label"].to(device),
            "Tone_Label": batch["Tone_Label"].float().to(device)
        }

        optimizer.zero_grad()
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(preds, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} — Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "multitask_bert_brazil.pt")
