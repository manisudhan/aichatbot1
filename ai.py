import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Custom Dataset
class CaseDataset(Dataset):
    def __init__(self, cases, labels, tokenizer, max_len):
        self.cases = cases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, item):
        case = str(self.cases[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            case,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'case_text': case,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Prepare Dataset
cases = ["Stealing from a rich student", "Stealing from a struggling father of 4 children"] # Example cases
labels = [0.3, 0.7]  # Example weights; you would need a larger dataset for training

dataset = CaseDataset(cases, labels, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Initialize the Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item()}")

# Save the trained model
model.save_pretrained('case_weightage_model')
tokenizer.save_pretrained('case_weightage_model')
print("done")
