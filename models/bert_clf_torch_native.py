from math import copysign
from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
print("Reading dataset complete, train/val/test is {}/{}/{}".format(
    len(train_texts), len(val_texts), len(test_texts)))

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
# print(len(train_encodings))
# exit()
import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx][:50]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)


from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from datasets import list_metrics, load_metric
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm.auto import tqdm

def evaluate_fn(model, eval_dataloader, name):
    if name == "train":
        return 0
    metric= load_metric("f1")
    model.eval()
    tot_loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss, logits = outputs[0], outputs[1]
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        tot_loss.append(loss.mean().item())
    score = metric.compute()
    print("Obtained a loss/f1 score of {} / {} on {} dataset.".format(
        np.mean(tot_loss), score['f1'], name))
    model.train()
    return score['f1']

def train_fn(model, train_dataloader, eval_dataloader, n_epochs=5, lr=5e-5):

    # Create optimizer and the scheduler.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    num_training_steps = n_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                   num_warmup_steps=num_training_steps//10, 
                                                   num_training_steps=num_training_steps)

    # write the train loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    eval_per_epoch = 5; eval_at = 1/eval_per_epoch
    best_f1 = 0.0
    pth = "./results/imdb"
    for epoch in range(n_epochs):
        print("Training for epoch {} .....".format(epoch))
        next_eval_at = eval_at
        for i, batch in enumerate(train_dataloader):
            if i / len(train_dataloader) > next_eval_at:
                # Do the evaluation
                eval_f1 = evaluate_fn(model, eval_dataloader, "eval");
                if eval_f1 > best_f1:
                    print("Saving model")
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(pth)
                    best_f1 = eval_f1

                evaluate_fn(model, train_dataloader, "train"); next_eval_at += eval_at
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            
            loss.backward(); 
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            progress_bar.update(1)

        evaluate_fn(model, train_dataloader, "train")
        evaluate_fn(model, eval_dataloader, "eval")
    model_to_load = AutoModelForSequenceClassification.from_pretrained(pth)
    model_to_load.to(device)
    return model_to_load

metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",
    num_labels=len(set(train_labels)))

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
eval_dataloader = DataLoader(val_dataset, batch_size=64)

for p in model.bert.parameters():
    p.requires_grad = True


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model = train_fn(model, train_dataloader, eval_dataloader, n_epochs=3)
print("Final evlauation results ..... ")
evaluate_fn(model, eval_dataloader, "eval")


