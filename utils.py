
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from collections import Counter
from datasets import list_metrics, load_metric
import numpy as np
import torch
import random, os, copy
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm.auto import tqdm

def set_seed(seed=1):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

def f1(y_true, y_pred, weighted=False, print_clfn_rpt=True):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    f1_macro    = f1_score(y_true, y_pred, average='macro')
    f1_micro    = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    if print_clfn_rpt:
        print(classification_report(y_true, y_pred))
    print("accuracy score : ", accuracy_score(y_true, y_pred))
    if weighted:
        return f1_macro, f1_micro, f1_weighted
    return f1_macro, f1_micro

def evaluate_fn(model, eval_dataloader, name, device, return_lbls=False, return_probs=False):
    metric_name="f1"
    metric=load_metric(metric_name)
    model.eval()
    tot_loss = []
    y_tr, y_pr, y_probs = [], [], []
    sftmx = torch.nn.Softmax(dim=1)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch['labels'].type(torch.LongTensor).to(model.device)
        with torch.no_grad():
            outputs = model(**batch)
        loss, logits = outputs[0], outputs[1]
        predictions = torch.argmax(logits, dim=-1)
        y_probs.extend(sftmx(logits).tolist())
        y_pr.extend(predictions.tolist()); y_tr.extend(batch["labels"].tolist())
        metric.add_batch(predictions=predictions, references=batch["labels"])
        tot_loss.append(loss.mean().item())
    score = metric.compute()
    print("Obtained a loss / f1 score of {:.3f} / {:.3f} on {} dataset.".format(
        np.mean(tot_loss), score['f1'], name))
    f1_macro, f1_micro, f1_weighted = f1(np.array(y_tr), np.array(y_pr), True)
    print("Obtained a f1 score macro / micro / weighted : {} / {} / {} \n\n".format(
        f1_macro, f1_micro, f1_weighted))
    model.train()
    print("********************************")
    
    if return_lbls:
        return y_pr
    
    if return_probs:
        return y_probs
    
    return f1_weighted

def softXEnt (input, target, wts=None):
    # print(target)
    target = torch.cat([target[:, None], (1.0-target)[:, None]], axis=1)
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    if wts is not None:
        wtsn = torch.cat([wts[:, None], wts[:, None]], axis=1)
        return  -(wtsn * target * logprobs).sum() / input.shape[0]
    return  -(target * logprobs).sum() / input.shape[0]

def train_fn(model, train_dataloader, eval_dataloader, device, n_epochs=5, lr=5e-5, 
             eval_per_epoch=4, load_best=True, weights=None, use_probab=False, eval_train=True):
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
    model.train(); eval_at = 1/eval_per_epoch;
    best_f1 = 0.0
    pth = "./results/sail"
    if weights is not None:
        print("Training the model with weighted loss, weights : {}".format(weights))
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    elif use_probab:
        loss_fn = softXEnt
    
    loss_fn_b = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch in range(n_epochs):
        print("Training for epoch {} .....".format(epoch))
        next_eval_at = eval_at
        for i, batch in enumerate(train_dataloader):
            if epoch == 0 and i == 0:
                print(batch.get('wts', None))

            if i / len(train_dataloader) > next_eval_at or i == len(train_dataloader)-1 and eval_train:
                # Do the evaluation
                eval_f1 = evaluate_fn(model, eval_dataloader, "eval", device);
                if eval_f1 > best_f1:
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(pth)
                    best_f1 = eval_f1

                evaluate_fn(model, train_dataloader, "train", device); next_eval_at += eval_at
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_old = copy.deepcopy(batch['labels'])
            batch['labels'] = batch['labels'].type(torch.LongTensor).to(model.device)
            wts = batch.get('wts', None)
            if 'wts' in batch:
                batch.pop('wts')
            outputs = model(**batch)

            # outputs[1] are logits of shape (batch_size, n_classes)
            if weights is not None:
                loss = loss_fn(outputs[1].view(-1, 2), batch["labels"].view(-1))
            elif use_probab:
                loss = loss_fn(outputs[1].view(-1, 2), labels_old.view(-1), wts)
            elif wts is not None:
                if epoch == 0 and i == 0:
                    print("using these weights for hard label based training : {}".format(wts))
                loss = loss_fn_b(outputs[1].view(-1, 2), labels_old.view(-1))
                loss = loss * wts
                loss = loss.mean()
            else:
                loss = outputs[0]
            
            loss.backward(); 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            progress_bar.update(1)

        if eval_train:
            evaluate_fn(model, train_dataloader, "train", device)
        evaluate_fn(model, eval_dataloader, "eval", device)

    if load_best:
        print("Loading best model with f1 score of .... {} .".format(best_f1))
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.from_pretrained(pth)
        model.to(device)
        return model_to_load

class textDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert(len(self.encodings[list(self.encodings.keys())[0]]) == len(self.labels))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx][:128]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


import re
def hin_eng_count(data):
    feh = [[0, 0] for _ in data]
    for i, sent in enumerate(data):
        txt = " ".join(sent)
        eng_only_cnt = len(re.sub(r'[^A-Za-z]+', ' ', txt).split())
        hin_only_cnt = len(re.sub(r'[^\u0900-\u097F]+', ' ', txt).split())
        if eng_only_cnt + hin_only_cnt == 0:
            feh[i] = [0.5, 0.5]
        else:
            fe = eng_only_cnt * 1.0 / (eng_only_cnt + hin_only_cnt)
            feh[i] = [fe, 1-fe]

    return np.array(feh)

def es_eng_count(data):
    # use a list of english words from the brown corpus.
    from nltk.corpus import brown
    wrds_set = set(brown.words())
    
    feh = [[0, 0] for _ in data]
    for i, sent in enumerate(data):
        # print(sent)
        txt = " ".join(sent)
        eng_wrds     = [aa for aa in sent.split() if aa in wrds_set]
        non_eng_wrds = [aa for aa in sent.split() if aa not in wrds_set]
        eng_only_cnt = len(eng_wrds)
        hin_only_cnt = len(non_eng_wrds)
        # print(eng_wrds)
        # print(non_eng_wrds)
        # print(eng_only_cnt, hin_only_cnt)
        if eng_only_cnt + hin_only_cnt == 0:
            feh[i] = [0.5, 0.5]
        else:
            fe = eng_only_cnt * 1.0 / (eng_only_cnt + hin_only_cnt)
            feh[i] = [fe, 1-fe]

    return np.array(feh)

def get_ids(eval_dl, y, model, device, return_prbab=False):
    print("\nGetting top 50 percent ids ..... ")
    y_pseudo_pr = evaluate_fn(model, eval_dl, "us eval", device, return_probs=True)
    y_prob      = (np.array(y_pseudo_pr))[:, 0]
    y_pred      = np.argmax(np.array(y_pseudo_pr), axis=-1)

    aa = zip(y_prob, y_pred, y, [i for i in range(len(y))])
    aa_sorted_e2h = sorted(aa, reverse=True)
    y_n_pseudo, y_n, ids = ([a[1] for a in aa_sorted_e2h], 
                            [a[2] for a in aa_sorted_e2h], 
                            [a[3] for a in aa_sorted_e2h])

    newlen = len(y_n) // (4)
    print("Predicted y counter : {} ".format(Counter(y_n_pseudo)))
    print("Actual    y counter : {} ".format(Counter(y_n)))
    y_n_pseudo, y_n, ids = y_n_pseudo[:newlen]+y_n_pseudo[-newlen:], y_n[:newlen] + y_n[-newlen:], ids[:newlen] + ids[-newlen:]
    print("selected - predicted y counter : {} ".format(Counter(y_n_pseudo)))
    print("Selcted  - actual    y counter : {} ".format(Counter(y_n)))
    # print(Counter(y_n))
    # print(Counter(y_n_pseudo))

    y_n_probab = [a[0] for a in aa_sorted_e2h]
    y_n_probab = y_n_probab[:newlen] + y_n_probab[-newlen:]
    if return_prbab:
        return ids, y_n_probab

    return ids

def get_ids_ratio(eval_dl, y, model, device, return_prbab=False, ratio=None):
    print("\nGetting top 50 percent ids ..... ")
    y_pseudo_pr = evaluate_fn(model, eval_dl, "us eval", device, return_probs=True)
    y_prob      = (np.array(y_pseudo_pr))[:, 0]
    y_pred      = np.argmax(np.array(y_pseudo_pr), axis=-1)

    aa = zip(y_prob, y_pred, y, [i for i in range(len(y))])
    aa_sorted_e2h = sorted(aa, reverse=True)
    y_n_pseudo, y_n, ids = ([a[1] for a in aa_sorted_e2h], 
                            [a[2] for a in aa_sorted_e2h], 
                            [a[3] for a in aa_sorted_e2h])

    newlen = len(y_n) // (2)
    freq_ctr = Counter(y_n_pseudo) 
    if ratio is None:
        frac0 = freq_ctr[0] / (freq_ctr[0] + freq_ctr[1])
    else:
        frac0 = ratio

    clss0_len, clss1_len = int(newlen * frac0), int(newlen * (1 - frac0))
    print("Counter {} ".format(frac0))
    print("Predicted y counter : {}".format(Counter(y_n_pseudo)))
    print("Actual    y counter : {}".format(Counter(y_n)))
    y_n_pseudo, y_n, ids = \
        y_n_pseudo[:clss0_len]+y_n_pseudo[-clss1_len:], \
        y_n[:clss0_len] + y_n[-clss1_len:], \
        ids[:clss0_len] + ids[-clss1_len:]
    
    print("selected - predicted y counter : {}".format(Counter(y_n_pseudo)))
    print("Selcted  - actual    y counter : {}".format(Counter(y_n)))
    # print(Counter(y_n))
    # print(Counter(y_n_pseudo))

    y_n_probab = [a[0] for a in aa_sorted_e2h]
    # y_n_probab = y_n_probab[:newlen] + y_n_probab[-newlen:]
    y_n_probab = y_n_probab[:clss0_len] + y_n_probab[-clss1_len:]
    if return_prbab:
        return ids, y_n_probab, frac0

    return ids, y_n_pseudo, frac0
