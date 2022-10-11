#!/usr/bin/env python
# -*- coding: utf-8 -*

# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
import argparse
import logging
import os
import random
import re

from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from tqdm import tqdm, trange
from transformers import (
    BertForSequenceClassification, BertTokenizer, XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW, BertConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score
from BertSequence import *


logger = logging.getLogger(__name__)

def get_labels(data_dir):
    all_path = os.path.join(data_dir, "train.txt")
    logger.info("all_path {}".format(all_path))
    labels = []
    with open(all_path, "r", encoding="utf-8") as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        splits = line.split('\t')
        label = splits[-1]
        if label not in labels:
            labels.append(label)
    return labels

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def create_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="Dataset on which the model is being run.")
    
    # Optional Parameters
    parser.add_argument("--output_file_prefix", default="", type=str,
                        help="Prefix for the output file to be written.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_type", type=str,
                        default='bert', help='type of model xlm/xlm-roberta/bert')
    parser.add_argument("--model_name", default='bert-base-multilingual-cased',
                        type=str, help='name of pretrained model/path to checkpoint')
    parser.add_argument("--save_steps", type=int, default=1, help='set to -1 to not save model')
    parser.add_argument("--max_seq_length", default=128, type=int, help="max seq length after tokenization")
    parser.add_argument("--lang", default="switched", type=str, help="language model to be used")
    parser.add_argument("--model_path", default="", type=str, help="saved model to be used")
    parser.add_argument("--eval_only", default=0, type=int, help="if set to true, only evaluate the model")
    return parser

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(
        y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    return{
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall,
        "counter preds": Counter(preds),
        "counter labels": Counter(labels)
    }

def get_example_length(data_dir, mode, lang):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
     
    # Assert the mode is validation.
    assert(mode in ["validation", "test"])

    with open(file_path, 'r', encoding="utf-8") as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        x = line.split('\t')
        text = x[0]
        label = x[1]
        eng_only = re.sub(r'[\u0900-\u097F]+', '', text)
        hin_only = re.sub(r'[A-Za-z]+', '', text)
        examples.append((len(eng_only.split()), len(hin_only.split())))
    return examples

def read_examples_from_file(data_dir, mode, lang):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []

    # Assert the lang provided is as specified.
    assert(lang in ["eng", "hin", "switched", "es", "twoBerts"])
    with open(file_path, 'r', encoding="utf-8") as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        text = x[0]
        label = x[1]
        eng_only = re.sub(r'[\u0900-\u097F]+', '', text)
        hin_only = re.sub(r'[A-Za-z]+', '', text)
        if len(eng_only.split()) <= 3:
            eng_only += " shortTextHere"
        if len(hin_only.split()) <= 3:
            hin_only += " shortTextHere"
        examples.append({
                         #'text': text,
                         'langA': eng_only, 
                         'langB': hin_only, 
                         'label': label})

    if mode == 'test':
        for i in range(len(examples)):
            if examples[i]['langA'] == 'not found':
                examples[i]['present'] = False
            else:
                examples[i]['present'] = True

    if mode == "train":
        logger.info("\nTraining examples for language {} count : {}, "
                    "actual count : {} \n".format(lang, len(examples), len(lines)))

    return examples

def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 max_seq_length=128):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):

        label = example['label']
        label = label_map[label]
        ex_dict = {}
        for lang in ['langA', 'langB']:
            sentence = example[lang]

            sentence_tokens = tokenizer.tokenize(sentence)[:max_seq_length - 2]
            sentence_tokens = [tokenizer.cls_token] + \
                sentence_tokens + [tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
            ex_dict["input_ids_" + lang] = input_ids

        ex_dict["label"] = label
        features.append(ex_dict)
        if 'present' in example:
            features[-1]['present'] = example['present']

    return features

def load_and_cache_examples(args, tokenizer, labels, mode, lang):

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode, lang)
    features = convert_examples_to_features(examples, labels, tokenizer, args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids_a = [f['input_ids_langA'] for f in features]
    all_input_ids_b = [f['input_ids_langB'] for f in features]
    all_labels = [f['label'] for f in features]
    args = [all_input_ids_a, all_input_ids_b, all_labels]
    if 'present' in features[0]:
        present = [1 if f['present'] else 0 for f in features]
        args.append(present)

    dataset = CustomDataset(*args)
    return dataset

class CustomDataset(Dataset):
    def __init__(self, input_ids_a, input_ids_b, labels, present=None):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.labels = labels
        self.present = present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return (torch.tensor(self.input_ids_a[i], dtype=torch.long), 
                    torch.tensor(self.input_ids_b[i], dtype=torch.long), 
                    torch.tensor(self.labels[i], dtype=torch.long), 
                    self.present[i])
        else:
            return (torch.tensor(self.input_ids_a[i], dtype=torch.long), 
                    torch.tensor(self.input_ids_b[i], dtype=torch.long), 
                    torch.tensor(self.labels[i], dtype=torch.long))

def collate(examples):
    padding_value = 0

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    second_sentence = [t[1] for t in examples]
    second_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        second_sentence, batch_first=True, padding_value=padding_value)

    max_length = second_sentence_padded.shape[1]
    second_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[1]), dtype=torch.long), torch.zeros(
        max_length - len(t[1]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([t[2] for t in examples])

    return (first_sentence_padded, first_sentence_attn_masks, 
            second_sentence_padded, second_sentence_attn_masks, 
            labels)

def train(args, train_dataset, valid_dataset, model, tokenizer, labels, lang):

    # Prepare train data
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate)
    train_batch_size = args.train_batch_size

    # Prepare optimizer
    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_f1_score = 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_dir = os.path.join(args.output_dir, lang + "" + args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids_a': batch[0],
                      'attention_mask_a': batch[1],
                      'input_ids_b': batch[2],
                      'attention_mask_b': batch[3],
                      'labels': batch[4]}
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        # Checking for validation accuracy and stopping after drop in accuracy for 3 epochs
        results, _ = evaluate(args, model, tokenizer, labels, 'validation', lang=lang)
        if results.get('f1') > best_f1_score and args.save_steps > 0:
            logger.info("Saving model with f1 score {}".format(results.get('f1')))
            best_f1_score = results.get('f1')
            # model_to_save = model.module if hasattr(model, "module") else model
            model_path = os.path.join(model_dir, "modelfile")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            torch.save(model, model_path)
            os.system("cp -r {} {}_{}".format(model_dir, model_dir, str(best_f1_score)[:6]))


    if results.get('f1') < best_f1_score and args.save_steps > 0:
        mpath = "{}_{}".format(model_dir, str(best_f1_score)[:6])
        model_path = os.path.join(mpath, "modelfile")
        logger.info("Loading best model with f1 score {}".format(best_f1_score))
        #model_to_load = model.module if hasattr(model, "module") else model
        model = torch.load(model_path)
    
    results, _ = evaluate(args, model, tokenizer, labels, 'validation', lang=lang)

    return global_step, tr_loss / global_step, model

def evaluate(args, model, tokenizer, labels, mode, lang, prefix=""):

    eval_dataset = load_and_cache_examples(args, tokenizer, labels, mode=mode, lang=lang)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate)
    results = {}

    # Evaluation
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids_a': batch[0],
                      'attention_mask_a': batch[1],
                      'input_ids_b': batch[2],
                      'attention_mask_b': batch[3],
                      'labels': batch[4]}
            '''print(inputs["input_ids"])
            print(inputs["attention_mask"])
            print(inputs["token_type_ids"])'''
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            # preds = logits.detach().cpu().numpy()
            preds = logits.softmax(dim=-1).detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            preds = np.append(
                preds, 
                logits.softmax(dim=-1).detach().cpu().numpy(), 
                axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    predmax = np.max(preds, axis=1)
    preds = np.argmax(preds, axis=1)
    preds_list = []
    label_map = {i: label for i, label in enumerate(labels)}

    ex_len = get_example_length(args.data_dir, mode, lang)
    for i in range(out_label_ids.shape[0]):
        if mode is "test" and eval_dataset[i][2] == 0:
            preds_list.append('not found')
        else:
            preds_list.append(
                label_map[preds[i]] + " " + 
                label_map[out_label_ids[i]] + " " + 
                str(predmax[i]) + " " +
                str(ex_len[i][0]) + " " + str(ex_len[i][1]))

    logger.info(" test data counuter : \n preds {} \n labels {} \n".format(Counter(list(preds)), Counter(out_label_ids)))
    if mode == "test":
        return preds_list
    else:
        result = acc_and_f1(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results %s *****", mode)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        return results, preds_list

class TwoLangBertForSeqClassification(torch.nn.Module):
    def __init__(self, bert_lang_a, bert_lang_b, 
            hidden_size=1536, do=0.1, n_labels=3):
        super().__init__()
        self.bert_lang_a = bert_lang_a
        self.bert_lang_b = bert_lang_b
        self.dropout = torch.nn.Dropout(do)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, n_labels)
        self.n_labels = n_labels

    def forward(
            self,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            labels=None,
            ret_pooled_output=False,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None):
        """
        Pass the inputs to the corresponding lang BERT models respectively.
        """
        
        input_dict = {'input_ids': input_ids_a, 
                      'attention_mask': attention_mask_a,
                      'ret_pooled_output': True}
        outputs_a = self.bert_lang_a(**input_dict)
        
        input_dict = {'input_ids': input_ids_b, 
                      'attention_mask': attention_mask_b,
                      'ret_pooled_output': True}
        outputs_b = self.bert_lang_b(**input_dict)
        
        outputs_concatenated = self.dropout(self.bn1(torch.cat([outputs_a, outputs_b], dim=1)))
        # outputs_concatenated = self.dropout(torch.cat([outputs_a, outputs_b], dim=1))
        logits = self.classifier(outputs_concatenated)
        outputs = (logits, outputs_concatenated)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def save_pretrained(self, mpath):
        # Save BERT with lang model A.
        model_dir = os.path.join(mpath, "lang_a")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.bert_lang_a.save_pretrained(model_dir)

        # Save BERT with lang model B.
        model_dir = os.path.join(mpath, "lang_b")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.bert_lang_b.save_pretrained(model_dir)

    def from_pretrained(self, mpath):
        # Save BERT with lang model A.
        model_dir = os.path.join(mpath, "lang_a")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.bert_lang_a.from_pretrained(model_dir)

        # Save BERT with lang model B.
        model_dir = os.path.join(mpath, "lang_b")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.bert_lang_b.from_pretrained(model_dir)

def main():

    parser = create_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # Print args
    logging.info(args)

    # Set seed
    set_seed(args)

    # Prepare data
    labels = get_labels(args.data_dir)
    labels = ["positive", "negative", "neutral"]
    num_labels = len(labels)

    # Initialize model
    tokenizer_class = {"xlm": XLMTokenizer, "bert": BertTokenizer, "xlm-roberta": XLMRobertaTokenizer}
    if args.model_type not in tokenizer_class.keys():
        print("Model type has to be xlm/xlm-roberta/bert")
        exit(0)
    tokenizer = tokenizer_class[args.model_type].from_pretrained(
        args.model_name, do_lower_case=True)
    model_class = {"xlm": XLMForSequenceClassification, "bert": BertForSequenceClassification, "xlm-roberta": XLMRobertaForSequenceClassification}
    if args.model_path:
        model_a = model_class[args.model_type].from_pretrained(
            args.model_path)
        model_b = model_class[args.model_type].from_pretrained(
            args.model_path)
    else:
        model_a = model_class[args.model_type].from_pretrained(
            args.model_name, num_labels=num_labels)
        model_b = model_class[args.model_type].from_pretrained(
            args.model_name, num_labels=num_labels)

    model = TwoLangBertForSeqClassification(model_a, model_b)
    model.to(args.device)

    # Training

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="train", lang=args.lang)
    valid_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="validation", lang=args.lang)

    # Starting results.
    # result, preds_val = evaluate(args, model, tokenizer, labels, mode="validation", lang=args.lang)

    if not args.eval_only:
        global_step, tr_loss, model = train(
            args, train_dataset, valid_dataset, model, tokenizer, labels, lang=args.lang)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation

    results = {}

    result, preds_val = evaluate(args, model, tokenizer, labels, mode="validation", lang=args.lang)
    preds = evaluate(args, model, tokenizer, labels, mode="test", lang=args.lang)

    # Saving predictions
    output_test_predictions_file = os.path.join(args.output_dir, 
                                                args.output_file_prefix + args.lang + "_test_predictions2.txt")
    with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
        writer.write('\n'.join(preds))

    output_test_predictions_file = os.path.join(args.output_dir, 
                                                args.output_file_prefix + args.lang + "_val_predictions2.txt")
    with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
        writer.write('\n'.join(preds_val))

    return results

# twoBert + BN : 0.5546
# hin : 0.5395
# eng : 0.545

if __name__ == "__main__":
    main()
