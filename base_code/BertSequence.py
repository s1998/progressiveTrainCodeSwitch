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
    get_linear_schedule_with_warmup, BertPreTrainedModel, BertModel
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    f1micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    precision = precision_score(
        y_true=labels, y_pred=preds, average='weighted')
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    clfreport = classification_report(y_true=labels, y_pred=preds, 
                                      target_names=["positive", "negative", "neutral"])
    return{
        "acc": acc,
        "f1": f1,
        "f1micro": f1micro,
        "f1macro": f1macro,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall,
        "clfreport": clfreport,
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
    assert(lang in ["eng", "hin", "switched", "es"])
    with open(file_path, 'r', encoding="utf-8") as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        text = x[0]
        label = x[1]
        if lang != "switched":
            if lang == "eng":
                eng_only = re.sub(r'[\u0900-\u097F]+', '', text)
                if len(eng_only.split()) <= 3:
                    eng_only += " shortTextHere"
                text = eng_only
            if lang == "hin":
                hin_only = re.sub(r'[A-Za-z]+', '', text)
                if len(hin_only.split()) <= 3:
                    hin_only += " shortTextHere"
                text = hin_only
        examples.append({'text': text, 'label': label})

    if mode == 'test':
        for i in range(len(examples)):
            if examples[i]['text'] == 'not found':
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

        sentence = example['text']
        label = example['label']

        sentence_tokens = tokenizer.tokenize(sentence)[:max_seq_length - 2]
        sentence_tokens = [tokenizer.cls_token] + \
            sentence_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        label = label_map[label]
        features.append({'input_ids': input_ids,
                         'label': label})
        if 'present' in example:
            features[-1]['present'] = example['present']

    return features


def collate(examples):
    padding_value = 0

    first_sentence = [t[0] for t in examples]
    first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
        first_sentence, batch_first=True, padding_value=padding_value)

    max_length = first_sentence_padded.shape[1]
    first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
        max_length - len(t[0]), dtype=torch.long)]) for t in examples])

    labels = torch.stack([t[1] for t in examples])

    return first_sentence_padded, first_sentence_attn_masks, labels


def load_and_cache_examples(args, tokenizer, labels, mode, lang, examples=None):

    logger.info("Creating features from dataset file at %s", args.data_dir)
    if examples is None:
        examples = read_examples_from_file(args.data_dir, mode, lang)
    else:
        print("Using the passed examples & not the file path.")
    features = convert_examples_to_features(examples, labels, tokenizer, args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f['input_ids'] for f in features]
    all_labels = [f['label'] for f in features]
    args = [all_input_ids, all_labels]
    if 'present' in features[0]:
        present = [1 if f['present'] else 0 for f in features]
        args.append(present)

    dataset = CustomDataset(*args)
    return dataset

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
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
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
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            torch.save(args, os.path.join(
                model_dir, "training_args.bin"))
            os.system("cp -r {} {}_{}".format(model_dir, model_dir, str(best_f1_score)[:6]))


    if results.get('f1') < best_f1_score and args.save_steps > 0:
        mpath = "{}_{}".format(model_dir, str(best_f1_score)[:6])
        logger.info("Loading best model with f1 score {}".format(best_f1_score))
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.from_pretrained(mpath)
        model = model_to_load
        results, _ = evaluate(args, model_to_load, tokenizer, labels, 'validation', lang=lang)  
    
    results, _ = evaluate(args, model, tokenizer, labels, 'validation', lang=lang)

    return global_step, tr_loss / global_step, model


def evaluate(args, model, tokenizer, labels, mode, lang, prefix="", dataset=None, 
             nb=False, addlen=False):

    if dataset is None:
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, mode=mode, lang=lang)
    else:
        print("Using the dataset passed to the function.")
        eval_dataset = dataset

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

    if nb:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2]}
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
            tempstr = label_map[preds[i]] + " " + \
                      label_map[out_label_ids[i]] + " " + \
                      str(predmax[i])
            if addlen:
                tempstr += str(ex_len[i][0]) + " " + str(ex_len[i][1])
            preds_list.append(tempstr)

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


class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, present=None):
        self.input_ids = input_ids
        self.labels = labels
        self.present = present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)

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
        model = model_class[args.model_type].from_pretrained(
            args.model_path)
    else:
        model = model_class[args.model_type].from_pretrained(
            args.model_name, num_labels=num_labels)

    model.to(args.device)

    # Training

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="train", lang=args.lang)
    valid_dataset = load_and_cache_examples(
        args, tokenizer, labels, mode="validation", lang=args.lang)

    # Starting results.
    result, preds_val = evaluate(args, model, tokenizer, labels, mode="validation", lang=args.lang)

    if not args.eval_only:
        global_step, tr_loss, model = train(
            args, train_dataset, valid_dataset, model, tokenizer, labels, lang=args.lang)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation

    results = {}

    result, preds_val = evaluate(args, model, tokenizer, labels, mode="validation", lang=args.lang)
    preds = evaluate(args, model, tokenizer, labels, mode="test", lang=args.lang)

    # Saving predictions
    # Include everything, probability, length, and prediction.
    output_test_predictions_file = os.path.join(args.output_dir, 
                                                args.output_file_prefix + args.lang + "_test_predictions2.txt")
    with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
        writer.write('\n'.join(preds))
    # Write the prediction for easy submission.
    output_test_predictions_file = os.path.join(args.output_dir, 
                                                args.output_file_prefix + args.lang + "_test_predictions1.txt")
    with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
        writer.write('\n'.join([pred.split()[0] for pred in preds]))

    output_test_predictions_file = os.path.join(args.output_dir, 
                                                args.output_file_prefix + args.lang + "_val_predictions2.txt")
    with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
        writer.write('\n'.join(preds_val))

    return results

# eng only f1: f1 = 0.4841052388357277 
# hin only f1: f1 = 0.5231766634955017
# hin Training examples for language hin count : 7931, actual count : 10079
# eng Training examples for language eng count : 6009, actual count : 10079


if __name__ == "__main__":
    main()
