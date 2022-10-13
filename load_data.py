import csv
import numpy as np
import os
import re
import itertools
from collections import Counter
from os.path import join
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from base_code.utils import *

def read_file(data_dir, with_evaluation, lbl_to_id_chng=True):
    data = []
    target = []
    
    if data_dir in ["sail", "enes", "taen", ]:
        train_data = read_examples_from_file("./data" + data_dir, "train")
        valid_data = read_examples_from_file("./data" + data_dir, "validation")
        labels = ["negative", "positive"]
        lbl_to_id = {lbl:i for i, lbl in enumerate(labels)}
        
        for lin in train_data:
            if lin["label"] == "neutral":
                continue
            data.append(lin["text"])
            if lbl_to_id_chng:
                target.append(lbl_to_id[lin["label"]])
            else:
                target.append(lin["label"])
        
        for lin in valid_data:
            if lin["label"] == "neutral":
                continue
            data.append(lin["text"])
            if lbl_to_id_chng:
                target.append(lbl_to_id[lin["label"]])
            else:
                target.append(lin["label"])
    else:
        raise NotImplementedError("Data {} not found .... ".format(data_dir))

    if with_evaluation:
        y = np.asarray(target)
        assert len(data) == len(y)
        if lbl_to_id_chng:
            assert set(range(len(np.unique(y)))) == set(np.unique(y))
    else:
        y = None

    return data, y

def clean_str(string):
    string = re.sub(r"[^A-Za-z\u0900-\u097F0-9(),.!?_\"\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess_doc(data):
    data = [s.strip() for s in data]
    data = [clean_str(s) for s in data]
    return data
