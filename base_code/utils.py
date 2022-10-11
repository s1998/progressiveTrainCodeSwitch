from __future__ import absolute_import
from __future__ import print_function

from sklearn.metrics import classification_report

import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras

import tensorflow as tf
import keras.backend as ktf
from tensorflow.python.keras import backend as ktf

def get_session(gpu_fraction=0.2):
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

DataPth = "../Data/Processed_Data/Sentiment_EN_HI/Devanagari/"

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

def read_examples_from_file(data_dir, mode, lang="switched", discard_neutral=False):
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
        if label == "neutral" and discard_neutral:
            if (mode=="train" or mode=="validation"):
                continue
            # For test set, we need to modify neutal to positive FixMe hack
            label = "positive"

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

    # if mode == "train":
    #     logger.info("\nTraining examples for language {} count : {}, "
    #                 "actual count : {} \n".format(lang, len(examples), len(lines)))

    return examples

