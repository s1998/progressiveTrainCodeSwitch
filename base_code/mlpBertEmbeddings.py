from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras

import pickle
import numpy as np
from BertSequence import *
from sklearn.metrics import classification_report

from tensorflow.python.keras import backend as ktf
import tensorflow as tf

# Set the fraction of GPU to be used.
def get_session(gpu_fraction=0.2):
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())



engpath = "ResultsA/Sentiment_EN_HI/eng_output_vec_dict"
hinpath = "ResultsA/Sentiment_EN_HI/hin_output_vec_dict"
swipath = "ResultsA/Sentiment_EN_HI/switched_output_vec_dict"

paths = [("eng", engpath), ("swi", swipath), ("hin", hinpath)]

model_vec_dict = {}
for lang, pth in paths:
    with open(pth, "rb") as f:
        model_vec_dict[lang] = pickle.load(f)

print(np.hstack([model_vec_dict['eng']['train'], 
                 model_vec_dict['swi']['train']]).shape)

traindata = np.hstack([model_vec_dict['hin']['train'], 
                       model_vec_dict['eng']['train'], 
                       model_vec_dict['swi']['train']])
testdata = np.hstack([model_vec_dict['hin']['test'], 
                      model_vec_dict['eng']['test'], 
                      model_vec_dict['swi']['test']])
validdata = np.hstack([model_vec_dict['hin']['validation'], 
                       model_vec_dict['eng']['validation'], 
                       model_vec_dict['swi']['validation']])

print(traindata.shape, testdata.shape, validdata.shape)

labels = ["positive", "negative", "neutral"]
num_labels = len(labels)

label_list = ["positive", "negative", "neutral"]
label_map = {label: i for i, label in enumerate(label_list)}
train_y = [label_map[ex['label']] for ex in 
    read_examples_from_file("./Data/Processed_Data/Sentiment_EN_HI/Devanagari", 
    "train", "switched")]
valid_y = [label_map[ex['label']] for ex in 
    read_examples_from_file("./Data/Processed_Data/Sentiment_EN_HI/Devanagari", 
    "validation", "switched")]
test_y = [label_map[ex['label']] for ex in 
    read_examples_from_file("./Data/Processed_Data/Sentiment_EN_HI/Devanagari", 
    "test", "switched")]
print(len(train_y), len(valid_y), len(test_y))

np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_classes = 3
nb_epoch = 5

Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)
Y_valid = np_utils.to_categorical(valid_y, nb_classes)

def get_model(cfg):
    model = Sequential()
    model.add(Dense(cfg["l1_size"]))
    model.add(Activation('relu'))
    model.add(Dropout(cfg["do"]))
    model.add(Dense(cfg["l2_size"]))
    model.add(Activation('relu'))
    model.add(Dropout(cfg["do"]))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    adam = Adam(learning_rate=cfg["lr"])
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    return model

def train_model(trial):
    cfg = {
            'lr'      : trial.suggest_loguniform('lr', 1e-9, 1e-5),
            'l1_size' : trial.suggest_int('l1_size', 768, 1024, 64), 
            'l2_size' : trial.suggest_int('l2_size', 64, 256, 64), 
            'do'      : trial.suggest_uniform('do', 0.1, 0.2), 
          }
    model = get_model(cfg)
    model.fit(traindata, np.array(Y_train),
              batch_size=batch_size, epochs=nb_epoch, 
              verbose=2,
              validation_data=(validdata, np.array(Y_valid)))
    score = model.evaluate(validdata, np.array(Y_valid), verbose=3)
    return score

import optuna, joblib

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=sampler, direction='minimize')
study.optimize(func=train_model, n_trials=2)
joblib.dump(study, './bert_emb_mlp.pkl')

df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
print(df)

# optuna.visualization.plot_parallel_coordinate(study,params=['lr','l1_size', 'l2_size', 'do'])
