from numpy.lib.function_base import append
from utils import textDataset, train_fn, evaluate_fn, f1
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
from collections import Counter

class ProbeRunner(object):
    def __init__(self, ds_pretrain_dataset, dataset, model_name="bert-base-multilingual-cased"):
        print("\n\nInitialising the ds trainer object .... ")

        print("Creating tokenizers .... ")
        self.model_name = model_name
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._create_dataloaders(ds_pretrain_dataset, dataset)
        self.model_m = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model_m1, self.model_m2 = None, None

    def _create_dataloaders(self, ds_pretrain_dataset, dataset):
        print("Creating dataloaders .... ")
        lbls = ["negative", "positive"]
        self.lbl_to_ind = {lbl:i for i, lbl in enumerate(lbls)}
        self.ind_to_lbl = {i:lbl for i, lbl in enumerate(lbls)}

        # Preprocess the ds dataset
        self.ds_x, self.ds_y = ds_pretrain_dataset
        self.ds_y = [self.lbl_to_ind[lbl] for lbl in self.ds_y]
        self.ds_x_tr, self.ds_x_vl, self.ds_y_tr, self.ds_y_vl = \
            train_test_split(self.ds_x, self.ds_y, test_size=0.2, random_state=42)
        self.ds_x_tr, self.ds_x_vl = self._preprocess_dataset(self.ds_x_tr), self._preprocess_dataset(self.ds_x_vl)

        # Preprocess the supervised dataset
        self.x, self.y = dataset
        self._old_x = copy.deepcopy(self.x)
        self.y = [self.lbl_to_ind[lbl] for lbl in self.y]
        self.x_tr, self.x_vl, self.y_tr, self.y_vl = \
            train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.x_tr, self.x_vl = self._preprocess_dataset(self.x_tr), self._preprocess_dataset(self.x_vl)

        self.ds_train_dl = DataLoader(textDataset(self.ds_x_tr, self.ds_y_tr), shuffle=True, batch_size=64)
        self.ds_eval_dl  = DataLoader(textDataset(self.ds_x_vl, self.ds_y_vl), shuffle=False, batch_size=64)
        self.s_train_dl = DataLoader(textDataset(self.x_tr, self.y_tr), shuffle=True, batch_size=64)
        self.s_eval_dl  = DataLoader(textDataset(self.x_vl, self.y_vl), shuffle=False, batch_size=64)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _preprocess_dataset(self, x):
        return self.model_tokenizer(x, truncation=True, padding=True, max_length=128)

    def train_s(self):
        del self.model_m1
        torch.cuda.empty_cache()
        print("\nTraining new supervised model on the CS dataset")
        self.model_m2 = copy.deepcopy(self.model_m)
        self.model_m2.to(self.device)
        print("\nCreated model m2 ....")
        train_fn(self.model_m2, self.s_train_dl, self.s_eval_dl, self.device, n_epochs=1, load_best=False, eval_per_epoch=1)
        y_pseudo = evaluate_fn(self.model_m2, self.s_eval_dl, "us eval", self.device, return_lbls=True)
        print("\nSelftraining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.y_vl), np.array(y_pseudo), True)
        print("shingeki Scores obtained after just training CS sup f1 macro / micro / weighted : {} / {} / {}".format(
            f1_macro, f1_micro, f1_weighted))
        print("shingeki Doing bucketwise evaluation ....")
        self.evaluate_bkt_wise(self.model_m2)

    def pretrain_ds(self):
        print("Creating model m1 .... ")
        self.model_m1 = copy.deepcopy(self.model_m)
        self.model_m1.to(self.device)
        print("Created model m1 ....")
        print("Pretraining on the mined data .... ")
        self.model_m1 = train_fn(self.model_m1, self.ds_train_dl, self.ds_eval_dl, self.device, n_epochs=1, eval_per_epoch=1)
        print("Evaluation of the ds trained model on ds dataset .... ")
        y_pseudo = evaluate_fn(self.model_m1, self.ds_eval_dl, "us eval", self.device, return_lbls=True)
        print("Pretrining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.ds_y_vl), np.array(y_pseudo), True)
        print("Scores obtained on ds dataset after pretraining f1 micro / macro / weighted : {:.3f} / {:.3f} / {:.3f}".format(
            f1_macro, f1_micro, f1_weighted))

    def train_pretrained_s(self):
        del self.model_m2
        torch.cuda.empty_cache()
        print("\n\n\n\nTraining pretrained supervised model on the CS dataset")
        train_fn(self.model_m1, self.s_train_dl, self.s_eval_dl, self.device, n_epochs=1, load_best=False, eval_per_epoch=1)
        print("\n\nFinal results of pretrained & then sup trained on CS dataset .... ")
        y_pseudo = evaluate_fn(self.model_m1, self.s_eval_dl, "us eval", self.device, return_lbls=True)
        print("\n\nSelftraining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.y_vl), np.array(y_pseudo), True)
        print("shingeki Scores obtained after pretraining + sup training f1 macro / micro / weighted : {} / {} / {}".format(f1_macro, f1_micro, f1_weighted))
        print("shingeki Doing bucketwise evaluation ....")
        self.evaluate_bkt_wise(self.model_m1)


    def evaluate_bkt_wise(self, model, bkt_cnt=4):
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
        
        aa = zip(hin_eng_count(self._old_x)[:, 0], self._old_x, self.y, self.y)
        aa_sorted_e2h = sorted(aa, reverse=True)
        x_n, y_n_pseudo, y_n = [a[1] for a in aa_sorted_e2h], [a[2] for a in aa_sorted_e2h], [a[3] for a in aa_sorted_e2h]

        # create a new data loader with the unsupervised data, this has to be in
        # the order of new x, need this to get the pseudo labels.
        self.us_eval_dl  = DataLoader(textDataset(self._preprocess_dataset(x_n) , y_n),
                                shuffle=False, batch_size=64)   

        bkt_sz = len(x_n) // bkt_cnt

        bkts_vl = []
        for i in range(bkt_cnt):
            if i == bkt_cnt-1:
                bkt_x, bkt_y = (x_n[i*bkt_sz:], y_n[i*bkt_sz:])
            else:
                bkt_x, bkt_y = (x_n[i*bkt_sz:(i+1)*bkt_sz], y_n[i*bkt_sz:(i+1)*bkt_sz])
            
            assert(len(bkt_x) == len(bkt_y))
            print("Created bucket {} w/ size {}".format(i, len(bkt_y)))            
            bkt_x = self._preprocess_dataset(bkt_x)
            bkt_vl_dl = DataLoader(textDataset(bkt_x, bkt_y), shuffle=False, batch_size=64)
            bkts_vl.append((bkt_vl_dl, bkt_y))

        scrs = []
        scrs_bkt = {}
        for j, (bkt, bkt_y) in enumerate(bkts_vl):
            print("      Evaluating model for bkt {} .... ".format(j))
            bkt_y_pred = evaluate_fn(model, bkt, "us eval",  self.device, return_lbls=True)
            scrs_bkt[j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)

        print("shingeki bktwise score .... ", scrs_bkt)
