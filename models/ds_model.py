from numpy.lib.function_base import append
from utils import textDataset, train_fn, evaluate_fn, f1, hin_eng_count, \
    get_ids, set_seed, es_eng_count, get_ids_ratio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import copy, time
import numpy as np
from collections import Counter
import math 

class DsTrainer(object):
    def __init__(self, ds_pretrain_dataset, dataset, 
                 model_name="bert-base-multilingual-cased", weightedloss=False,
                 datasetname="sail", seed=None):
        print("\n\nInitialising the ds trainer object .... ")

        print("Creating tokenizers .... ")
        self.model_name = model_name
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._create_dataloaders(ds_pretrain_dataset, dataset)
        self.weightedloss = weightedloss
        self.datasetname  = datasetname
        self.seed = seed

        dataset_to_cnt_frac_fn_map = {
            "sail" : hin_eng_count,
            "enes" : es_eng_count,
            "taen" : es_eng_count
        }
        self.cnt_frac_fn = dataset_to_cnt_frac_fn_map[datasetname]

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
        self.ds_x_tr_txt = copy.deepcopy(self.ds_x_tr)
        self.ds_x_tr = self._preprocess_dataset(self.ds_x_tr)
        self.ds_x_vl = self._preprocess_dataset(self.ds_x_vl)

        # Preprocess the us dataset
        self.x, self.y = dataset
        self._old_x = copy.deepcopy(self.x)
        self.x = self._preprocess_dataset(self.x)
        self.y = [self.lbl_to_ind[lbl] for lbl in self.y]

        self.ds_train_dl = DataLoader(textDataset(self.ds_x_tr, self.ds_y_tr), 
                                      shuffle=True, batch_size=64)
        self.ds_eval_dl  = DataLoader(textDataset(self.ds_x_vl, self.ds_y_vl),
                                      shuffle=False, batch_size=64)
        self.us_eval_dl  = DataLoader(textDataset(self.x, self.y),
                                      shuffle=False, batch_size=64)
        self.us_train_dl = None

        print("Creating models .... ")
        self.model_m1 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_m1.to(self.device)
        print("\n\nCreated model m1 ....")
        # Create a placeholder for the model to be fine tuned with real data. 
        # Can we use M1 here directly?
        self.model_m2 = None
        self.pretrain_done = False

    def _preprocess_dataset(self, x):
        return self.model_tokenizer(x, truncation=True, padding=True, max_length=128)

    def pretrain(self, n_epochs=5):
        print("\n\nPretraining on the mined data .... ")

        wts = None
        if self.weightedloss:
            cnts = Counter(self.ds_y_tr)
            dnmr = (cnts[0] + cnts[1])
            wts = torch.Tensor([cnts[0]/dnmr, cnts[1]/dnmr]).to(self.device)
        
        assert(self.seed is not None)
        if self.model_name == "bert-base-multilingual-cased":
            pth = "./savedmodels/{}_{}".format(self.datasetname, self.seed)
        else:
            pth = "./savedmodels/{}_{}_{}".format(self.model_name, self.datasetname, self.seed)
        print("randomtexthereforgreppurposesBBBBBBBB model pth {}".format(pth))
        import os
        if os.path.exists(pth):
            print("Loading from existing checkpint using pth : {}".format(pth))
            model_bl = AutoModelForSequenceClassification.from_pretrained(self.model_name); model_bl.load_state_dict(torch.load(pth)); model_bl.to(self.device)
            self.model_m1 = model_bl
        else:
            print("Creating new model and pretraining")
            self.model_m1 = train_fn(
                self.model_m1, self.ds_train_dl, self.ds_eval_dl, self.device, 
                n_epochs=n_epochs, weights=wts, eval_per_epoch=1)
            torch.save(self.model_m1.state_dict(), pth)
        print("\n\nEvaluation of the ds trained model on ds dataset .... ")
        evaluate_fn(self.model_m1, self.ds_eval_dl, "ds eval", self.device)
        print("\n\nEvaluation of the ds trained model on us dataset .... ")
        self.y_pseudo = evaluate_fn(self.model_m1, self.us_eval_dl, "us eval", 
                                    self.device, return_lbls=True)
        print("\n\nPretrining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.y), np.array(self.y_pseudo), True)
        print("Scores obtained after pretraining f1 micro / macro / weighted : {:.3f} / {:.3f} / {:.3f}".format(
            f1_macro, f1_micro, f1_weighted))

    def _create_us_buckets(self, x_n, y_n, bkt_cnt):
        bkt_sz = len(x_n) // bkt_cnt

        bkts_vl = []
        for i in range(bkt_cnt):
            if i == bkt_cnt-1:
                bkt_x, bkt_y = (x_n[i*bkt_sz:], y_n[i*bkt_sz:])
            else:
                bkt_x, bkt_y = (x_n[i*bkt_sz:(i+1)*bkt_sz], y_n[i*bkt_sz:(i+1)*bkt_sz])
            
            assert(len(bkt_x) == len(bkt_y))
            print("Created bucket {} w/ size {}".format(i, len(bkt_y)))            
            bkt_vl_dl = DataLoader(textDataset(self._preprocess_dataset(bkt_x), bkt_y), shuffle=False, batch_size=64)
            bkts_vl.append((bkt_vl_dl, bkt_x, bkt_y))

        return bkts_vl

    def _get_scrs(self, model, bkts_vl, nm=""):
        scrs_bkt_curr = {}
        for j, (bkt, _, bkt_y) in enumerate(bkts_vl):
            print("      Evaluating model for bkt {} .... ".format(j))
            bkt_y_pred = evaluate_fn(model, bkt, "{} us eval".format(nm),  self.device, return_lbls=True)
            scrs_bkt_curr[j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True, False)
            print("Predicted y cntr : {}".format(Counter(bkt_y_pred)))
            print("Actual    y cntr : {}".format(Counter(bkt_y)))
            print("\n\n")
        return scrs_bkt_curr

    def supervised_upperbound(self):
        print("Doing supervised training to get the upperboung using ....."
              " model {}    for dataset   {} ".format(self.model_name, self.datasetname))
        # create the train-test split
        x_tr, x_te, y_tr, y_te = train_test_split(self._old_x, self.y, 
            test_size=0.2, random_state=42)

        # divide the train split into train-validation split
        x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr, y_tr, 
            test_size=0.2, random_state=42)

        # train dl, valid dl, test dl

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name); 
        model.to(self.device)

        train_dl = DataLoader(textDataset(self._preprocess_dataset(x_tr) , y_tr), 
                              shuffle=True, batch_size=64)
        valid_dl = DataLoader(textDataset(self._preprocess_dataset(x_vl) , y_vl), 
                              shuffle=False, batch_size=64)
        test__dl = DataLoader(textDataset(self._preprocess_dataset(x_te) , y_te), 
                              shuffle=False, batch_size=64)
        model_trained  =  train_fn(model, train_dl, valid_dl, 
            self.device, n_epochs=4, load_best=True, eval_per_epoch=1, eval_train=False)

        y_te_pr = evaluate_fn(model, test__dl, 
                        " randomtexthereforgreppurposesBBBBBBBB  model m test eval", self.device, return_lbls=True)
            
        print("    randomtexthereforgreppurposesAAAAAAA    ")
        print("Trained model in supervised fashion to obtain a final test score of .....   {}".format(
            f1(np.array(y_te), np.array(y_te_pr), True)
        ))

    def get_ood_pct(self, model, valid_data_dl, new_data_dl):
        # getting OOD on the current valid set
        y_pseudo_pr = evaluate_fn(model, valid_data_dl, " ood valid ", self.device, return_probs=True)
        y_pseudo_pr = np.max(np.array(y_pseudo_pr), axis=1)
        percentiles = np.percentile(y_pseudo_pr, [10,5,1])
        print("probability array shape {}".format(y_pseudo_pr.shape))
        a, b, c = percentiles[0], percentiles[1], percentiles[2]
        print("percentiles 1/5/10 : {}/{}/{}".format(a, b, c))
        y_pseudo_pr = evaluate_fn(model, new_data_dl, " ood valid new ", self.device, return_probs=True)
        y_pseudo_pr = np.max(np.array(y_pseudo_pr), axis=1)
        ood1, ood5, ood10 = np.sum(y_pseudo_pr < a) * 1.0 / len(y_pseudo_pr), np.sum(y_pseudo_pr < b) * 1.0 / len(y_pseudo_pr), np.sum(y_pseudo_pr < c) * 1.0 / len(y_pseudo_pr)
        print("percentiles 1/5/10 : {}/{}/{} ood fraction : {}/{}/{}".format(a, b, c, ood1, ood5, ood10))
        return ood1, ood5, ood10

    def selftrain_bucketing_different_m_merged_data_half(self, bkt_cnt=2):
        aa = zip(self.cnt_frac_fn(self._old_x)[:, 0], self._old_x, self.y_pseudo, self.y)
        aa_sorted_e2h = sorted(aa, reverse=True)
        x_n, y_n_pseudo, y_n = [a[1] for a in aa_sorted_e2h], [a[2] for a in aa_sorted_e2h], [a[3] for a in aa_sorted_e2h]

        # create a new data loader with the unsupervised data, this has to be in
        # the order of new x, need this to get the pseudo labels.
        self.us_eval_dl  = DataLoader(textDataset(self._preprocess_dataset(x_n) , y_n), shuffle=False, batch_size=64)   

        bkt_sz = len(x_n) // bkt_cnt
        bkts_vl = self._create_us_buckets(x_n, y_n, bkt_cnt)

        ood_res = {"pt" : {}, "st1" : {}}
        for i in range(bkt_cnt):
            bkt_dl = bkts_vl[i][0]
            ood_res["pt"][i] = self.get_ood_pct(self.model_m1, self.ds_eval_dl, bkt_dl)

        scrs = []
        scrs_bkt = {}
        
        scrs_bkt[-2] = self._get_scrs(self.model_m1, bkts_vl)
        scrs.append(("pt", f1(np.array(y_n), np.array(y_n_pseudo), True)))

        import random
        PATH = "logs/tempmodel_{}_{}".format(random.randint(1, 100), float(time.time()))
        torch.save(self.model_m1.state_dict(), PATH)
        # model_bl = AutoModelForSequenceClassification.from_pretrained(self.model_name); model_bl.load_state_dict(torch.load(PATH)); model_bl.to(self.device)
        ids, y_n_pseudo, _ = get_ids_ratio(self.us_eval_dl, y_n, self.model_m1, self.device)
        us_pseudo_train_ds = textDataset(self._preprocess_dataset([x_n[id_] for id_ in ids]+ self.ds_x_tr_txt), y_n_pseudo + self.ds_y_tr)
        self.us_pseudo_train_dl = DataLoader(us_pseudo_train_ds, shuffle=True, batch_size=64)
        
        bkt_ids = {}
        bkt_ids[0] = [id_ for id_ in ids if id_ < len(y_n) // 2]
        bkt_ids[1] = [id_ for id_ in ids if id_ > len(y_n) // 2]

        print("randomtexthereforgreppurposesBBBBBBBB -- bktwise count selected : {} ".format({k:len(bkt_ids[k]) for k in bkt_ids}))

        del model_bl
        torch.cuda.empty_cache()

        final_y_pred = []
        # self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name); self.model_m2.load_state_dict(torch.load(PATH)); self.model_m2.to(self.device)
        bkt_y_pred_dict = {}
        prevx, prevy = [], []
        prevmodel = self.model_m1
        print("randomtexthereforgreppurposesBBBBBBBB scrs bkt \n" + "\n".join(str(scr_bkt) +  "   " + str(scrs_bkt[scr_bkt]) for scr_bkt in scrs_bkt))
        for i in range(bkt_cnt):
            print("\n\n" + "*********" * 10)
            print("\n\nrandomtexthereforgreppurposesBBBBBBBB Self training w/ bucket {} .... ".format(i))

            # get predictions on the bkt i            
            y_pred = evaluate_fn(prevmodel, self.us_eval_dl, " ** ignore ** model 2 us eval bkt {}".format(i), self.device, return_lbls=True)
            del prevmodel
            torch.cuda.empty_cache()
            self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name);  self.model_m2.to(self.device); self.model_m2.load_state_dict(torch.load(PATH))
            prevx.extend([x_n[id_] for id_ in bkt_ids[i]]); prevy.extend([y_pred[id_] for id_ in bkt_ids[i]])
            curr_trn_ds = textDataset(self._preprocess_dataset(prevx + self.ds_x_tr_txt), prevy + self.ds_y_tr)
            print("randomtexthereforgreppurposesBBBBBBBB Bucket / total size : {} / {}".format(len(prevy), len(prevy + self.ds_y_tr)))
            self.us_pseudo_train_dl = DataLoader(curr_trn_ds, shuffle=True, batch_size=64)
            train_fn(self.model_m2, self.us_pseudo_train_dl, self.us_eval_dl, self.device, n_epochs=4, load_best=False, eval_per_epoch=1, eval_train=False)
            y_n_pseudo = evaluate_fn(self.model_m2, self.us_eval_dl, " randomtexthereforgreppurposesBBBBBBBB  model m2 us eval bkt {}".format(i), self.device, return_lbls=True)
            scrs_bkt[i] = self._get_scrs(self.model_m2, bkts_vl, " model m2")
            print("Evaluating the entire eval dataset .... ")
            scrs.append(("st-bkt-{}".format(i), f1(np.array(y_n), np.array(y_n_pseudo), True)))
            prevmodel = self.model_m2
            print("randomtexthereforgreppurposesBBBBBBBB scrs bkt \n" + "\n".join(str(scr_bkt) +  "   " + str(scrs_bkt[scr_bkt]) for scr_bkt in scrs_bkt))

        print("randomtexthereforgreppurposesAAAAAAA")
        print("\n".join(str(scr) for scr in scrs))
        print("randomtexthereforgreppurposesBBBBBBBB scrs bkt \n" + "\n".join(str(scr_bkt) +  "   " + str(scrs_bkt[scr_bkt]) for scr_bkt in scrs_bkt))
        # print("\n".join(str(scr_bkt) +  "   " + str(scrs_bkt[scr_bkt]) for scr_bkt in scrs_bkt))
        # print(scrs_bkt)
        print("Printing ood dictionary : {}".format(ood_res))
        print("\n\n\n\n")
        
        with open("./logs2/dx/{}".format(self.datasetname), "w") as f:
            for x_i, y_bkt_i, y_bl_i, y_i in zip(x_n, y_n_pseudo, y_n_pred_bl, y_n):
                if y_bl_i != y_bkt_i:
                    f.write("{} {} {} {} \n".format(y_i, y_bkt_i, y_bl_i, x_i))

        del self.model_m2
        torch.cuda.empty_cache()
        return (scrs, scrs_bkt)

# {python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py; } > basic_ds_run

