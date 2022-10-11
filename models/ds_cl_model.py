from numpy.lib.function_base import append
from utils import textDataset, train_fn, evaluate_fn, f1
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertPreTrainedModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np
from collections import Counter

from transformers import BertForSequenceClassification


class ClBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.projector = nn.Linear(config.hidden_size, 128)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, None)

    def forward_ct(
        self,
        batch_size,
        neg_samples=16,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        temperature=0.5, 
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        pooled_output = self.projector(pooled_output)
        norm_pooled_output = torch.div(pooled_output, torch.linalg.norm(pooled_output, dim=-1)[:, None])
        sim_mat = torch.matmul(norm_pooled_output, norm_pooled_output.T)
        posv    = torch.diagonal(     sim_mat[0:batch_size, batch_size:2*batch_size]  )
        neg_sim_mat  = sim_mat[:batch_size, 2*batch_size:]
        indx = torch.multinomial(torch.ones(batch_size, batch_size), neg_samples)
        negv = torch.gather(neg_sim_mat, 1, indx)
        allv  = torch.cat([posv[:, None], negv], dim=1)
        allv = allv / torch.Tensor(temperature).to(input_ids.device)
        labels = torch.zeros(4*batch_size)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(allv.view(-1, neg_samples), labels.view(-1))

        return (loss, None)

class DsClTrainer(object):
    def __init__(self, ds_pretrain_dataset, cl_pretrain_dataset, dataset, model_name="bert-base-multilingual-cased"):
        print("\n\nInitialising the ds trainer object .... ")

        print("Creating tokenizers .... ")
        self.model_name = model_name
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._create_dataloaders(ds_pretrain_dataset, dataset)
        self.model = ClBertForSequenceClassification.from_pretrained(model_name, num_classes=2)

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

    def pretrain(self):
        print("\n\nPretraining on the mined data .... ")
        self.model_m1 = train_fn(self.model_m1, self.ds_train_dl, 
                                 self.ds_eval_dl, self.device, n_epochs=2)
        print("\n\nEvaluation of the ds trained model on ds dataset .... ")
        evaluate_fn(self.model_m1, self.ds_eval_dl, "ds eval", self.device)
        print("\n\nEvaluation of the ds trained model on us dataset .... ")
        self.y_pseudo = evaluate_fn(self.model_m1, self.us_eval_dl, "us eval", 
                                    self.device, return_lbls=True)
        print("\n\nPretrining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.y), np.array(self.y_pseudo), True)
        print("Scores obtained after pretraining f1 micro / macro / weighted : {:.3f} / {:.3f} / {:.3f}".format(
            f1_macro, f1_micro, f1_weighted))

    def predict_m1(self, x, y=None):
        print("Predicting using the first model m1 .... ")
        pass

    def predict_m2(self, x):
        print("Predicting using the first model m2 .... ")
        pass

    def selftrain(self):
        raise NotImplementedError("current implementation creates a new model, fix it to include the past model")
        self.us_pseudo_train_dl = DataLoader(textDataset(self.x, self.y_pseudo),
                                             shuffle=True, batch_size=64)
        print("Self training with pseudo labels ....")
        self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model_m2.to(self.device)
        print("\n\nCreated model m2 ....")
        train_fn(self.model_m2, self.us_pseudo_train_dl, self.us_eval_dl, self.device, n_epochs=1, load_best=False)
        print("\n\nFinal results of self trained  on us dataset .... ")

        self.y_pseudo = evaluate_fn(self.model_m2, self.us_eval_dl, "us eval", 
                                    self.device, return_lbls=True)
        print("\n\nSelftraining complete .... ")
        f1_macro, f1_micro, f1_weighted = f1(np.array(self.y), np.array(self.y_pseudo), True)
        print("Scores obtained after selftraing f1 macro / micro / weighted : {} / {} / {}".format(
            f1_macro, f1_micro, f1_weighted))

    def selftrain_bucketing(self, bkt_cnt=4):
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
        
        aa = zip(hin_eng_count(self._old_x)[:, 0], self._old_x, self.y_pseudo, self.y)
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
        scrs_bkt[-1] = {}
        scrs.append(f1(np.array(y_n), np.array(y_n_pseudo), True))
        for j, (bkt, bkt_y) in enumerate(bkts_vl):
            print("      Evaluating model for bkt {} .... ".format(j))
            bkt_y_pred = evaluate_fn(self.model_m1, bkt, "us eval",  self.device, return_lbls=True)
            scrs_bkt[-1][j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)
        # delete previous model
        del self.model_m1
        torch.cuda.empty_cache()

        for i in range(bkt_cnt):
            scrs_bkt[i] = {}
            print("\n\n" + "*********" * 10)
            print("\n\nshingeki Self training w/ bucket {} .... ".format(i))
            x_curr_trn = x_n[:(i+1)*bkt_sz]; y_curr_trn = y_n_pseudo[:(i+1)*bkt_sz]
            x_curr_trn = self._preprocess_dataset(x_curr_trn)
            print("Bucket size : {}".format(len(y_curr_trn)))
            self.us_pseudo_train_dl = DataLoader(textDataset(x_curr_trn, y_curr_trn),
                                                 shuffle=True, batch_size=64)
            self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model_m2.to(self.device)
            train_fn(self.model_m2, self.us_pseudo_train_dl, self.us_eval_dl, self.device, n_epochs=1, load_best=False)
            y_n_pseudo = evaluate_fn(self.model_m2, self.us_eval_dl, "us eval", 
                                        self.device, return_lbls=True)
            for j, (bkt, bkt_y) in enumerate(bkts_vl):
                print("      Evaluating model for bkt {} .... ".format(j))
                bkt_y_pred = evaluate_fn(self.model_m2, bkt, "us eval",  self.device, return_lbls=True)
                scrs_bkt[i][j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)
            print("Evaluating the entire eval dataset .... ")
            scrs.append(f1(np.array(y_n), np.array(y_n_pseudo), True))



        print("Final evaluation .... ")        
        evaluate_fn(self.model_m2, self.us_eval_dl, "us eval", 
                                        self.device, return_lbls=True)
        
        del self.model_m2
        torch.cuda.empty_cache()

        for scr in scrs:
            print(scr)

        print(scrs_bkt)

    def selftrain_bucketing_same_m(self, bkt_cnt=4):
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
        
        aa = zip(hin_eng_count(self._old_x)[:, 0], self._old_x, self.y_pseudo, self.y)
        aa_sorted_e2h = sorted(aa, reverse=True)
        x_n, y_n_pseudo, y_n = [a[1] for a in aa_sorted_e2h], [a[2] for a in aa_sorted_e2h], [a[3] for a in aa_sorted_e2h]

        # create a new data loader with the unsupervised data, this has to be in
        # the order of new x, need this to get the pseudo labels.
        self.us_eval_dl  = DataLoader(textDataset(self._preprocess_dataset(x_n) , y_n), shuffle=False, batch_size=64)   

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
        
        scrs_bkt[-2] = {}
        scrs.append(f1(np.array(y_n), np.array(y_n_pseudo), True))
        for j, (bkt, bkt_y) in enumerate(bkts_vl):
            print("      Evaluating model for bkt {} .... ".format(j))
            bkt_y_pred = evaluate_fn(self.model_m1, bkt, "us eval",  self.device, return_lbls=True)
            scrs_bkt[-2][j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)

        # get the bl results.
        PATH = "logs/tempmodel"
        torch.save(self.model_m1.state_dict(), PATH)
        # model_bl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # model_bl.load_state_dict(torch.load(PATH))
        # model_bl.to(self.device)
        # self.us_pseudo_train_dl = DataLoader(textDataset(self._preprocess_dataset(x_n), y_n_pseudo),
        #                                         shuffle=True, batch_size=64)
        # train_fn(model_bl, self.us_pseudo_train_dl, self.us_eval_dl, self.device, n_epochs=2, load_best=False)
        # y_n_pred_bl = evaluate_fn(model_bl, self.us_eval_dl, "us eval", self.device, return_lbls=True)
        # scrs_bkt[-1] = {}
        # scrs.append(f1(np.array(y_n), np.array(y_n_pred_bl), True))
        # for j, (bkt, bkt_y) in enumerate(bkts_vl):
        #     print("      Evaluating model for bkt {} .... ".format(j))
        #     bkt_y_pred = evaluate_fn(model_bl, bkt, "us eval",  self.device, return_lbls=True)
        #     scrs_bkt[-1][j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)

        final_y_pred = []
        self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model_m2.load_state_dict(torch.load(PATH))
        self.model_m2.to(self.device)
        for i in range(bkt_cnt):
            scrs_bkt[i] = {}
            print("\n\n" + "*********" * 10)
            print("\n\nshingeki Self training w/ bucket {} .... ".format(i))
            x_curr_trn = x_n[:(i+1)*bkt_sz]; y_curr_trn = y_n_pseudo[:(i+1)*bkt_sz]
            x_curr_trn = self._preprocess_dataset(x_curr_trn)
            print("Bucket size : {}".format(len(y_curr_trn)))
            self.us_pseudo_train_dl = DataLoader(textDataset(x_curr_trn, y_curr_trn),
                                                 shuffle=True, batch_size=64)
            # self.model_m2 = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            # self.model_m2 = self.model_m1
            # self.model_m2.to(self.device)
            train_fn(self.model_m2, self.us_pseudo_train_dl, self.us_eval_dl, self.device, n_epochs=2, load_best=False)
            y_n_pseudo = evaluate_fn(self.model_m2, self.us_eval_dl, "us eval", 
                                        self.device, return_lbls=True)
            for j, (bkt, bkt_y) in enumerate(bkts_vl):
                print("      Evaluating model for bkt {} .... ".format(j))
                bkt_y_pred = evaluate_fn(self.model_m2, bkt, "us eval",  self.device, return_lbls=True)
                scrs_bkt[i][j] = f1(np.array(bkt_y), np.array(bkt_y_pred), True)
                if i == j:
                    final_y_pred.extend(bkt_y_pred)
            print("Evaluating the entire eval dataset .... ")
            scrs.append(f1(np.array(y_n), np.array(y_n_pseudo), True))

        scrs.append(f1(np.array(y_n), np.array(final_y_pred), True))

        print("Final evaluation .... ")        
        evaluate_fn(self.model_m2, self.us_eval_dl, "us eval", 
                                        self.device, return_lbls=True)
        
        del self.model_m2
        torch.cuda.empty_cache()

        for scr in scrs:
            print(scr)

        for k in scrs_bkt:
            print(scrs_bkt[k])

        print(scrs_bkt)

    def selftrain_diff_loop(self):

        def get_ids():
            print("\nGetting top 50 ids ..... ")
            y_pseudo_pr = evaluate_fn(self.model_m1, self.us_eval_dl, "us eval", self.device, return_probs=True)
            y_pred      = np.argmax(np.array(y_pseudo_pr), axis=-1)
            y_prob      = np.max(np.array(y_pseudo_pr), axis=-1)
            
            aa = zip(y_prob, y_pred, self._old_x, self.y, [i for i in range(len(self._old_x))])
            aa_sorted_e2h = sorted(aa, reverse=True)
            x_n, y_n_pseudo, y_n, ids = ([a[2] for a in aa_sorted_e2h], 
                                         [a[1] for a in aa_sorted_e2h], 
                                         [a[3] for a in aa_sorted_e2h],  
                                         [a[4] for a in aa_sorted_e2h])

            newlen = len(x_n) // 2
            print(Counter(y_n_pseudo))
            x_n, y_n_pseudo, y_n, ids = x_n[:newlen], y_n_pseudo[:newlen], y_n[:newlen], ids[:newlen]
            print(Counter(y_n))
            print(Counter(y_n_pseudo))
            return ids

        previds = get_ids()

        scrs = []
        diff = 1.0
        cntr = 0
        while diff > 0.1:

            print("Iteration / current diff  : {} / {}".format(cntr, diff))
            cntr += 1
            # create the train datalodaer with the correct ids
            y_pseudo_cls = evaluate_fn(self.model_m1, self.us_eval_dl, "us eval", self.device, return_lbls=True)
            x_curr_trn, y_curr_trn = [self._old_x[t] for t in previds], [y_pseudo_cls[t] for t in previds]
            x_curr_trn = self._preprocess_dataset(x_curr_trn)
            print("Bucket size : {}".format(len(y_curr_trn)))
            self.us_pseudo_train_dl = DataLoader(textDataset(x_curr_trn, y_curr_trn), shuffle=True, batch_size=64)
            train_fn(self.model_m1, self.us_pseudo_train_dl, 
                     self.us_eval_dl, self.device, n_epochs=1, load_best=False, eval_per_epoch=1)
            print("Evaluating the entire eval dataset .... ")
            y_n_pseudo = evaluate_fn(self.model_m1, self.us_eval_dl, "us eval", 
                                        self.device, return_lbls=True)
            scrs.append(f1(np.array(self.y), np.array(y_n_pseudo), True))
            currids = get_ids()

            diff = (len(currids) - len(set(currids).intersection(set(previds)))) / len(currids)
            previds = currids

# {python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py; } > basic_ds_run