import pickle
import argparse

from transformers import AutoModelForSequenceClassification
from load_data import read_file
from models.ds_model import DsTrainer
from utils import set_seed

def get_filtered_texts(fname, return_leaky_texts=False):
    with open(fname, "rb") as f:
        print("Pretrianing with file {} ....".format(fname))
        allposstexts_sorted = pickle.load(f)

    filtered_texts = []
    leaky_texts    = []

    import editdistance

    tot = 0

    for sim, text_cs, text_en, label in allposstexts_sorted:
        text_cs_splitted = text_cs.split()
        text_en_splitted = text_en.split()

        newtext_cs = ""
        for wrd in text_cs_splitted:
            if wrd in text_en_splitted:
                newtext_cs += wrd + " "

        newtext_en = ""
        for wrd in text_en_splitted:
            if wrd in text_cs_splitted:
                newtext_en += wrd + " "

        eps = 0.50
        if (len(newtext_cs.split()) >= eps * len(text_cs_splitted) and 
            len(newtext_en.split()) >= eps * len(text_en_splitted)) or sim < 0.75:
            dist = editdistance.eval(newtext_cs, newtext_en)
            if dist < 0.5 * len(newtext_cs) or dist < 0.5 * len(newtext_en) or sim < 0.75:
                tot += 1
                leaky_texts.append(text_en)
                continue
        
        filtered_texts.append([sim, text_cs, text_en, label])
    print("Found {}/{} leaky texts ... ".format(tot, len(allposstexts_sorted)))
    if return_leaky_texts:
        return set(leaky_texts)
    return filtered_texts

def read_mined_texts(imbalancefix, datasetname, ablation=False):
    all_fname = "./exdata/all_postprocessing.pkl"
    mined_fname = "./exdata/mined_same_postprocessing.pkl"
    texts = get_filtered_texts(all_fname)
    leaky_texts    = get_filtered_texts(mined_fname, return_leaky_texts=True)
    filtered_texts = []

    if datasetname == "sail":
        for t in texts:
            if t[2] not in leaky_texts:
                filtered_texts.append(t)
    else:
        filtered_texts = texts

    print("Length of original text : {}".format(len(texts)))
    print("Length of filtered text : {}".format(len(filtered_texts)))

    # from random import shuffle
    # shuffle(filtered_texts)

    for txt in filtered_texts[:10]:
        print(txt)
    print(len(filtered_texts))
    from collections import Counter
    clss_cnt = dict(Counter([a[3] for a in filtered_texts]))
    # print("Class wise statistics : ", clss_cnt)
    max_entries = min(clss_cnt.values())
    # print("Max entries : ", max_entries)
    entry_cnt = {k:0 for k in clss_cnt}
    ds_x, ds_y = [], []

    if imbalancefix == "none":
        for ft in filtered_texts:
            if entry_cnt[ft[3]] < max_entries:
                ds_x.append(ft[2]); ds_y.append(ft[3])
                entry_cnt[ft[3]] += 1
    elif imbalancefix == "upsample":
        ds_x_cwise, ds_y_cwise = {}, {}
        for k in clss_cnt:
            ds_y_cwise[k] = []
            ds_x_cwise[k] = []
        for ft in filtered_texts:
            ds_x_cwise[ft[3]].append(ft[2])
            ds_y_cwise[ft[3]].append(ft[3])
        
        if clss_cnt["positive"] < clss_cnt["negative"]:
            min_clss = "positive"; max_clss = "negative"
        else:
            min_clss = "negative"; max_clss = "positive"

        j=0
        while len(ds_y_cwise[min_clss]) < len(ds_y_cwise[max_clss]):
            ds_x_cwise[min_clss].append(ds_x_cwise[min_clss][j])
            ds_y_cwise[min_clss].append(ds_y_cwise[min_clss][j])
            j += 1

        for k in ds_x_cwise:
            ds_x.extend(ds_x_cwise[k]); ds_y.extend(ds_y_cwise[k])
    elif imbalancefix == "weightedloss":
        for ft in filtered_texts:
            ds_x.append(ft[2]); ds_y.append(ft[3])
    else:
        raise NotImplementedError("imbalance fix not found")

    print("Filtered class wise statistics : ")
    print(Counter(ds_y))

    # print("\n\nSamples of ds_x and ds_y")
    # print(ds_x[:10])
    # print(ds_x[-10:])
    # print(ds_y[:10])
    # print(ds_y[-10:])
    # return ds_x[:100], ds_y[:100]
    return ds_x, ds_y

def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibfix", choices=["none", "upsample", "weightedloss"], 
                        help="how to fix imbalance in the dataset", default="none")
    parser.add_argument("--seed", default=1,  help="choose the seed")
    parser.add_argument("--seed_new", default=1,  help="choose the seed")
    parser.add_argument("--bkts", default=4,  help="choose the seed")
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--zsl_ds_us_data_merged_multiple_m_half_data', action='store_true')
    parser.add_argument('--zsl_ds_us_data_merged_multiple_m_half_data_many_runs', action='store_true')
    parser.add_argument("--zz_data", choices=["sail", "enes", "taen"],  default="sail")
    args = parser.parse_args()
    return args

args = get_parsed_args()
print(args)

ds_x, ds_y = read_mined_texts(args.ibfix, args.zz_data, ablation=args.ablation)
x, y = read_file("./" + args.zz_data, True, False)

set_seed(args.seed)

ds_trainer =  DsTrainer((ds_x, ds_y), 
                        (x, y), 
                        datasetname=args.zz_data,
                        seed=args.seed)
ds_trainer.pretrain(n_epochs=5)
if args.zsl_ds_us_data_merged_multiple_m_half_data_many_runs:
    print("Running multiple runs : ZSL with both ds and pl data, "
          "using half the instances and hard labels run 0")
    resout = []
    ds_trainer.pretrain(n_epochs=5)
    resout.append(ds_trainer.selftrain_bucketing_different_m_merged_data_half())
    print("Running multiple runs : ZSL with both ds and pl data, "
          "using half the instances and hard labels run 1")
    set_seed(1)
    ds_trainer.pretrain(n_epochs=5)
    resout.append(ds_trainer.selftrain_bucketing_different_m_merged_data_half())
    print("Running multiple runs : ZSL with both ds and pl data, "
          "using half the instances and hard labels run 2")
    set_seed(2)
    ds_trainer.pretrain(n_epochs=5)
    resout.append(ds_trainer.selftrain_bucketing_different_m_merged_data_half())
    print("Running multiple runs : ZSL with both ds and pl data, "
          "using half the instances and hard labels run 3")
    set_seed(3)
    ds_trainer.pretrain(n_epochs=5)
    resout.append(ds_trainer.selftrain_bucketing_different_m_merged_data_half())
    print("Running multiple runs : ZSL with both ds and pl data, "
          "using half the instances and hard labels run 4")
    set_seed(4)
    ds_trainer.pretrain(n_epochs=5)
    resout.append(ds_trainer.selftrain_bucketing_different_m_merged_data_half())
    print("randomTexthereForGrepPurposesAAAAAA")
    print("Final results : {}".format(resout))
    for (scrs, scrs_bkt) in resout:
        print("\n".join(str(scr) for scr in scrs))
        print("randomtexthereforgrepPurposesBBBBB scrs bkt \n" + "\n".join(str(scr_bkt) +  "   " + str(scrs_bkt[scr_bkt]) for scr_bkt in scrs_bkt))
elif args.zsl_ds_us_data_merged_multiple_m_half_data:
    print("Running ZSL with both ds and pl data, using half the instances and hard labels")
    ds_trainer.selftrain_bucketing_different_m_merged_data_half()
elif args.supervised:
    print("Running model to get supervised upperbound ")
    ds_trainer.supervised_upperbound()
else:
    raise NotImplementedError(" self-training method has not been found ")

print("Self-training 1 completed ..... randomtexthereforgrepPurposesBBBBB \n\n\n\n")

# { python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py; python main_ds.py;} > ./basic_ds_run

