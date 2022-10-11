from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import copy
import random

def get_predictions(lang):
    with open("./Results/Sentiment_EN_HI/{}_val_predictions2.txt".format(lang), "r") as f:
        lines = f.readlines()
    y_pred = []
    y_true = []
    y_prob = []
    count_eng = []
    count_hin = []
    for lin in lines:
        a, b, prob, c, d = lin.strip().split() 
        y_pred.append(a)
        y_true.append(b)
        y_prob.append(float(prob))
        count_eng.append(int(c))
        count_hin.append(int(d))
    return y_pred, y_true, y_prob, count_eng, count_hin

y_pred_eng, y_true_eng, y_prob_eng, count_eng, count_hin = get_predictions("eng")
y_pred_hin, y_true_hin, y_prob_hin, count_eng, count_hin = get_predictions("hin")
y_pred_swi, y_true_swi, y_prob_swi, count_eng, count_hin = get_predictions("switched")

clf_rep_eng = classification_report(y_true_eng, y_pred_eng)
clf_rep_hin = classification_report(y_true_hin, y_pred_hin)
clf_rep_swi = classification_report(y_true_swi, y_pred_swi)

reports = [
    (clf_rep_eng, "clf_rep_eng"),
    (clf_rep_hin, "clf_rep_hin"),
    (clf_rep_swi, "clf_rep_swi"),
]

for rep, name in reports:
    print("Classification report name : {}".format(name))
    print(rep)
    print("\n\n\n")

check_improve_perf = {
    ("eng", "hin") : [y_pred_eng, y_pred_hin, y_true_hin],
    ("hin", "eng") : [y_pred_hin, y_pred_eng, y_true_eng],
    ("eng", "swi") : [y_pred_eng, y_pred_swi, y_true_swi],
    ("hin", "swi") : [y_pred_hin, y_pred_swi, y_true_swi],
}

for k in check_improve_perf:
    print("Checking if lang a ---{}--- predictions help in improving lang B ---{}--- results"
          ".".format(k[0], k[1]))
    y_pred_a, y_pred_b, y_true_b = check_improve_perf[k]
    y_pred_a_copy = copy.deepcopy(y_pred_a)
    all_res = []                     
    
    def compute_new_f1(y_pred_a, y_pred_b, y_true_b, print_rep=False):
        y_pred_mixed_ = []                                                          
        for a, b, c in zip(y_pred_a, y_pred_b, y_true_b):                    
            if a == c or b == c:                                                    
                y_pred_mixed_.append(c)                                             
            else:                                                                   
                y_pred_mixed_.append(b)                                             
        clf_rpt_mixed_n = classification_report(y_true_b, y_pred_mixed_);
        if print_rep:
            print(clf_rpt_mixed_n)           
        f1 = float(clf_rpt_mixed_n.split("\n")[-2].split()[-2])
        return f1                  

    new_modified_f1 = compute_new_f1(y_pred_a, y_pred_b, y_true_b, True)                      

    for i in range(100):                                                            
        random.shuffle(y_pred_a_copy)                                                   
        f1 = compute_new_f1(y_pred_a_copy, y_pred_b, y_true_b)                        
        all_res.append(f1)

    print(all_res)
    print(sum(all_res) * 1.0 / len(all_res))
    print(new_modified_f1)
    print(sum([1 for f1_ in all_res if f1_ > new_modified_f1]))


print("\n\n\nSanity check with heuristic 2")
check_improve_perf = {
    ("eng", "hin") : [y_pred_eng, y_prob_eng, count_eng, y_pred_hin, y_prob_hin, y_true_hin],
    ("hin", "eng") : [y_pred_hin, y_prob_hin, count_hin, y_pred_eng, y_prob_eng, y_true_eng],
    ("eng", "swi") : [y_pred_eng, y_prob_eng, count_eng, y_pred_swi, y_prob_swi, y_true_swi],
    ("hin", "swi") : [y_pred_hin, y_prob_hin, count_hin, y_pred_swi, y_prob_swi, y_true_swi],
}

for k in check_improve_perf:
    print("Checking if lang a ---{}--- predictions help in improving lang B ---{}--- results"
          ".".format(k[0], k[1]))
    y_pred_a, y_prob_a, count_a, y_pred_b, y_prob_b, y_true_b = check_improve_perf[k]
    y_pred_a_copy = copy.deepcopy(y_pred_a)
    y_prob_a_copy = copy.deepcopy(y_prob_a)
    count_a_copy = copy.deepcopy(count_a)
    all_res = []                     
    all_chn = []                     
    
    def compute_new_f1(y_pred_a, y_prob_a, count_a, y_pred_b, y_prob_b, y_true_b, print_rep=False):
        y_pred_mixed_ = []
        changed = 0                                                          
        for a, b, c, d, e in zip(y_pred_a, y_prob_a, count_a, y_pred_b, y_prob_b):                    
            if c > 3:
                if b <= e:                                                    
                    y_pred_mixed_.append(d)
                else:
                    y_pred_mixed_.append(a)
                    changed += 1
            else:                                                                   
                y_pred_mixed_.append(d)                                             
        clf_rpt_mixed_n = classification_report(y_true_b, y_pred_mixed_);
        if print_rep:
            print(clf_rpt_mixed_n)           
        f1 = float(clf_rpt_mixed_n.split("\n")[-2].split()[-2])
        return f1, changed                  

    new_modified_f1, changed = compute_new_f1(y_pred_a, y_prob_a, count_a, y_pred_b, y_prob_b, y_true_b, True)                      
    actual_f1, changed = compute_new_f1(y_pred_b, y_prob_a, count_a, y_pred_b, y_prob_b, y_true_b)                      

    for i in range(100):
        c = list(zip(y_pred_a_copy, y_prob_a_copy, count_a_copy))
        random.shuffle(c)
        y_pred_a_copy, y_prob_a_copy, count_a_copy = zip(*c)                                   
        # random.shuffle(y_pred_a_copy)                                                   
        f1, chn = compute_new_f1(y_pred_a_copy, y_prob_a, count_a, y_pred_b, y_prob_b, y_true_b)                        
        all_res.append(f1)
        all_chn.append(chn)

    print("All f1 generated by random trials: ", all_res)
    print("Avg of all f1 generated by randolm trials : ", sum(all_res) * 1.0 / len(all_res))
    print("F1 obtained with the modified heuristic (when lang A model helps lang B): ", new_modified_f1, changed)
    print("F1 obtained by direct model predictions (only using the lang B model) : ", actual_f1)
    print("No of trials where random gave better results then modified heuristic", sum([1 for f1_ in all_res if f1_ > new_modified_f1]))
    print("\n\n\n")
