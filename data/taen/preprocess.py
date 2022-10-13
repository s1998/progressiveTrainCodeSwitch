import pandas as pd

train_df      = pd.read_csv("./tamil_train.tsv", sep="\t")
validation_df = pd.read_csv("./tamil_dev.tsv",   sep="\t")

print(set(train_df.category.tolist())) 
print(set(validation_df.category.tolist()))

from collections import Counter
print(Counter(train_df.category.tolist())) 
print(Counter(validation_df.category.tolist())) 


all_sents = []
for _, row in train_df.iterrows():
    if not (row["category"].strip() in ["Positive", "Negative"]):
        print(row)

def get_transliterated_sents(currdf):
    from nltk.corpus import brown
    from ai4bharat.transliteration import XlitEngine

    eng_wrds_set = set(brown.words())
    translit = XlitEngine("ta")
    transliterated_sents = []

    for ind, row in currdf.iterrows():
        if not (row["category"].strip() in ["Positive", "Negative"]):
            continue
        wrds_sent = row["text"].split()
        new_sent = ""
        for wrd in wrds_sent:
            if wrd in eng_wrds_set:
                new_sent += wrd + " "
            else:
                ta_wrd = translit.translit_word(wrd, topk=1, beam_width=10)
                new_sent += ta_wrd["ta"][0] + " "
        transliterated_sents.append([new_sent, row["category"].strip().lower()])
        print("Completed for {} / {} sentences .... ".format(
            ind, train_df.shape[0]))     
    return pd.DataFrame(transliterated_sents)

train_df_new      = get_transliterated_sents(train_df)
validation_df_new = get_transliterated_sents(validation_df)

print(train_df_new.head())

train_df_new.to_csv(index=False, header=False, sep="\t")
validation_df_new.to_csv(index=False, header=False, sep="\t")

train_df_new.to_csv(     "train.txt", index=False, 
                         header=False, sep="\t")
validation_df_new.to_csv("validation.txt", index=False, 
                         header=False, sep="\t")



from ai4bharat.transliteration import XlitEngine
e = XlitEngine("hi")
out = e.translit_word("computer", topk=1, beam_width=10)
print(out)
print(out['hi'][0])


# In[29]:


e = XlitEngine("ta")
out = e.translit_sentence("vanakkam ulagam !", beam_width=10)






