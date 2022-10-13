# Progressive training for code-switched sentiment analysis

- [Model](#model)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Commands](#commands)
	- [Requirements](#requirements)
- [Citation](#citation)

## Model

Most of the experiments have been carried out using ```bert-base-multilingual-cased``` as the backbone model. 

The framework is present in the file ``` models/ds_model.py ```.

<!-- ## Training -->

### Required inputs

External english data for pretraining should be present in the ```data/english_data``` file.

For code-switched datasets, create a folder with the dataset name and create ```train.txt``` and ```validation.txt```.

For example, for the ```sail``` dataset used in ```GLUECoS```, files should be ```data/sail/train.txt``` and ```data/sail/validation.txt```.

Each row should contain the text and the label (i.e. positive or negative). The words in Hindi/Tamil should be transliterated to devanagari script.

You can obtain the Hindi-English (sail) or Spanish-English (enes) dataset from [here](https://github.com/microsoft/GLUECoS) and put it in the data folder.  Tamil-English (taen) dataset can be downloaded from [here](https://dravidian-codemix.github.io/2020/datasets.html) . 

### Commands

The ```main_ds.py``` can take following arguments: 
- ```arg_data``` to denote the dataset, currently takes ```sail``` , ```enes``` or ```taen``` as input.
- ```external_data_imbalance_fix``` to deal with imbalance in the source dataset used for pretraining.
- ```seed``` to fix the seed scross experiments
- ```zsl_ds_us_data_merged_multiple_m_half_data``` or ```zsl_ds_us_data_merged_multiple_m_half_data_many_runs``` or ```supervised``` to do single run or multiple runs or supervised run.

Example commands to run:

```
python main_ds.py --external_data_imbalance_fix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --arg_data sail > logs/sail_half_data_hrd_lbl_merged_bkts_ds_us_run22 &

python main_ds.py --external_data_imbalance_fix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --arg_data taen > logs/taen_half_data_hrd_lbl_merged_bkts_ds_us_run22 &

python main_ds.py --external_data_imbalance_fix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --arg_data enes > logs/enes_half_data_hrd_lbl_merged_bkts_ds_us_run22 &
```

### Requirements

This project is based on ```python==3.6.10```. The dependencies are as follow:
```
torch==1.9.1
argparse
transformers==3.5.1
nltk==3.5
sklearn
```




