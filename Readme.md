# to run commands use: 

CUDA_VISIBLE_DEVICES=12 python -u main_ds.py --ibfix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --zz_data sail > logs2/sail_half_data_hrd_lbl_merged_bkts_ds_us_run22 &
CUDA_VISIBLE_DEVICES=13 python -u main_ds.py --ibfix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --zz_data taen > logs2/taen_half_data_hrd_lbl_merged_bkts_ds_us_run22 &
CUDA_VISIBLE_DEVICES=15 python -u main_ds.py --ibfix upsample  --seed 22 --zsl_ds_us_data_merged_multiple_m_half_data_many_runs --zz_data enes > logs2/enes_half_data_hrd_lbl_merged_bkts_ds_us_run22 &


