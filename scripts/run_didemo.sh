CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
DATA_PATH=/mnt/workspace/workgroup/multimodal/datasets/DiDeMo/anns

python -m torch.distributed.launch --nproc_per_node=8 main_my.py --do_train --eval_in_train --num_thread_reader=8 --seed 0  \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/train_data_mp4.json \
--val_csv ${DATA_PATH}/test_data_mp4.json \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_3fps_mp4 \
--output_dir ckpts/didemo --datatype didemo \
--cross_num_hidden_layers 4 \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 8 \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32 --max_patch 12 --max_word_pro 28 
