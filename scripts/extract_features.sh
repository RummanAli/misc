#!/bin/bash
git clone https://github.com/OpenGVLab/VideoMAEv2.git
cd VideoMAEv2
mkdir weights
cd weights
wget https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth
cd ..
python extract_tad_feature.py \
    --data_set THUMOS14 \
    --data_path YOUR_PATH/ \
    --save_path YOUR_PATH/th14_vit_g_16_4 \
    --model vit_giant_patch14_224 \
    --ckpt_path ./weights/vit_g_hyrbid_pt_1200e_k710_ft.pth