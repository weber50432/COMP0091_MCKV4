CUDA_VISIBLE_DEVICES=0 \
python extract_features_fp.py \
    --data_h5_dir ../TCGA-patches/BRCA-non-TSX \
    --data_slide_dir ../TCGA-WSI/BRCA-non-TSX \
    --csv_path ../TCGA-patches/BRCA-non-TSX/process_list_autogen.csv \
    --feat_dir ../TCGA-graph/BRCA-non-TSX \
    --batch_size 512 \
    --slide_ext .svs