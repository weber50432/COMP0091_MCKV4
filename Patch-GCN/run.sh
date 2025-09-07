job="patchgcn"
cancer="blca"
mode="graph"
feature_dir="../TCGA-CLAM-graph/BLCA"

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--which_splits 5foldcv \
--split_dir tcga_${cancer} \
--mode $mode \
--model_type $job \
--data_root_dir $feature_dir \
> ./log/${cancer}_${job}.log 2>&1 &