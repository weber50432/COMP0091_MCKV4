python run_single_slide.py \
--gpu 0 \
--slide_path ../TCGA-WSI/LUAD/TCGA-05-5420-01Z-00-DX1.8C253A99-44FD-48B6-AF31-D808CCB7DB1E.svs \
--custom_mpp_keys aperio.MPP \
--job_dir ./trident_processed/LUAD/single-slide \
--patch_encoder gigapath \
--mag 20 \
--patch_size 256