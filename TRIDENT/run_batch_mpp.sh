python run_batch_of_slides.py \
--gpu 0 \
--task all \
--custom_list_of_wsis ../TCGA-WSI/BRCA/custom_wsis.csv \
--wsi_dir ../TCGA-WSI/BRCA \
--job_dir ../trident_processed/BRCA \
--patch_encoder gigapath \
--mag 20 \
--patch_size 256
# --wsi_dir ../TCGA-patches/BRCA-non-TSX/patches \