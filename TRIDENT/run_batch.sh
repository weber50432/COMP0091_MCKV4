python run_batch_of_slides.py \
--gpu 0 \
--task all \
--wsi_dir ../TCGA-WSI/UCEC \
--job_dir ../trident_processed/UCEC \
--patch_encoder gigapath \
--mag 20 \
--patch_size 256
# --wsi_dir ../TCGA-patches/BRCA-non-TSX/patches \