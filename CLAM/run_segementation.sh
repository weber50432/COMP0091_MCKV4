python create_patches_fp.py \
    --source ../TCGA-WSI/BRCA-non-TSX\
    --save_dir ../TCGA-patches/BRCA-non-TSX\
    --patch_size 256 \
    --seg --patch --stitch \
    --preset tcga.csv