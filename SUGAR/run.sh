DATASET=TCGA-LUAD
NUM_INFO=1
LR=0.001
MAX_POOL=2
MOMENTUM=0.9
NUM_EPOCH=500
BATCH_SIZE=32
SG_ENCODER=GCN
MI_LOSS=0.1
START_K=0.1

python train.py \
     --dataset $DATASET\
     --num_info $NUM_INFO\
     --lr $LR\
     --max_pool $MAX_POOL\
     --momentum $MOMENTUM\
     --num_epoch $NUM_EPOCH\
     --batch_size $BATCH_SIZE\
     --sg_encoder $SG_ENCODER\
     --MI_loss $MI_LOSS\
     --start_k $START_K\
     > ./log/${DATASET}_${SG_ENCODER}.log 2>&1 &