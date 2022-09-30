
# ResNet18 protocol 1
FATHER_DIR=output/bc/ResNet18_Adam/HiFi+CelebA+SIW/; CUDA_VISIBLE_DEVICES=3 python train.py --trainer bc_ewc MODEL.ARCH resnet18 TRAIN.PROTOCOL 1 OUTPUT_DIR output/bc_ewc/ResNet18_Adam/protocol1_2debug/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 200 MODEL.ARCH resnet18 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 50 TRAIN.EPOCHS 500 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 128 TRAIN.EWC_ONLINE False
FATHER_DIR=output/bc/ResNet18_Adam/HiFi+CelebA+SIW/; CUDA_VISIBLE_DEVICES=3 python train.py --trainer bc_ewc MODEL.ARCH resnet18 TRAIN.PROTOCOL 1 OUTPUT_DIR output/bc_ewc/ResNet18_Adam/protocol1_online/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 200 MODEL.ARCH resnet18 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 500 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 64 TRAIN.EWC_ONLINE True

# protocol 1 offline
FATHER_DIR=output/bc/ResNet18_Adam/HiFi+CelebA+SIW/; CUDA_VISIBLE_DEVICES=0 python train.py --trainer bc_ewc MODEL.ARCH resnet18 TRAIN.PROTOCOL 1 OUTPUT_DIR output/bc_ewc/ResNet18_Adam/protocol1/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 200 MODEL.ARCH resnet18 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 500 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 128 TRAIN.EWC_ONLINE True



# ResNet18 protocol 2
FATHER_DIR=output/bc/ResNet18_Adam/HiFi+CelebA+SIW/; CUDA_VISIBLE_DEVICES=3 python train.py --trainer bc_ewc MODEL.ARCH resnet18 TRAIN.PROTOCOL 2 OUTPUT_DIR output/bc_ewc/ResNet18_Adam/protocol2/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 200 MODEL.ARCH resnet18 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 500 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 128


# Adapter protocol 1
FATHER_DIR=output/vit_adapter/vit_base_patch16_224_Adam_pretrain_update_adapter/HiFi+CelebA+SIW; CUDA_VISIBLE_DEVICES=2 python train.py --trainer bc_ewc MODEL.ARCH adapter-vit_base_patch16_224 TRAIN.PROTOCOL 1 OUTPUT_DIR output/bc_ewc/vit_adapter/protocol1_2debug/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 50 TRAIN.EPOCHS 500 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 500 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 128 TRAIN.EWC_ONLINE False

# Adapter protocol 2
FATHER_DIR=output/vit_adapter/vit_base_patch16_224_Adam_pretrain_update_adapter/HiFi+CelebA+SIW; CUDA_VISIBLE_DEVICES=0 python train.py --trainer bc_ewc MODEL.ARCH adapter-vit_base_patch16_224 TRAIN.PROTOCOL 1 OUTPUT_DIR output/bc_ewc/vit_adapter/protocol1_online/ TRAIN.INIT ${FATHER_DIR}/ckpt/best.ckpt TRAIN.VAL_FREQ 10 TRAIN.EPOCHS 500 DATA.NUM_WORKERS 3 DATA.BATCH_SIZE 20 TRAIN.OPTIM.TYPE Adam DATA.IN_SIZE 224 TRAIN.INIT_LR 1e-4 TRAIN.VAL_FREQ 1 TRAIN.EPOCHS 2 MODEL.FIX_BACKBONE True TEST.BATCH_SIZE 128 TRAIN.EWC_ONLINE True




# Test


# ViT (no ewc, no
FATHER_DIR=output/bc_ewc/vit_adapter/protocol1; TASK_DIR=task_1_ft_REPLAY; CUDA_VISIBLE_DEVICES=4
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/CASIA-SURF-3DMASK-TEST.csv']" TEST.TAG TestOnCasia3DMask TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/WMCA-GRANDTEST-TEST.csv']" TEST.TAG TestOnWMCA TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/WFFD-P123-TEST.csv']" TEST.TAG TestOnWFFD TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/CASIA-SURF-COLOR-TEST.csv']" TEST.TAG TestOnCasiaSurf TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/CSMAD-TEST.csv']" TEST.TAG TestOnCSMAD TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/OULU-NPU-TEST.csv']" TEST.TAG TestOnOULU TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/HKBU-TEST.csv']" TEST.TAG TestOnHKBU TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/MSU-MFSD-TEST.csv']" TEST.TAG TestOnMSU TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/CASIA-FASD-TEST.csv']" TEST.TAG TestOnCASIA TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/REPLAY-ATTACK-TEST.csv']" TEST.TAG TestOnREPLAY TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/CeFA-RGB-TEST.csv']"  TEST.TAG TestOnCeFA TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31
python test.py --trainer bc_ewc --config ${FATHER_DIR}/train_config.yaml TEST.CKPT ${FATHER_DIR}/${TASK_DIR}/ckpt/best.ckpt OUTPUT_DIR ${FATHER_DIR}/${TASK_DIR}/ DATA.TEST "['data_list/ROSE-YOUTU-TEST.csv']"  TEST.TAG TestOnROSE TEST.BATCH_SIZE 256 DATA.NUM_WORKERS 31

