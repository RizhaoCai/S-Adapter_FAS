from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "output/bc/resnet50/SIW_HIFI_MASK_CelebSpoof/"
_C.NORM_FLAG = True
_C.SEED = 666

_C.DATA = CN()
_C.DATA.DATASET='ImageDataset'
_C.DATA.TRAIN = ['data/data_list/CASIA-ALL.csv']
_C.DATA.VAL = ['data/data_list/REPLAY-ALL.csv']



_C.MODEL = CN()
_C.MODEL.ARCH = 'vit_base_patch16_224'
_C.MODEL.CONV = 'cdc_hist' # cdc 'cdc_hist_cat_conv'
_C.MODEL.CDC_THETA = 0.7 # for cdc conv or cdc 5x5 unshared




_C.MODEL.IN_CHANNELS = 3

_C.MODEL.CHANNELS = CN()
_C.MODEL.CHANNELS.RGB = True
_C.MODEL.CHANNELS.HSV = False
_C.MODEL.CHANNELS.YCRCB = False
_C.MODEL.CHANNELS.YUV = False
_C.MODEL.CHANNELS.LAB = False
_C.MODEL.CHANNELS.XYZ = False


_C.MODEL.IMAGENET_PRETRAIN = True
_C.MODEL.FIX_BACKBONE = False # True for fixing the backbone but not the adapter
_C.MODEL.FIX_NORM = True # True for fixing norm layer


_C.TRAIN = CN()
_C.TRAIN.VAL_FREQ = 200
_C.TRAIN.MAX_ITER = 1000000000
_C.TRAIN.W_GRAM = 1.0

# _C.TRAIN.CONTRAST_ALPHA = 0.0 # if 0.0: then no contrastive_loss
