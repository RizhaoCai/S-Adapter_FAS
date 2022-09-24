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
_C.MODEL.ARCH = 'resnet18'



_C.MODEL.IN_CHANNELS = 3

_C.MODEL.CHANNELS = CN()
_C.MODEL.CHANNELS.RGB = True
_C.MODEL.CHANNELS.HSV = False
_C.MODEL.CHANNELS.YCRCB = False
_C.MODEL.CHANNELS.YUV = False
_C.MODEL.CHANNELS.LAB = False
_C.MODEL.CHANNELS.XYZ = False
_C.MODEL.FIX_BACKBONE = False

_C.MODEL.IMAGENET_PRETRAIN = True