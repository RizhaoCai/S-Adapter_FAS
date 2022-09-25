from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "output/bc/resnet50/SIW_HIFI_MASK_CelebSpoof/"
_C.NORM_FLAG = True
_C.SEED = 666

_C.DATA = CN()
_C.DATA.DATASET='ImageDataset'

_C.DATA.PROTOCOL1 = CN()
_C.DATA.PROTOCOL1.TRAIN = ['data_list/REPLAY-ATTACK-TRAIN-s100.csv', 'data_list/CASIA-FASD-TRAIN-s100.csv', 'data_list/MSU-MFSD-TRAIN-s100.csv',
                            'data_list/HKBU-TRAIN-s100.csv', 'data_list/OULU-NPU-TRAIN-s100.csv', 'data_list/CSMAD-TRAIN-s100.csv',
                           'data_list/CASIA-SURF-COLOR-TRAIN-s100.csv', 'data_list/WFFD-P123-TRAIN-s100.csv',
                            'data_list/WMCA-GRANDTEST-TRAIN-s100.csv', 'data_list/CASIA-SURF-3DMASK-TRAIN-s100.csv'
                           ]



_C.DATA.PROTOCOL1.VAL = [   'data_list/REPLAY-ATTACK-TEST.csv', 'data_list/CASIA-FASD-TEST.csv', 'data_list/MSU-MFSD-TEST.csv',
                        'data_list/HKBU-TEST.csv', 'data_list/OULU-NPU-TEST.csv', 'data_list/CSMAD-TEST.csv',
                            'data_list/CASIA-SURF-COLOR-TEST.csv', 'data_list/WFFD-P123-TEST.csv',
                            'data_list/WMCA-GRANDTEST-TEST.csv', 'data_list/CASIA-SURF-3DMASK-TEST.csv'
]
# _C.DATA.PROTOCOL1.TASK_NAME = {
#     1:'ft_REPLAY', 2:'ft_CASIA', 3:'ft_MSU', 4:'ft_HKBU', 5:'ft_OULU', 6:'ft_CSMAD',
#     7:'ft_CasiaSurf', 8:'ft_WFFD', 9:'ft_WMCA', 10:'ft_Casia3DMask'
# }
_C.DATA.PROTOCOL1.TASK_NAME = [
     'ft_REPLAY', 'ft_CASIA', 'ft_MSU', 'ft_HKBU', 'ft_OULU', 'ft_CSMAD',
    'ft_CasiaSurf', 'ft_WFFD', 'ft_WMCA', 'ft_Casia3DMask'
]

_C.DATA.PROTOCOL2 = CN()
_C.DATA.PROTOCOL2.TRAIN = _C.DATA.PROTOCOL1.TRAIN.copy()
_C.DATA.PROTOCOL2.TRAIN.reverse()
_C.DATA.PROTOCOL2.VAL =   _C.DATA.PROTOCOL1.VAL.copy()
_C.DATA.PROTOCOL2.VAL.reverse()
_C.DATA.PROTOCOL2.TASK_NAME = _C.DATA.PROTOCOL1.TASK_NAME.copy()
_C.DATA.PROTOCOL2.TASK_NAME.reverse()


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


_C.TRAIN = CN()
_C.TRAIN.PROTOCOL = 1
_C.TRAIN.VAL_AFTER = 1
_C.TRAIN.EWC_LAMBDA = 1.0
