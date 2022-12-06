import torch
#import os
import glob

root_dir = '/home/rizhao/projects/DCL_FAS/output/bc_ewc/no_ewc/adaptive_cdc_3classes/learnable/protocol1/'


all_theta_dict = {}
sessions = []
for ckpt in glob.glob(root_dir + '*/ckpt/best.ckpt'):
    print(ckpt)
    theta_dict = {}
    state_dict = torch.load(ckpt)['model_state']
    thetas = []
    for key, value in state_dict.items():
        if 'adapter' in key and 'mu' in key:
            z = value
            theta = torch.sigmoid(z)
            # theta_dict[key] = [z, theta]
            thetas.append(theta.cpu().numpy().reshape(1)[0])
    sessions.append(thetas)


for session in sessions:
    print(session)
import IPython; IPython.embed()



