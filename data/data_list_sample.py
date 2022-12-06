import random



# data_list_path = '/home/rizhao/projects/DCL_FAS/data_list/CASIA-FASD-TRAIN.csv'
data_list = [
     '/home/rizhao/projects/DCL_FAS/data_list/CASIA-SURF-3DMASK-TRAIN.csv',
  '/home/rizhao/projects/DCL_FAS/data_list/WMCA-GRANDTEST-TRAIN.csv',
     '/home/rizhao/projects/DCL_FAS/data_list/WFFD-P123-TRAIN.csv',
    "/home/rizhao/projects/DCL_FAS/data_list/CASIA-SURF-COLOR-TRAIN.csv",
    "/home/rizhao/projects/DCL_FAS/data_list/CSMAD-TRAIN.csv",
    "/home/rizhao/projects/DCL_FAS/data_list/OULU-NPU-TRAIN.csv",
     "/home/rizhao/projects/DCL_FAS/data_list/HKBU-TRAIN.csv",
    '/home/rizhao/projects/DCL_FAS/data_list/MSU-MFSD-TRAIN.csv',
     "/home/rizhao/projects/DCL_FAS/data_list/CASIA-FASD-TRAIN.csv",
    "/home/rizhao/projects/DCL_FAS/data_list/REPLAY-ATTACK-TRAIN.csv"




]



def sampling(data_list_path):
    #save_data_list_path_s10 = data_list_path.replace('.csv', '-s10.csv')
    save_data_list_path_s50 = data_list_path.replace('.csv', '-s50.csv')
    #save_data_list_path_s100 = data_list_path.replace('.csv', '-s100.csv')



    # Read file
    with open(data_list_path, 'r') as f:
        lines = f.readlines()

    # Split real and spoof

    def sample(real_face_list, spoofing_face_list, idx):

        for line in lines:

            if ',0' in line and ('{}.png'.format(idx) in line or '{}.jpg'.format(idx) in line or '{}.JPG'.format(idx) in line ):
                real_face_list.append(line)

            if ',0' not in line and ('{}.png'.format(idx) in line or '{}.jpg'.format(idx) in line or '{}.JPG'.format(idx) in line):
                spoofing_face_list.append(line)

    real_face_list = []
    spoofing_face_list = []
    for idx in ['1', '10', '5', '7', '3', '_L', '_R']:
        if len(real_face_list)<100 or len(spoofing_face_list)<100:
            sample(real_face_list, spoofing_face_list, idx)
        else:
            break
    #import pdb;pdb.set_trace()
    num_sample = 5
    random.shuffle(real_face_list)
    real_list_s5 = random.sample(real_face_list, k=num_sample)

    random.shuffle(spoofing_face_list)
    spoof_list_s5 = random.sample(spoofing_face_list, k=num_sample)

    num_sample = 50
    random.shuffle(real_face_list)
    real_list_s50 = random.sample(real_face_list, k=num_sample)
    random.shuffle(spoofing_face_list)
    spoof_list_s50 = random.sample(spoofing_face_list, k=num_sample)

    num_sample = 25
    random.shuffle(real_face_list)
    real_list_s25 = random.sample(real_face_list, k=num_sample)
    random.shuffle(spoofing_face_list)
    spoof_list_s25 = random.sample(spoofing_face_list, k=num_sample)


    # with open(save_data_list_path_s10, 'w') as f:
    #     print('Writing to ', save_data_list_path_s10)
    #     for x in real_list_s5:
    #         f.write(x)
    #     for x in spoof_list_s5:
    #         f.write(x)

    with open(save_data_list_path_s50, 'w') as f:
        print('Writing to ', save_data_list_path_s50)
        for x in real_list_s25:
            f.write(x)
        for x in spoof_list_s25:
            f.write(x)

    # with open(save_data_list_path_s100, 'w') as f:
    #     print('Writing to ', save_data_list_path_s100)
    #     for x in real_list_s50:
    #         f.write(x)
    #     for x in spoof_list_s50:
    #         f.write(x)


for f in data_list:
    sampling(f)
