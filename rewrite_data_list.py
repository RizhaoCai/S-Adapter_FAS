

# Get data list path
from glob import glob


for x in glob('data_list/*.csv'):
    with open(x, 'r') as f:
        new_lines = []
        for line in f.readlines():
            if '/home/rizhao/data/frames/':

                new_lines.append(line.replace('/home/rizhao/data/FAS/frames/', '/home/Dataset/Face_Spoofing/frames/'))
            else:
                break
    if new_lines is not []:
        with open(x, 'w') as f:
            for k in new_lines:
                print(k)
                f.write(k)





# open data list
# read

# writew