f2d = 'CeFA-RGB-TEST-2D.csv'
f3d = 'CeFA-RGB-TEST-3D.csv'

with open('CeFA-RGB-TEST.csv', 'r') as f, open(f2d, 'w') as f2, open(f3d, 'w') as f3:
    for x in f.readlines():
        if ',3' in x:
            f3.write(x)
        elif ',1' in x or ',2' in x:
            f2.write(x)
        else:
            f2.write(x)
            f3.write(x)
