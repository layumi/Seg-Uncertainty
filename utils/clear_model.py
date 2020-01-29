import os

root = '../snapshots/'
nn = []
for f in os.listdir(root):
    for ff in os.listdir(root+f):
        dir_name = root+f
        for fff in os.listdir(dir_name):
            if fff =='opts.yaml':
                continue
            try:
                if int(fff[5:10])<25000:
                    dst = dir_name+'/'+fff
                    print(dst)
                    os.remove(dst)
            except:
                continue
