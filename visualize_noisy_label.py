from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

devkit_dir = 'dataset/cityscapes_list'
with open(devkit_dir+'/info.json', 'r') as fp:
  info = json.load(fp)
num_classes = np.int(info['classes'])
name_classes = np.array(info['label'], dtype=np.str)
mapping = np.array(info['label2train'], dtype=np.int)

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

img1 = 'result/cityscapesSE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/frankfurt_000001_005898_leftImg8bit.png'
img2 = 'data/Cityscapes/data/gtFine/val/frankfurt/frankfurt_000001_005898_gtFine_labelIds.png'

img1 = np.asarray(Image.open(img1))
img2 = np.asarray(Image.open(img2))
img2 = label_mapping(img2, mapping)

print(img1)
print(img2)
output = np.abs(img1-img2)
output[output>200] = 0
output[output>1] = 1

fig = plt.figure()
plt.axis('off')
heatmap = plt.imshow(output, cmap='viridis')
fig.colorbar(heatmap)
fig.savefig('label_heatmap.png')
