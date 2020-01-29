import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
from evaluate_robot import colorize_mask,save
from compute_iou import label_mapping

def main(gt_dir='./data/Oxford_Robot_ICCV19/anno', devkit_dir = './dataset/robot_list/'):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    mapping = np.array(info['label2train'], dtype=np.int)
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]

    for ind in range(len(gt_imgs)):
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        label = label[:,:,0].astype(np.uint8)
        name_tmp = gt_imgs[ind].replace('anno','anno_color')
        save([label, name_tmp])
    
    return 



if __name__ == "__main__":
    main()
