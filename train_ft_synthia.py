import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter

from trainer_ms_variance import AD_Trainer
from utils.loss import CrossEntropy2d
from utils.tool import adjust_learning_rate, adjust_learning_rate_D, Timer 
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.cityscapes_pseudo_dataset import cityscapes_pseudo_DataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

AUTOAUG = False
AUTOAUG_TARGET = False

MODEL = 'DeepLab'
BATCH_SIZE = 16
ITER_SIZE = 1
NUM_WORKERS = 2
DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
DROPRATE = 0.1
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
CROP_SIZE = '640, 360'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
MAX_VALUE = 2
NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
WARM_UP = 0 # no warmup
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

LAMBDA_ME_TARGET = 0
LAMBDA_KL_TARGET = 0

TARGET = 'cityscapes'
SET = 'train'
NORM_STYLE = 'bn' # or in

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--autoaug", type=bool, default=AUTOAUG, help="use augmentation or not" )
    parser.add_argument("--autoaug_target", type=bool, default=AUTOAUG_TARGET, help="use augmentation or not" )
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--droprate", type=float, default=DROPRATE,
                        help="DropRate.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-me-target", type=float, default=LAMBDA_ME_TARGET,
                        help="lambda_me for minimize cross entropy loss on target.")
    parser.add_argument("--lambda-kl-target", type=float, default=LAMBDA_KL_TARGET,
                        help="lambda_me for minimize kl loss on target.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--max-value", type=float, default=MAX_VALUE,
                        help="Max Value of Class Weight.")
    parser.add_argument("--norm-style", type=str, default=NORM_STYLE,
                        help="Norm Style in the final classifier.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--warm-up", type=float, default=WARM_UP, help = 'warm up iteration')
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--class-balance", action='store_true', help="class balance.")
    parser.add_argument("--use-se", action='store_true', help="use se block.")
    parser.add_argument("--only-hard-label",type=float, default=0,  
                         help="class balance.")
    parser.add_argument("--train_bn", action='store_true', help="train batch normalization.")
    parser.add_argument("--sync_bn", action='store_true', help="sync batch normalization.")
    parser.add_argument("--often-balance", action='store_true', help="balance the apperance times.")
    parser.add_argument("--gpu-ids", type=str, default='0', help = 'choose gpus')
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()

# save opts
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

with open('%s/opts.yaml'%args.snapshot_dir, 'w') as fp:
    yaml.dump(vars(args), fp, default_flow_style=False)


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (h, w)

    w, h = map(int, args.input_size_target.split(','))
    args.input_size_target = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True


    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    num_gpu = len(gpu_ids)
    args.multi_gpu = False
    if num_gpu>1:
        args.multi_gpu = True
        Trainer = AD_Trainer(args)
        Trainer.G = torch.nn.DataParallel( Trainer.G, gpu_ids)
        Trainer.D1 = torch.nn.DataParallel( Trainer.D1, gpu_ids)
        Trainer.D2 = torch.nn.DataParallel( Trainer.D2, gpu_ids)
    else:
        Trainer = AD_Trainer(args)

    print(Trainer)

    trainloader = data.DataLoader(
        cityscapes_pseudo_DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    resize_size=args.input_size,
                    crop_size=args.crop_size,
                    scale=True, mirror=True, mean=IMG_MEAN, 
                    set='train', autoaug = args.autoaug, synthia=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     resize_size=args.input_size_target,
                                                     crop_size=args.crop_size,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set, autoaug = args.autoaug_target),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)


    targetloader_iter = enumerate(targetloader)

    # set up tensor board
    if args.tensorboard:
        args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0


        adjust_learning_rate(Trainer.gen_opt , i_iter, args)
        #adjust_learning_rate_D(Trainer.dis1_opt, i_iter, args)
        #adjust_learning_rate_D(Trainer.dis2_opt, i_iter, args)

        for sub_i in range(args.iter_size):

            # train G

            # train with source

            _, batch = trainloader_iter.__next__()
            _, batch_t = targetloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            images_t, labels_t, _, _ = batch_t
            images_t = images_t.cuda()
            labels_t = labels_t.long().cuda()

            with Timer("Elapsed time in update: %f"):
                loss_seg1, loss_seg2, loss_adv_target1, loss_adv_target2, loss_me, loss_kl, pred1, pred2, pred_target1, pred_target2, val_loss = Trainer.gen_update(images, images_t, labels, labels_t, i_iter)
                loss_seg_value1 += loss_seg1.item() / args.iter_size
                loss_seg_value2 += loss_seg2.item() / args.iter_size
                loss_adv_target_value1 += loss_adv_target1 / args.iter_size
                loss_adv_target_value2 += loss_adv_target2 / args.iter_size
                loss_me_value = loss_me

                if args.lambda_adv_target1 > 0 and args.lambda_adv_target2 > 0:
                    loss_D1, loss_D2 = Trainer.dis_update(pred1, pred2, pred_target1, pred_target2)
                    loss_D_value1 += loss_D1.item()
                    loss_D_value2 += loss_D2.item()
                else:
                    loss_D_value1 = 0
                    loss_D_value2 = 0

        del pred1, pred2, pred_target1, pred_target2

        if args.tensorboard:
            scalar_info = {
                'loss_seg1': loss_seg_value1,
                'loss_seg2': loss_seg_value2,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_adv_target2': loss_adv_target_value2,
                'loss_me_target': loss_me_value,
                'loss_kl_target': loss_kl,
                'loss_D1': loss_D_value1,
                'loss_D2': loss_D_value2,
                'val_loss': val_loss,
            }

            if i_iter % 100 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        '\033[1m iter = %8d/%8d \033[0m loss_seg1 = %.3f loss_seg2 = %.3f loss_me = %.3f  loss_kl = %.3f loss_adv1 = %.3f, loss_adv2 = %.3f loss_D1 = %.3f loss_D2 = %.3f, val_loss=%.3f'%(i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_me_value, loss_kl, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, val_loss))

        # clear loss
        del loss_seg1, loss_seg2, loss_adv_target1, loss_adv_target2, loss_me, loss_kl, val_loss

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(Trainer.D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(Trainer.D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(Trainer.D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(Trainer.D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
