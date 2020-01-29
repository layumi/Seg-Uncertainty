import torch
import time

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_step(base_lr, iter):
    lr = base_lr
    if iter>40000:
        lr = base_lr * 0.5
    if iter>60000:
        lr = base_lr * 0.5 * 0.5
    if iter>70000:
        lr = base_lr * 0.5 * 0.5 * 0.5
    return lr

def adjust_learning_rate(optimizer, i_iter, args):
    if i_iter < args.warm_up:
        lr = args.learning_rate * (0.1 + 0.9 * i_iter / args.warm_up)
    else: 
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
        #lr = lr_step(args.learning_rate, i_iter)
    optimizer.param_groups[0]['lr'] = lr
    print('-------lr_G: %f-------'%lr)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    #if len(optimizer.param_groups) > 1:
    #    optimizer.param_groups[1]['lr'] = lr * 10


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
