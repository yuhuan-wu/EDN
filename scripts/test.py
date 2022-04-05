import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from saleval import SalEval
from models import model as net
from tqdm import tqdm

@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    eval = SalEval()

    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255

        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std

        img = img[:,:, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = Variable(img)

        label = torch.from_numpy(label).float().unsqueeze(0)

        if args.gpu:
            img = img.cuda()
            label = label.cuda()

        img_out = model(img)[:, 0, :, :].unsqueeze(dim=0)

        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        eval.add_batch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

        sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '.png'), sal_map)

    F_beta, MAE = eval.get_metric()
    print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))

 
def main(args, file_list):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(args.data_dir + '/' + file_list + '.txt') as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())

    model = net.EDN(arch=args.arch)
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key.replace('module.', ''))
        new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict, strict=False)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    save_dir = args.savedir + '/' + file_list + '/'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    test(args, model, image_list, label_list, save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='vgg16', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default=None, help='Pretrained model')

    args = parser.parse_args()
    
    import shutil

    if 'Lite' in args.pretrained:
        args.arch = 'mobilenetv2'
    elif 'VGG16' in args.pretrained:
        args.arch = 'vgg16'
    elif 'R50' in args.pretrained:
        args.arch = 'resnet50'
    else:
        raise NotImplementedError("recognized unknown backbone given the model_path")
    
    if 'LiteEX' in args.pretrained:
        # EDN-LiteEX
        args.width = 224
        args.height = 224
    
    print('Called with args:')
    print(args)

    data_lists = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'PASCAL-S', 'HKU-IS']
    for data_list in data_lists:
        print("processing ", data_list)
        main(args, data_list)
