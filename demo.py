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
from models import model as net
from tqdm import tqdm
import glob

@torch.no_grad()
def test(args, model, image_list):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])

        # resize and normalize the image 
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std

        img = img[:,:, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = Variable(img)

        if args.gpu:
            img = img.cuda()

        img_out = model(img)[:, 0, :, :].unsqueeze(dim=0)

        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)

        sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        cv2.imwrite(image_list[idx].replace(".jpg", "_edn.png"), sal_map)


 
def main(args, example_dir="examples/"):
    # read all the images in the example folder
    image_list = glob.glob("{}*.jpg".format(example_dir))

    model = net.EDN(arch=args.arch)
    
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        print("start to download the pretrained model!")
        os.system("wget -c https://github.com/yuhuan-wu/EDN/releases/download/v1.0/EDN-VGG16.pth -O pretrained/EDN-VGG16.pth")
    
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key.replace('module.', ''))
        new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict)

    if args.gpu:
        model = model.cuda()

    model.eval()
    test(args, model, image_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='vgg16', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
    parser.add_argument('--example_dir', default="examples/", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default="pretrained/EDN-VGG16.pth", help='Pretrained model')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
