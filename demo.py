import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
import re
from easydict import EasyDict as edict
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='/Users/ChenYuanQin/Downloads/B_test', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='/Users/ChenYuanQin/fsdownload/CRNN_Chinese_Characters_Rec/output/checkpoints/model.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model,  device):
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # h, w,c = img.shape
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    # img = img.astype(np.float32)
    # img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    # img = img.transpose([2, 0, 1])
    # img = torch.from_numpy(img)
    # img = img.to(device)
    # img = img.view(1, *img.size())
    # if torch.cuda.is_available():
    #     img = img.cuda()
    # model.eval()
    # preds = model(img)

    transform2 = transforms.Compose([
        transforms.ToTensor()  # PIL Image/ndarray (H,W,C) [0,255] to tensor (C,H,W) [0.0,1.0]
    ])

    h, w, c = img.shape
    r = w * 1.0 / h
    standard_ratio = config.MODEL.IMAGE_SIZE.W* 1.0 / config.MODEL.IMAGE_SIZE.H
    if r > standard_ratio:
        resized_width = config.MODEL.IMAGE_SIZE.W
        resized_height = int(config.MODEL.IMAGE_SIZE.W / r)
    else:
        resized_height = config.MODEL.IMAGE_SIZE.H
        resized_width = int(config.MODEL.IMAGE_SIZE.H * r)
    image = cv2.resize(img, (0, 0), fx=resized_width / w, fy=resized_height / h, interpolation=cv2.INTER_CUBIC)

    image = image.reshape((resized_height, resized_width, 3))

    bg = np.zeros((config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 3), dtype='uint8')

    bg[:] = 255
    bg[:resized_height, :resized_width, :] = image


    image = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))



    image = transform2(image)

    image = image.view(1, *image.size())
    image = Variable(image)
    if torch.cuda.is_available():
        image = image.cuda()

    preds = model(image)




    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred

if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint,map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()
    result = open('submit.csv', 'w', encoding='utf-8')
    for temp in sorted(os.listdir(args.image_path)):
        img_path = os.path.join(args.image_path, temp)
        img = cv2.imread(img_path)
        sim_pred =recognition(config, img, model, device)
        flag =re.search(r'(\d+)年(\d+)月(\d+)日', sim_pred)
        if flag:
            aa = flag.group(2)
            sim_pred = '{}年{:0>2d}月{:0>2d}日'.format(int(flag.group(1)), int(flag.group(2)), int(flag.group(3)))

        string =  temp+'\t'+sim_pred+'\n'
        print(string)
        result.write(string)


    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

