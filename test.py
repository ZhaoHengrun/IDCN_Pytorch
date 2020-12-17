from __future__ import print_function
import argparse
from os import listdir
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image

from dataset import *

# Training settings
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--quality', default=10, type=int, help='the qf of jpeg')
parser.add_argument('--input_LR_path', type=str, default='datasets/valid/qf_10/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='datasets/valid/bsd500/', help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/model.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, input_tensor)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def padding8(img):
    h, w = img.shape[0:2]
    pad_h = 8 - h % 8 if h % 8 != 0 else 0
    pad_w = 8 - w % 8 if w % 8 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'edge')
    return img


loader = transforms.Compose([
    transforms.ToTensor()])

path = opt.input_LR_path
path_HR = opt.input_HR_path

image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print(image_nums, 'test images')

sigma = get_sigma_c1(opt.quality)

for i in listdir(path):
    if is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]

            target_img = cv.imread('{}{}'.format(path_HR, i))
            # target_img = padding8(target_img)

            input_img = cv.imread('{}{}'.format(path, i))
            # input_img = padding8(input_img)

            input_tensor = transforms.ToTensor()(input_img)
            input_tensor = torch.unsqueeze(input_tensor, dim=0).float()

            input_with_label = np.concatenate([input_img,
                                               sigma[0:input_img.shape[0], 0:input_img.shape[1], :]], axis=-1)
            input = NumpyToTensor()(input_with_label)

            input = torch.unsqueeze(input, dim=0).float()

            # model = IDCN
            # model.load_state_dict(torch.load(opt.model))
            model = torch.load(opt.model, map_location='cuda:0')
            model.eval()
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)

            save_path_input = '{}qf_{}/input/'.format(opt.output_path, opt.quality)
            save_path_output = '{}qf_{}/output/'.format(opt.output_path, opt.quality)
            save_path_gt = '{}qf_{}/gt/'.format(opt.output_path, opt.quality)
            if not os.path.exists(save_path_input):
                os.makedirs(save_path_input)
            if not os.path.exists(save_path_output):
                os.makedirs(save_path_output)
            if not os.path.exists(save_path_gt):
                os.makedirs(save_path_gt)

            save_image_tensor(input[:, 0:3, :, :], '{}{}.png'.format(save_path_input, img_num))
            save_image_tensor(out, '{}{}.png'.format(save_path_output, img_num))
            cv.imwrite('{}{}.png'.format(save_path_gt, img_num), target_img)
            print('output image saved to[{}{}.png]'.format(save_path_output, img_num))
