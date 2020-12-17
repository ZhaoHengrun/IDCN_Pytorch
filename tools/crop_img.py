import os
from os import listdir
import cv2 as cv

input_path = '../datasets/valid/bsd500/'
output_path = '../datasets/valid/bas500_cropped/'

if not os.path.exists(output_path):
    os.makedirs(output_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class Crop(object):
    def __call__(self, img_input, h_decrease, w_decrease):
        h, w = img_input.shape[0], img_input.shape[1]
        new_h, new_w = h - h_decrease, w - w_decrease

        new_input = img_input[h - new_h:h, 0:new_w]
        return new_input


for i in listdir(input_path):
    if is_image_file(i):
        img_path = '{}{}'.format(input_path, i)
        output_dir = '{}{}'.format(output_path, i)
        img = cv.imread(img_path)
        img = Crop()(img, 1, 1)
        print('saving:[{}]'.format(output_dir))
        cv.imwrite(output_dir, img, [cv.IMWRITE_PNG_COMPRESSION, 0])
        # img.save(output_dir, format='JPEG', quality=100)
