from os import *
import cv2

input_path = '../datasets/valid/qp_37_video/'
output_path = '../datasets/valid/qp_37_img/'


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4"])


for i in listdir(input_path):
    if is_video_file(i):
        cap = cv2.VideoCapture('{}{}'.format(input_path, i))
        ret, frame = cap.read()
        output_name = i.replace('_batch.mp4', '.jpg')
        output_dir = '{}{}'.format(output_path, output_name)

        print('saving:[{}]'.format(output_dir))
        cv2.imwrite(output_dir, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print('finish')
