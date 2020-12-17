from os import listdir
import cv2 as cv

input_path = '../datasets/valid/bsd500/'
output_path = '../datasets/valid/yuv420/'

path = './frames/'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


fps = 30
size = (1920, 1080)

for i in listdir(input_path):
    if is_image_file(i):
        img_path = '{}{}'.format(input_path, i)
        output_name = i.replace('.jpg', '.avi')
        output_dir = '{}{}'.format(output_path, output_name)
        img = cv.imread(img_path)
        h = img.shape[0]
        w = img.shape[1]
        video = cv.VideoWriter(output_dir, cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, (w, h))
        print('saving:[{}]'.format(output_dir))
        video.write(img)
        video.release()
print('finish')
