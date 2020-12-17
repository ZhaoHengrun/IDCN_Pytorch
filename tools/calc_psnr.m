close all;
clear;

file_path = 'E:\Project\VideoEnhance\IDCN_Pytorch\results\qf_10\output\'; % input路径
file_path_gt = 'E:\Project\VideoEnhance\IDCN_Pytorch\results\qf_10\gt\';  % gt路径
img_path_list = dir(strcat(file_path,'*.png')); %获取该文件夹中所有.jpg格式的图像

img_num = length(img_path_list);%获取图像总数
psnr_avr = 0;   %psnr平均值
if img_num > 0 %有满足条件的图像
    for pn = 1:img_num %逐一读取图像
        image_name = img_path_list(pn).name;% 图像名
        i1 =  imread(strcat(file_path,image_name));%读取图像
        i2 =  imread(strcat(file_path_gt,image_name));%读取图像
        fprintf('%d %s\n',pn,strcat(file_path,image_name));% 显示正在处理的图像名
        
%         [psnr_val] = compute_psnr(i1, i2);  % L
        [psnr_val, psn] = psnr(i1, i2);   % RGB
        psnr_avr = psnr_avr + psnr_val;
        fprintf('psnr: %0.4f\n', psnr_val);
    end
    psnr_avr = psnr_avr / img_num;
    fprintf('\npsnr_avr:%0.4f\n', psnr_avr);
else
    fprintf('no image');
end

% psnr_bic = compute_psnr(im_gnd,im_b)
