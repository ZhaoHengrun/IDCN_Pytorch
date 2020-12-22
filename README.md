# IDCN_Pytorch
Implicit Dual-domain Convolutional Network for Robust Color Image Compression Artifact Reduction    <br/>
IDCN的pytorch复现 <br/>
参考自https://github.com/zhenngbolun/IDCN <br/>
## Requirements
Python 3.8<br/>
PyTorch 1.6.0<br/>
Numpy 1.19.2<br/>
Pillow 7.2.0<br/>
OpenCV 4.4.0.44<br/>
Visdom 0.1.8.9<br/>
Wandb 0.10.10<br/>
## Usage:
### Make datasets
所用的目录需要手动创建<br/>
#### Train_dataset
下载REDS数据集https://data.vision.ee.ethz.ch/cvl/DIV2K/ <br/>
解压后将`DIV2K/DIV2K_train_HR/`中的900张图片移动至`datasets/train_jpg/gt/`下	<br/>
运行`tools/img2jpg.py`生成指定QF的JPEG压缩图片至`datasets/train_jpg/qf_[..]/`下	<br/>
#### Test_dataset
下载BSD500数据集https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html <br/>
解压后将`BSR/BSDS500/data/images/test/`中的200张图片移动至`datasets/valid/bsd500/`下	<br/>
运行`tools/img2jpg.py`生成指定QF的JPEG压缩图片至`datasets/valid/qf_[..]/`下	<br/>
### Train
运行`python train.py`进行训练	<br/>
模型保存在`checkpoints/`目录下	<br/>
### Test&Eval
运行`python test.py`进行测试，生成的图片保存在`results/`目录下	<br/>
使用`tools/`目录下的`compute_psnr.m`计算psnr	<br/>
### Results
只使用一个dual domain block（程序中参数默认为8个），训练12270 epoch，在RGB通道上的psnr为27.6283（论文中为27.69），模型在`checkpoints/`中提供。<br/>