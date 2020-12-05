import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os
from img_scale import scale_img
from torchvision.utils import save_image

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class Sampling_data(Dataset):
    def __init__(self, path, size):
        self.path = path
        self.size = size
        "获得所有要被分割的图片名列表"
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        # print(self.name)  # ['2007_000032.png', '2007_000033.png',...]

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        ''
        "seg_name是.PNG文件"
        seg_name = self.name[index]
        # print(seg_name)
        "将当前图片名后缀改为.jpg,对应原始图"
        raw_name = seg_name[:-3] + 'jpg'
        # print(raw_name)  # 2007_000032.jpg

        "获得原始图片文件夹路径"
        raw_path = os.path.join(self.path, 'JPEGImages')
        "获得类别语义分割的图片文件夹路径"
        seg_path = os.path.join(self.path, 'SegmentationClass')
        "打开原图片"
        raw_img = Image.open(os.path.join(raw_path, raw_name))
        "打开语义分割图片"
        seg_img = Image.open(os.path.join(seg_path, seg_name))
        "获得缩放后的原始图和分割图"
        scale_rawimg = scale_img(raw_img, self.size)
        scale_segimg = scale_img(seg_img, self.size)
        "将缩放后的原始图和分割图转换成tensor格式"
        raw_tensor = transform(scale_rawimg)
        seg_tensor = transform(scale_segimg)
        return raw_tensor, seg_tensor


if __name__ == '__main__':
    i = 1
    data_path = r'G:\BaiduNetdiskDownload\VOC2012'
    zoom_size = 416
    dataset = Sampling_data(data_path, zoom_size)
    for a, b in dataset:
        print(i)
        print(a.shape)
        print(b.shape)

        save_image(a, r"D:\PycharmProjects(2)\my_U2_Net\jpg/{0}.jpg".format(i), nrow=1)
        save_image(b, r"D:\PycharmProjects(2)\my_U2_Net\png/{0}.png".format(i), nrow=1)
        i += 1
        if i == 2900:
            break
