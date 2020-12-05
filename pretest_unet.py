import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import model_UNet
from sample_data import Sampling_data
from torchvision.utils import save_image
from evaluation_code.dice_coefficient import dice_coeff

if __name__ == '__main__':
    with torch.no_grad() as grad:
        img_path = r'G:\BaiduNetdiskDownload\VOC2012'
        params_path = r'params/module.pth'
        img_save_path = r'./test_img'

        dataloader = DataLoader(Sampling_data(img_path, 416), 1, shuffle=True, drop_last=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model_UNet.MainNet().to(device)
        if os.path.exists(params_path):
            net.load_state_dict(torch.load(params_path))
        else:
            print("No parameters!")

        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)

        for i, (xs, ys) in enumerate(dataloader):
            x = xs.to(device)
            y = ys.to(device)
            x_ = net(x)
            dice = dice_coeff(x_, y)

            # print(y.shape)
            "将三张图像拼起来，便于保存"  # 四维还是四维，在第0轴拼接
            img = torch.cat([x, x_, y], 0)
            # img = torch.stack([x,x_,y],0)#四维变成五维了，多出一个维度
            # print(img.shape)
            save_image(img.cpu(), os.path.join(img_save_path, '{}.png'.format(i)))
            print(i)
            print("dice ", dice.item())
            if i == 10:
                break
