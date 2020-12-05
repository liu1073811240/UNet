import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import model_UNet
from sample_data import Sampling_data
from torchvision.utils import save_image
from evaluation_code.segmentation_loss import SoftDiceLoss, Focal_Loss

if __name__ == '__main__':

    img_path = r'G:\BaiduNetdiskDownload\VOC2012'
    # params_path = r'params/focal+dice_module.pth'
    params_path = r'params2/bce+dice_module.pth'
    # params_path = r'params/module.pth'
    img_save_path = r'./train_img'
    epoch = 1
    dataloader = DataLoader(Sampling_data(img_path, 416), 1, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_UNet.MainNet().to(device)

    if os.path.exists(params_path):
        net.load_state_dict(torch.load(params_path))
    else:
        print("No parameters!")

    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(),lr=1e-3,momentum=0.9)
    # mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    dice_loss = SoftDiceLoss()
    # focal_loss = Focal_Loss(alpha=0.5, gamma=2)
    # focal_loss = Focal_Loss_sigmoid(gamma=2)

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    if not os.path.exists("./params2"):
        os.mkdir("./params2")

    while True:
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.to(device)
            ys = ys.to(device)
            xs_ = net(xs)

            loss1 = bce_loss(xs_, ys)
            # loss1 = focal_loss(xs_, ys)
            loss2 = dice_loss(xs_, ys)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('epoch:{},count:{},loss-1:{:.3f},loss-2:{:.3f},loss:{:.3f}'.
                      format(epoch, i, loss1, loss2, loss))

                x = xs[0]
                x_ = xs_[0]
                y = ys[0]

                # print(y.shape)
                "将三张图像堆叠起来，便于保存"  # 三维变四维，堆叠后多一个维度
                img = torch.stack([x, x_, y], 0)
                # img = torch.cat([x,x_,y],0)#拼接不会增加维度，还是三维
                # print(img.shape)
                save_image(img.cpu(), os.path.join(img_save_path, '{}.png'.format(epoch)))

                # z = torch.cat((x, x_, y),2)
                # print(z.shape)
                # img_save = transforms.ToPILImage()(z.cpu())
                # img_save.save(os.path.join(img_save_path, '{}.png'.format(epoch)))

        torch.save(net.state_dict(), params_path)
        print('Model parameters saved successfully !')
        epoch += 1
        if epoch == 300:
            break
