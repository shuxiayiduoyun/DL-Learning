#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：gan_test.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/9/24 10:47 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# 对数据做归一化 （-1， 1）
transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换成Tensor格式，channel, high, witch,数据在（0， 1）范围内
    transforms.Normalize(0.5, 0.5)  # 通过均值和方差将数据归一化到（-1， 1）之间
])

train_ds = torchvision.datasets.MNIST('D:\datasets',
                                    train=True,
                                    transform=transform,
                                    download=False)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# 返回一个批次的数据
imgs, _ = next(iter(dataloader))

# 输入是长度为 100 的 噪声（正态分布随机数）
# 输出为（1， 28， 28）的图片
# linear 1 :   100----256
# linear 2:    256----512
# linear 2:    512----28*28
# reshape:     28*28----(1, 28, 28)

class Generator(nn.Module):  # 创建的 Generator 类继承自 nn.Module
    def __init__(self):  # 定义初始化方法
        super(Generator, self).__init__()  # 继承父类的属性
        self.main = nn.Sequential(  # 使用Sequential快速创建模型
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()  # 输出层使用Tanh()激活函数，使输出-1, 1之间
        )

    def forward(self, x):  # 定义前向传播 x 表示长度为100 的noise输入
        img = self.main(x)
        img = img.view(-1, 28, 28)  # 将img展平，转化成图片的形式，channel为1可写可不写
        return img


## 输入为（1， 28， 28）的图片  输出为二分类的概率值，输出使用sigmoid激活 0-1
# BCEloss计算交叉熵损失

# nn.LeakyReLU   f(x) : x>0 输出 x， 如果x<0 ,输出 a*x  a表示一个很小的斜率，比如0.1
# 判别器中一般推荐使用 LeakyReLU

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 输入是28*28的张量，也就是图片
            nn.LeakyReLU(),  # 小于0的时候保存一部分梯度
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),  # 二分类问题，输出到1上
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.main(x)
        return x

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 初始化模型
gen = Generator().to(device)
dis = Discriminator().to(device)
# 优化器
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
# 损失函数
loss_fn = nn.BCELoss()


def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)  # 确保prediction[i] + 1)/2输出的结果是在0-1之间
        plt.axis('off')
    # plt.show()
    plt.savefig(f'results/{epoch}.png')
    plt.close(fig)


test_input = torch.randn(16, 100, device=device)

# 保存每个epoch所产生的loss值
D_loss = []
G_loss = []

# 训练循环
for epoch in range(100):  # 训练20个epoch
    d_epoch_loss = 0  # 初始损失值为0
    g_epoch_loss = 0
    # len(dataloader)返回批次数，len(dataset)返回样本数
    count = len(dataloader)
    # 对dataloader进行迭代
    for step, (img, _) in enumerate(dataloader):  # enumerate加序号
        img = img.to(device)
        size = img.size(0)  # 获取每一个批次的大小
        random_noise = torch.randn(size, 100, device=device)  # 随机噪声的大小是size个

        d_optim.zero_grad()  # 将判别器前面的梯度归0

        real_output = dis(img)  # 判别器输入真实的图片，real_output是对真实图片的预测结果

        # 得到判别器在真实图像上的损失
        # 判别器对于真实的图片希望输出的全1的数组，将真实的输出与全1的数组进行比较
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()  # 求解梯度

        gen_img = gen(random_noise)
        # 判别器输入生成的图片，fake_output是对生成图片的预测
        # 优化的目标是判别器，对于生成器的参数是不需要做优化的，需要进行梯度阶段，detach()会截断梯度，
        # 得到一个没有梯度的Tensor，这一点很关键
        fake_output = dis(gen_img.detach())
        # 得到判别器在生成图像上的损失
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()  # 求解梯度

        d_loss = d_real_loss + d_fake_loss  # 判别器总的损失等于两个损失之和
        d_optim.step()  # 进行优化

        g_optim.zero_grad()  # 将生成器的所有梯度归0
        fake_output = dis(gen_img)  # 将生成器的图片放到判别器中，此时不做截断，因为要优化生成器
        # 生层器希望生成的图片被判定为真
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))  # 生成器的损失
        g_loss.backward()  # 计算梯度
        g_optim.step()  # 优化

        # 将损失累加到定义的数组中，这个过程不需要计算梯度
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    # 计算每个epoch的平均loss，仍然使用这个上下文关联器
    with torch.no_grad():
        # 计算平均的loss值
        d_epoch_loss /= count
        g_epoch_loss /= count
        # 将平均loss放入到loss数组中
        D_loss.append(d_epoch_loss.item())
        G_loss.append(g_epoch_loss.item())
        print(f'Epoch: {epoch}, D_Loss: {d_epoch_loss.item():.4f}, G_Loss: {g_epoch_loss.item():.4f}')
        # 调用绘图函数
        gen_img_plot(gen, epoch, test_input)
