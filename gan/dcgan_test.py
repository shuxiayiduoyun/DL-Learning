#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：dcgan_test.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/9/24 14:57 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import torchvision
from torchvision import transforms

# 加载数据
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])

train_ds = torchvision.datasets.MNIST('D:\datasets',
                                      train=True,
                                      transform=transform,
                                      download=False)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)


# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(256 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                        kernel_size=(3, 3),
                                        stride=1,
                                        padding=1
                                        )  # 得到128*7*7的图像
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1  # 64*14*14
                                        )
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                        kernel_size=(4, 4),
                                        stride=2,
                                        padding=1  # 1*28*28
                                        )

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.tanh(self.deconv3(x))
        return x


# 定义判别器
# input:1，28，28
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)  # 第一层不适用bn  64，13，13
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 128，6，6
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 6 * 6, 1)  # 输出一个概率值

    def forward(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))  # (batch, 128,6,6)
        x = self.bn(x)
        x = x.view(-1, 128 * 6 * 6)  # (batch, 128,6,6)--->  (batch, 128*6*6)
        x = torch.sigmoid(self.fc(x))
        return x


# 初始化模型
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
gen = Generator().to(device)
dis = Discriminator().to(device)

# 损失计算函数
loss_function = nn.BCELoss()

# 定义优化器
d_optim = torch.optim.Adam(dis.parameters(), lr=1e-5)
g_optim = torch.optim.Adam(gen.parameters(), lr=1e-4)

test_input = torch.randn(16, 100, device=device)

# 开始训练
D_loss = []
G_loss = []
# 训练循环
for epoch in range(30):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    # 对全部的数据集做一次迭代
    for i, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.shape[0]  # 返回img的第一维的大小
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()  # 将上述步骤的梯度归零
        real_output = dis(img)  # 对判别器输入真实的图片，real_output是对真实图片的预测结果
        d_real_loss = loss_function(real_output, torch.ones_like(real_output, device=device))
        d_real_loss.backward()  # 求解梯度

        # 得到判别器在生成图像上的损失
        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_function(fake_output, torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()
        torchvision.utils.save_image(gen_img, fp=f'results/image_{epoch}.png')
    print('Epoch:', epoch)
