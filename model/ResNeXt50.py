
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch as t


# In[39]:


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, base_width, widen_factor,stride):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.left = nn.Sequential(
                      nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False),#直接固定stride=1
                      nn.BatchNorm2d(D),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),#stride=2，feature map的尺寸减半。
                      nn.BatchNorm2d(D),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False),#直接固定stride=1
                      nn.BatchNorm2d(out_channels))
        
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
                                 nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        out = self.left(x)
#         print("out:",out.shape)
        residual = self.shortcut.forward(x)
#         print("residual:",residual.shape)
        out += residual  # addition操作
#         print("the last out:",out.shape)
        return F.relu(out)


# In[40]:


class ResNeXt50(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, nlabels, base_width, widen_factor=2):
        """ Constructor
        Args:
            cardinality: number of convolution groups.论文中分组卷积个数为32
            depth: number of layers. 我暂时理解为block的个数
            nlabels: number of classes  分类类别数
            base_width: base number of channels in each group.分组卷积中基本的通道数量
            widen_factor: factor to adjust the channel dimensionality通道扩增系数，这里默认为4倍
        """
        super(ResNeXt50, self).__init__()
#         self.block_depth = (depth - 2) // 9 # //表示向下取整
        self.stages = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor, 512 * widen_factor,1024 * widen_factor]
        # [64,128,256,512,1024,2048]
        self.pre = nn.Sequential(
                                nn.Conv2d(3,64,7,2,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2,1) ) # 尺寸/4
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[2], 3, cardinality, base_width, widen_factor, stride=1) #输入64，输出256
        self.stage_2 = self.block('stage_2', self.stages[2], self.stages[3], 4, cardinality, base_width, widen_factor, stride=2) #输入256，输出512
        self.stage_3 = self.block('stage_3', self.stages[3], self.stages[4], 6, cardinality, base_width, widen_factor, stride=2) #输入512，输出1024
        self.stage_4 = self.block('stage_4', self.stages[4], self.stages[5], 3, cardinality, base_width, widen_factor, stride=2) #输入1024，输出2048
        self.classifier = nn.Linear(self.stages[5], nlabels) # 即y=wx+b,这里的x：self.stage[3],y:nlabel                                        
        init.kaiming_normal_(self.classifier.weight) # 参数初始化，主要初始化W
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, block_num, cardinality,base_width, widen_factor, stride=1):
        """ 
        构建layer,包含多个resnext block
        in_channels:输入channel个数
        middle_channels:中间层channel个数
        out_channels:输出channel个数
        block_num:block个数
        stride=1:默认步长为1
        """
#         shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
#                                  nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResNeXtBottleneck(in_channels,out_channels, cardinality, base_width, widen_factor,stride))
        
        for i in range(1,block_num):
            layers.append(ResNeXtBottleneck(out_channels,out_channels, cardinality, base_width, widen_factor,1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x) # after pre： torch.Size([2, 64, 56, 56])卷积和池化各减少一半，变为原来的1/4
#         print("after pre：",x.shape)
        x = self.stage_1.forward(x)   # 3*3
#         print("stage_1:",x.shape)
        x = self.stage_2.forward(x)
#         print("stage_2:",x.shape)
        x = self.stage_3.forward(x)
#         print("stage_3:",x.shape)
        x = self.stage_4.forward(x)
#         print("stage_4:",x.shape)
        x = F.avg_pool2d(x, 8) # 我们想用GAP替代FC，也就是我们获得的feature map尺寸为[1,1],也就是一个数
#         print("average pooling::",x.shape)
#         x = x.view(-1, self.stages[3])
        x = x.view(x.size()[0],-1)
#         print("faltten:",x.shape)
        x = self.classifier(x)
#         x = self.classifier2(x)
#         print("after fc:",x.shape)
        return x


# In[42]:


if __name__ == '__main__':
    input = t.randn(2,3,224,224)
    net = ResNeXt50(cardinality=32, nlabels=10, base_width=4, widen_factor=4)
#     print(net)
    o = net(input)
    print(o.shape)

