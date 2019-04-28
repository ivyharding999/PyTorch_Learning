
# coding: utf-8

# In[1]:


from torch import nn
from torch.nn import functional as F


# In[15]:


class ResdualBlock(nn.Module):
    """
    实现子Module:Residual Block
    即ResNet中的BottleNeck 由于resnet34和>=50的情况不一样，这里只涉及2个kernel=[3,3]的卷积
    """
    def __init__(self,in_channels,middle_channels,out_channels,stride=1,shortcut=None):
        super(ResdualBlock,self).__init__()
        self.left = nn.Sequential(
                  nn.Conv2d(in_channels,middle_channels,1,1,bias=False),#直接固定stride=1
                  nn.BatchNorm2d(middle_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(middle_channels,middle_channels,3,stride,1,bias=False),#stride=2，feature map的尺寸减半。
                  nn.BatchNorm2d(middle_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(middle_channels,out_channels,1,1,bias=False),#直接固定stride=1
                  nn.BatchNorm2d(out_channels))
        self.right = shortcut
    
    def forward(self,x):
        out = self.left(x)
#         print("out:",out.shape)
        residual = x if self.right is None else self.right(x)
#         print("residual:",residual.shape)
        out += residual  # addition操作
#         print("the last out:",out.shape)
        return F.relu(out)


# In[32]:


class ResNet50(nn.Module):
    """
    实现module：resnet50
    resnet50包含多个layer,每个layer又包含多个residual block
    用子module来实现residual block,用__make_layer函数来实现layer
    nn.Conv2d()的参数：
    nn.Conv2d(in_channels, out_channels, kernel_size, 
              stride=1, padding=0, dilation=1, groups=1, bias=True)
    """
    def __init__(self,num_classes=10):
        super(ResNet50,self).__init__()
        self.stages = [64,128,256,512,1024,2048]
        #前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1) )

        #重复的layer,分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(self.stages[0],self.stages[0],self.stages[2],3)
        self.layer2 = self._make_layer(self.stages[2],self.stages[1],self.stages[3],4,stride=2)
        self.layer3 = self._make_layer(self.stages[3],self.stages[2],self.stages[4],6,stride=2)
        self.layer4 = self._make_layer(self.stages[4],self.stages[3],self.stages[5],3,stride=2)

        ## 卷积操作以后降为原来的1/32
        # 分类用的全连接
        self.fc = nn.Linear(self.stages[5],num_classes)
            
    def _make_layer(slef,in_channels,middle_channels,out_channels,block_num,stride=1):
        """
        构建layer,包含多个residule block
        in_channels:输入channel个数
        middle_channels:中间层channel个数
        out_channels:输出channel个数
        block_num:block个数
        stride=1:默认步长为1
        """
        shortcut = nn.Sequential(
                                nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
                                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResdualBlock(in_channels,middle_channels,out_channels,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResdualBlock(out_channels,middle_channels,out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.pre(x) # after pre： torch.Size([2, 64, 56, 56])卷积和池化各减少一半，变为原来的1/4
#         print("after pre：",x.shape)
        x = self.layer1(x) # after layers1： torch.Size([2, 256, 56, 56]) 尺寸不变
#         print("after layers1：",x.shape)
        x = self.layer2(x) # after layers2： torch.Size([2, 512, 28, 28]) 尺寸减半 。/2
#         print("after layers2：",x.shape)
        x = self.layer3(x) # after layers3： torch.Size([2, 1024, 14, 14]) 尺寸减半。/2
#         print("after layers3：",x.shape)
        x = self.layer4(x) # after layers4： torch.Size([2, 2048, 7, 7]) 尺寸减半。/2
#         print("after layers4：",x.shape)
        x = F.avg_pool2d(x,7) # after GAP： torch.Size([2, 2048, 1, 1]) 全局池化。尺寸为[1,1]
#         print("after GAP：",x.shape)
        x = x.view(-1,x.shape[1])
        x = self.fc(x)  # after FC： torch.Size([2, 10]) F操作
#         print("after FC：",x.shape)
        return  x   


# In[33]:


if __name__ == '__main__':
    import torch as t
    input = t.randn(2,1,224,224)
    net = ResNet50()
#     print(net)
    o = net(input)
    print(o.shape)

