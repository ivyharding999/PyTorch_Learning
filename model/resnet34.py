from torch import nn
from torch.nn import functional as F

class ResdualBlock(nn.Module):
    """
    实现子Module:Residual Block
    即ResNet中的BottleNeck 由于resnet34和>=50的情况不一样，这里只涉及2个kernel=[3,3]的卷积
    """
    def __init__(self,in_channels,out_channels,stride=1,shortcut=None):
        super(ResdualBlock,self).__init__()
        self.left = nn.Sequential(
                                  nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
                                  nn.BatchNorm2d(out_channels))
        self.right = shortcut
    
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet34(nn.Module):
    """
    实现module：resnet34
    resnet34包含多个layer,每个layer又包含多个residual block
    用子module来实现residual block,用__make_layer函数来实现layer
    nn.Conv2d()的参数：
    nn.Conv2d(in_channels, out_channels, kernel_size, 
              stride=1, padding=0, dilation=1, groups=1, bias=True)
    """
    def __init__(self,num_classes=10):
        super(ResNet34,self).__init__()
        self.stages = [64,128,256,512]
        self.pre = nn.Sequential(
                                nn.Conv2d(1,64,7,2,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2,1) )   # 尺寸/4
        #重复的layer,分别有3，4，6，3个residual block 
        self.layer1 = self._make_layer(self.stages[0],self.stages[0],3)
        self.layer2 = self._make_layer(self.stages[0],self.stages[1],4,stride=2) # 尺寸/2
        self.layer3 = self._make_layer(self.stages[1],self.stages[2],6,stride=2) # 尺寸/2
        self.layer4 = self._make_layer(self.stages[2],self.stages[3],3,stride=2) # 尺寸/2  总共 尺寸/32
        # 分类用的全连接
        self.fc = nn.Linear(self.stages[3],num_classes)
            
    def _make_layer(slef,in_channels,out_channels,block_num,stride=1):
        """
        构建layer,包含多个residule block
        in_channels:输入channel个数
        out_channels:输出channel个数
        block_num:block个数
        stride=1:默认步长为1
        """
        shortcut = nn.Sequential(
                                nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
                                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResdualBlock(in_channels,out_channels,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResdualBlock(out_channels,out_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,7)
        x = x.view(-1,x.shape[1])
        x = self.fc(x)
        return  x    

if __name__ == '__main__':
    import torch as t
    input = t.randn(2,1,224,224)
    net = ResNet34()
#     print(net)
    o = net(input)
    print(o.shape)
