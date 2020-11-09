# https://blog.csdn.net/qq_30105095/article/details/102761644
# coding=UTF-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

################## Activation Function ##################
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")
    
    def forward(self, x):
        x = x *(torch.tanh(F.softplus(x)))
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        # print("Swish activation loaded...")
    
    def forward(self, x, beta = 1):
        x = x * F.sigmoid(beta * x)
        return x

class Auto_swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    
class Auto_swish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Auto Swish activation loaded...")

    def forward(self, input_tensor):
        return Auto_swish.apply(input_tensor)

class Shift_tanh(nn.Module):
    def __init__(self):
        super(Shift_tanh, self).__init__()
        # print("Swish activation loaded...")
    
    def forward(self, x):
        x = 0.5*(torch.tanh(5*x - 2.5) + 1)
        return x
####################### backbone ########################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.5)
        if args.model_name == 'resnet18':
            self.fclass = nn.Linear(512, args.num_classes)
        else:
            self.fclass = nn.Linear(2048, args.num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax()
        # self.swish = Swish()
        # self.mish = Mish()
        self.Stanh = Shift_tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x, args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)

        x = self.dropout(x)
        embedding = x.view(x.size(0), -1)
        x = self.fclass(embedding)
        if args.loss_func == 'MSE' or args.loss_func == 'L1':
            x = self.bn2(x)
            x = self.Stanh(x)
        return x, embedding

def resnet18(args, num_classes, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        num_classes = 1000 (default)
    """
    model = ResNet(args = args, block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnet18(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet34(args, num_classes, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        num_classes = 1000 (default)
    """
    model = ResNet(args = args, block = BasicBlock, layers = [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnet34(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet50(args, num_classes, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = ResNet(args = args, block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnet50(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet101(args, num_classes, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = ResNet(args = args, block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnet101(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet152(args, num_classes, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = ResNet(args = args, block = Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnet152(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

#---------------------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SELayerX(nn.Module):
    
    def __init__(self, inplanes):
        super(SELayerX, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class SEBottleneckX(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(SEBottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.selayer = SELayerX(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, num_classes=1000):
        super(SEResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4 = self.layer4(x)

        x = self.avgpool(layer4)
        x = self.dropout(x)
        embedding = x.view(x.size(0), -1)

        x = self.fc(embedding)

        return x

def se_resnet18(num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet34(num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet50(args,num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(args = args, block = SEBottleneck, layers = [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        pretrained_dict = load_state_dict_from_url("https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def se_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnext50(num_classes, pretrained=False):
    """Constructs a SE-ResNeXt-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(SEBottleneckX, [3, 4, 6, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnext50_32x4d(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def se_resnext101(num_classes, pretrained=False):
    """Constructs a SE-ResNeXt-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(SEBottleneckX, [3, 4, 23, 3], num_classes = num_classes)
    if pretrained:
        pretrained_dict = models.resnext101_32x8d(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

#-------------------------------------------------------------------------------
class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

def Cifar_se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model

def Cifar_se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model

def Cifar_se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model

def Cifar_se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model

def Cifar_se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model

def Cifar_se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model

#-------------------------------------------------------------------------------
def w_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
def w_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class wide_Bottleneck(nn.Module):
    expansion = 4  
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(wide_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups 
        self.conv1 = w_conv1x1(inplanes, width)  
        self.bn1 = norm_layer(width)
        self.conv2 = w_conv3x3(width, width, stride, groups, dilation) 
        self.bn2 = norm_layer(width)
        self.conv3 = w_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x 
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
class wide_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(wide_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
 
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation 
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential( 
                w_conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = [] 
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))  
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4 = self.layer4(x)
 
        x = self.avgpool(layer4)
        x = self.dropout(x)
        embedding = torch.flatten(x, 1)
        x = self.fc(embedding)
 
        return x

def wide_resnet(arch, block, layers, pretrained, num_classes, progress, **kwargs):
    model = wide_ResNet(block, layers, num_classes, **kwargs)
    if pretrained:
        if arch == 'wide_resnet50_2':
            pretrained_dict = models.wide_resnet50_2(pretrained = pretrained)
        elif arch == 'wide_resnet101_2':
            pretrained_dict = models.wide_resnet101_2(pretrained = pretrained)
        num_ftrs = pretrained_dict.fc.in_features
        pretrained_dict.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = pretrained_dict.state_dict()
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def wide_resnet50_2(pretrained=False, num_classes=1000, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return wide_resnet('wide_resnet50_2', wide_Bottleneck, [3, 4, 6, 3], pretrained, num_classes, progress, **kwargs)
 
def wide_resnet101_2(pretrained=False, num_classes=1000, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return wide_resnet('wide_resnet101_2', wide_Bottleneck, [3, 4, 23, 3],pretrained, progress, **kwargs)
# ------------------------------------------------------------------------------
def select_model(args):
    if args.model_name == 'resnet18':
        if args.isPretrain == False :
            model = resnet18(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = resnet18(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'resnet50':
        if args.isPretrain == False :
            model = resnet50(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = resnet50(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'resnet101':
        if args.isPretrain == False :
            model = resnet101(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = resnet101(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'se_resnet50':
        if args.isPretrain == False :
            model = se_resnet50(num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = se_resnet50(args = args, num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'se_resnet101':
        if args.isPretrain == False :
            model = se_resnet101(num_classes = args.num_classes)
        else:
            model = se_resnet101(num_classes = args.num_classes)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'se_resneXt50':
        if args.isPretrain == False :
            model = se_resnext50(num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = se_resnext50(num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'se_resneXt101':
        if args.isPretrain == False :
            model = se_resnext101(num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = se_resnext101(num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'wide_resnet50':
        if args.isPretrain == False :
            model = wide_resnet50_2(num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = wide_resnet50_2(num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)
    elif args.model_name == 'wide_resnet101':
        if args.isPretrain == False :
            model = wide_resnet101_2(num_classes = args.num_classes, pretrained = args.pretrained_weight)
        else:
            model = wide_resnet101_2(num_classes = args.num_classes, pretrained = args.pretrained_weight)
            model = torch.load(args.save_model + '/' + args.pretrain_model)    
    model = model.cuda()
    if len(args.gpu_id.split(',')) > 1 :
        model = nn.DataParallel(model)
    return model