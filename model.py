import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,num_joints=14):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_joints=num_joints

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_conv=nn.Conv2d(512 * block.expansion, 64, kernel_size=1, stride=1,bias=False)
        
        self.fc1 = nn.Linear(64*8*8, 256)
        self.b1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(256, 256)
        self.b2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 3*self.num_joints)
        self.dropout1=nn.Dropout(p=0.15)
        #self.dropout2=nn.Dropout(p=0.15)
        #self.dropout3=nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=self.final_conv(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x) # Relu should be added, and batch norm
        x = self.b1(x)
        x = self.relu(x)
        #x = self.dropout1(x)

        
        x = self.fc2(x)
        x =  self.b2(x)
        x = self.relu(x)
#         #x= self.dropout2(x)
        x = self.fc3(x)
        x=x.reshape(-1,self.num_joints,3)
        #UV=x[:,:,0:2]
        #D=x[:,:,2]
        return x





######################################################################################
class AdaptiveSpatialSoftmaxLayer(nn.Module):
    def __init__(self,spread=None,train=True,num_channel=14):
        super(AdaptiveSpatialSoftmaxLayer, self).__init__()
        # spread should be a torch tensor of size (1,num_chanel,1)
        # the softmax is applied over spatial dimensions 
        # train determines whether you would like to train the spread parameters as well
        #self.SpacialSoftmax = nn.Softmax(dim=2)
        if train:
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #print(device)
            self.spread=nn.Parameter(torch.ones(1,num_channel,1))#.to(device))
            # self.spread.requires_grad=True
        else:
            self.spread=spread#.to(device)
            #if spread is not None:
            #    self.spread.requires_grad=False



    def forward(self, x):
        # the input is a tensor of shape (batch,num_channel,height,width)
        SpacialSoftmax = nn.Softmax(dim=2)
        num_batch=x.shape[0]
        num_channel=x.shape[1]
        height=x.shape[2]
        width=x.shape[3]
        inp=x.view(num_batch,num_channel,-1)
        #if self.spread is not None:
        res=torch.mul(inp,self.spread)
        res=SpacialSoftmax(inp)
        
        return res.reshape(num_batch,num_channel,height,width)


class Bottleneck1(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, BN,num_G,stride=1,downsample=None):
        super(Bottleneck1, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes) if BN else nn.GroupNorm(num_G, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth,BN,num_G):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.BN = BN
        self.num_G = num_G
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth,BN,num_G)

    def _make_residual(self, block, num_blocks, planes,BN,num_G):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes,BN=BN,num_G=num_G))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth,BN,num_G):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes,BN,num_G))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes,BN,num_G))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)
    
    

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=14,BN=True,num_G=16): #BN ture -> batch normalization, otherwise Group Normalization
        super(HourglassNet, self).__init__()
        #self.sp=nn.Parameter(torch.ones(1,num_classes,1) )
        #self.sp=torch.requires_grad=True
        self.soft0=AdaptiveSpatialSoftmaxLayer(train=True,num_channel=num_classes)#.cuda()#to(device)
        if num_stacks>1:
            self.soft1=AdaptiveSpatialSoftmaxLayer(train=True,num_channel=num_classes)#.cuda()#to(device)
        self.Xs=GetValuesX()
        self.Ys=GetValuesY()
        self.Xs.requires_grad=False
        self.Ys.requires_grad=False

        self.BN=BN
        self.num_G=num_G
        self.num_classes = num_classes
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) if BN else nn.GroupNorm(num_G, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes,1,BN,num_G )
        self.layer2 = self._make_residual(block, self.inplanes, 1,BN,num_G )
        self.layer3 = self._make_residual(block, self.num_feats, 1,BN,num_G)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        resD, fcD, scoreD, fc_D, score_D=[], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4,BN,num_G))
            res.append(self._make_residual(block, self.num_feats, num_blocks,BN,num_G))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            
            resD.append(self._make_residual(block, self.num_feats, num_blocks,BN,num_G))
            conv_temp=nn.Conv2d(ch, 128, kernel_size=1, bias=True)
            bn_temp=nn.BatchNorm2d(128) if BN else nn.GroupNorm(num_G, 128)
            fcD.append(nn.Sequential(self._make_fc(ch, ch),conv_temp,bn_temp,self.relu))
            scoreD.append(nn.Conv2d(128, num_classes, kernel_size=1, bias=True))
            
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
                
                fc_D.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_D.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.resD = nn.ModuleList(resD)
        self.fcD = nn.ModuleList(fcD)
        self.scoreD = nn.ModuleList(scoreD)
        self.fc_D = nn.ModuleList(fc_D)
        self.score_D = nn.ModuleList(score_D)


    def _make_residual(self, block, planes, blocks, BN,num_G,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes,BN,num_G, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,BN,num_G))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes) if self.BN else nn.GroupNorm(self.num_G, inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x , return_heatmap=False):
        out = []
        outD= []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            d=y
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            
            yD = self.resD[i](d)
            hh=yD
            yD = self.fcD[i](yD)
            scoreD = self.scoreD[i](yD)

            out.append(score)
            outD.append(scoreD)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                
                fc_D = self.fc_D[i](hh)
                score_D = self.score_D[i](scoreD)
                x = x + fc_ + score_ + fc_D + score_D

    
        num_batch=x.shape[0]
        h0=self.soft0(out[0])
        X0=torch.mul(h0.view(num_batch,self.num_classes,-1),self.Xs)
        X0=torch.sum(X0,dim=-1)
        Y0=torch.mul(h0.view(num_batch,self.num_classes,-1),self.Ys)
        Y0=torch.sum(Y0,dim=-1)
        X0=torch.unsqueeze(X0,dim=-1)
        Y0=torch.unsqueeze(Y0,dim=-1)
        UV0=torch.cat((X0,Y0),dim=-1)
        
        d0=outD[0]
        l0=h0.view(num_batch,self.num_classes,-1)
        d0=d0.view(num_batch,self.num_classes,-1)
        D0=torch.sum(torch.mul(d0,l0),dim=-1)

            
        if self.num_stacks==1:
            uv=UV0*2
            uvd=torch.cat([uv,D0[...,None]],dim=-1)

            if return_heatmap:
                return uvd, h0
            return uvd
        else:
            h1=self.soft1(out[1])
            X=torch.mul(h1.view(num_batch,self.num_classes,-1),self.Xs)
            X=torch.sum(X,dim=-1)
            Y=torch.mul(h1.view(num_batch,self.num_classes,-1),self.Ys)
            Y=torch.sum(Y,dim=-1)
            X=torch.unsqueeze(X,dim=-1)
            Y=torch.unsqueeze(Y,dim=-1)
            UV1=torch.cat((X,Y),dim=-1)

            d1=outD[1]
            l1=h1.view(num_batch,self.num_classes,-1)
            d1=d1.view(num_batch,self.num_classes,-1)
            D1=torch.sum(torch.mul(d1,l1),dim=-1)

            return UV0,D0,UV1,D1





def GetValuesX(dimension=64,num_channel=14):
    n=dimension
    num_channel=14
    vec=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    Xs=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    for i in range(n-1):
        Xs=np.concatenate([Xs,vec],axis=1)

    #Xs=np.repeat(Xs,num_channel,axis=0)
    Xs=np.float32(np.expand_dims(Xs,axis=0))
    return nn.Parameter(torch.from_numpy(Xs))

def GetValuesY(dimension=64,num_channel=14):
    res=np.zeros((1,dimension*dimension))
    for i in range(dimension):
        res[0,(i*dimension):((i+1)*dimension)]=i
    res=np.float32( np.expand_dims(res,axis=0) )
    return nn.Parameter(torch.from_numpy(res))



def model_builder(model_name,num_joints):
    assert model_name in ["resnet18","resnet50","hourglass_1"]

    if model_name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2],num_joints=num_joints)
    elif model_name == "resnet50":
        return ResNet(BasicBlock, [3, 4, 6, 3],num_joints=num_joints)
    elif model_name == "hourglass_1":
        return HourglassNet(Bottleneck1,num_stacks=1,num_blocks=2, num_classes=num_joints,BN=True,num_G=16)
    else:
        raise NotImplementedError
