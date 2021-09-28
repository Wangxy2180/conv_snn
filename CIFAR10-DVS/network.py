import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
thresh = 0.5  # neuronal threshold
lens = 0.5 / 3  # hyper-parameters of approximate function
decay = 0.8  # decay constants


# approximate firing function
class ActFun(torch.autograd.Function):
    # 静态方法没有self?不太清楚，回头研究一下静态方法，使用ctx(context)
    # 自定义的forward和backward第一个参数必须是ctx，可以保存forward中的变量，再backward中继续使用
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # 阈值0.5
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = torch.exp(-(input - thresh) ** 2 / (2 * lens ** 2)) / ((2 * lens * 3.141592653589793) ** 0.5)
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update
def mem_update(x, mem, spike):
    # 发生脉冲的，置0，并衰减，再加上新来的电流
    mem = mem * decay * (1. - spike) + x  # with rest mechanism
    # mem = mem * decay + x                 # no rest mechanism
    # 超出阈值的，产生脉冲. spike是float的？
    spike = act_fun(mem)  # act_fun: approximation firing function
    return mem, spike


# 这个也是ResNet中有的
class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, image_size, batch_size, stride=1, downsample=None):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.drop1 = nn.Dropout(0.6)   # This brings imbalacement
        self.drop2 = nn.Dropout(0.2)
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        w, h = image_size
        # --- custom initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if not isinstance(m.bias, torch.nn.parameter.Parameter):
        #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             m.weight.data.fill_(np.sqrt(1 / n))
        #         else:
        #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             m.weight.data.normal_(0, np.sqrt(1. / n))
        #             m.bias.data.uniform_(0, 0.4)
        # print(m._tracing_name)

        #         # m.weight.data.normal_(0, 1)
        #         # print(type(m.bias))

    def forward(self, x, c1_mem, c1_spike, c2_mem, c2_spike):
        # 先给自己做个备份
        residual = x
        out = self.conv1(x)
        c1_mem, c1_spike = mem_update(out, c1_mem, c1_spike)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # Addition
        out += residual
        c2_mem, c2_spike = mem_update(out, c2_mem, c2_spike)
        c2_spike = self.drop2(c2_spike)
        return c2_spike, c1_mem, c1_spike, c2_mem, c2_spike


# 这里边的代码应该就是ResNet的代码改的
class SpikingResNet(nn.Module):
    # layer[2,2,2,2] 128*128,4,10,2
    def __init__(self, block, layers, image_size, batch_size, nb_classes=101, channel=20):
        self.inplanes = 64
        super(SpikingResNet, self).__init__()
        self.nb_classes = nb_classes
        # layers是四个block那部分
        self.layers = []
        self.layer_num = layers
        self.size_devide = np.array([4, 4, 4, 4])
        self.planes = [64, 64, 64, 64]
        # --- custom initialization start---
        # 对应第一个7*7卷积
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        # self.drop = nn.Dropout(0.2)
        # 对应第一个平均池化
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 你这四个layer,尺度也没变啊
        self._make_layer(block, 64, layers[0], image_size // 4,
                         batch_size)  # //2 due to stride=2 conv1_custom & avgpool
        self._make_layer(block, 64, layers[1], image_size // 4, batch_size,
                         stride=1)  # //2 due to stride=2 conv1_custom & avgpool
        self._make_layer(block, 64, layers[2], image_size // 4, batch_size,
                         stride=1)  # //4 due to stride=2 conv1_custom & avgpool & layer2
        self._make_layer(block, 64, layers[3], image_size // 4, batch_size,
                         stride=1)  # //8 due to stride=2 conv1_custom & avgpool & layer2 & layer3
        # 这个7*7的是哪来的?论文中3*3啊
        self.avgpool2 = nn.AvgPool2d(7)
        # self.fc_custom = nn.Linear(64 * 4 * 4, nb_classes)                  # no concatenation
        # 对应全连接
        self.fc_custom = nn.Linear(64 * 4 * 4 * 4, nb_classes)  # with concatenation
        # --- custom initialization ---
        # 这里有一些讲解 https://blog.csdn.net/tsq292978891/article/details/79382306
        # 这似乎就是ResNet中的一步实现，权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 1)
                # 7*7*64
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 权重初始化，这个是卷积核参数，m.bias.data是偏置参数
                m.weight.data.normal_(0, np.sqrt(1. / n))
                # if isinstance(m.bias, torch.nn.parameter.Parameter):
                #     m.bias.data.zero_()

            # else:
            #     m.weight.data.normal_(0, math.sqrt(1/10))
        # --- if we needed initialization --- 
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, image_size, batch_size, stride=1):
        # plans=64 blocks=2(subBlock) stride全都是1啊
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))
        # block就是指basicblock
        # block(64,64,[32,32],4,1,none)
        self.layers.append(block(self.inplanes, planes, image_size, batch_size, stride, downsample).cuda())
        # block.expansion=1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # image_size还除4？
            self.layers.append(block(self.inplanes, planes, image_size // stride, batch_size).cuda())

    def forward(self, input):
        # --- init variables and create memory for the SNN
        # input 4*10*2*128*128
        batch_size, time_window, ch, w, h = input.size()
        c_mem = c_spike = torch.zeros(batch_size, 64, w // 2, h // 2, device=device)  # //2 due to stride=2
        c2_spike, c2_mem, c1_spike, c1_mem = [], [], [], []
        # layer_num is [2,2,2,2]; size_devide is [4,4,4,4]
        # 也就是说他有8层的膜电位
        for i in range(len(self.layer_num)):
            d = self.size_devide[i]
            for j in range(self.layer_num[i]):
                # zeros(4,64,32,32)
                c1_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c1_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c2_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c2_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
        fc_sumspike = fc_mem = fc_spike = torch.zeros(batch_size, self.nb_classes, device=device)  # //2 due to stride=2

        # --- main SNN window, time_window is 10，
        for step in range(time_window):
            # x又tm从哪蹦出来的？
            x = input[:, step, :, :, :]
            x = self.conv1_custom(x)
            # 这个c_mem, c_spike是跟着10的时间序列向前更新的
            c_mem, c_spike = mem_update(x, c_mem, c_spike)
            x = self.avgpool1(c_spike)
            # x = self.drop(x)
            # layers就是block的
            # c1_mem is list[8], layers is 8, 因为一共8个subBlock
            for i in range(len(self.layers)):
                # 这里...没问题吗？
                # x是c2_spike,也就是每个SubBlock的输入，同时也是输出
                x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i] = \
                    self.layers[i](x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i])
            # ? c2_spike 8*4*64*32*32 x:4*256*32*32
            # 从0开始，步长是2，直到结束？？？不太对吧
            x = torch.cat(c2_spike[0::2], dim=1)
            x = self.avgpool2(x)
            x = x.view(x.size(0), -1)
            out = self.fc_custom(x)
            fc_mem, fc_spike = mem_update(out, fc_mem, fc_spike)
            fc_sumspike += fc_spike
        fc_sumspike = fc_sumspike / time_window
        return fc_sumspike


def spiking_resnet_18(image_size, batch_size, nb_classes=101, channel=3, **kwargs):
    model = SpikingResNet(SpikingBasicBlock, [2, 2, 2, 2], image_size, batch_size, nb_classes, channel=channel,
                          **kwargs)
    return model
