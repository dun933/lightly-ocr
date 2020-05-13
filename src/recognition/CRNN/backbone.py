import torch.nn as nn

def activation_fn(act, inplace=True):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['leaky', nn.LeakyReLU(negative_slope=0.01, inplace=inplace)],
        ['selu', nn.SELU(inplace=inplace)],
        ['none', nn.Identity()]
    ])[act]

class conv_bn(nn.Module):
    def __init__(self, nIn, nOut, kernel_size=3, stride=1, padding=1, bias=False, bn=True, activation='leaky'):
        super(conv_bn, self).__init__()
        assert activation is not None, f'activation can be set to str "none", got {type(activation)} instead'
        self.bn = bn
        self.activation = activation
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = activation_fn(activation)

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.bn:
            out = self.bn(out)
        if self.activation:
            out = self.act(out)
        return out

class residual_block(nn.Module):
    expansion = 1

    def __init__(self, nIn, nOut, stride=1, downsample=None, activation='leaky'):
        super(residual_block, self).__init__()
        self.conv_1 = conv_bn(nIn, nOut, stride=stride, activation=activation)
        self.conv_2 = conv_bn(nOut, nOut, stride=stride, activation='none')
        self.stride = stride
        self.downsample = downsample
        self.act = activation

    def forward(self, inputs):
        residual = inputs
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        out += residual
        out = activation_fn(self.act)(out)
        return out

    # return layers

class _resnet(nn.Module):
    def __init__(self, nIn, nOut, block_fn, layers):
        super(_resnet, self).__init__()

        self.inplanes, self.out_blocks = int(nOut / 8), [int(nOut / 4), int(nOut / 2), nOut, nOut]

        self.conv0_1 = conv_bn(nIn, int(nOut / 16))
        self.conv0_2 = conv_bn(int(nOut / 16), self.inplanes)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = self.stack_residual_block(block_fn, self.out_blocks[0], layers[0])
        self.conv1 = conv_bn(self.out_blocks[0], self.out_blocks[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = self.stack_residual_block(block_fn, self.out_blocks[1], layers[1])
        self.conv2 = conv_bn(self.out_blocks[1], self.out_blocks[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.block3 = self.stack_residual_block(block_fn, self.out_blocks[2], layers[2])
        self.conv3 = conv_bn(self.out_blocks[2], self.out_blocks[2])

        self.block4 = self.stack_residual_block(block_fn, self.out_blocks[3], layers[3], stride=1)
        self.conv4_1 = conv_bn(self.out_blocks[3], self.out_blocks[3], kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.conv4_2 = conv_bn(self.out_blocks[3], self.out_blocks[3], kernel_size=2, stride=1, padding=0)

    def stack_residual_block(self, block_fn, nOut, num_layers, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != nOut * block_fn.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, nOut*block_fn.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(nOut*block_fn.expansion),
                    )
        layers = []
        layers.append(block_fn(self.inplanes, nOut, stride, downsample))
        self.inplanes = nOut * block_fn.expansion
        for i in range(1, num_layers):
            layers.append(block_fn(self.inplanes, nOut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.conv0_2(x)

        x = self.maxpool1(x)
        x = self.block1(x)
        x = self.conv1(x)

        x = self.maxpool2(x)
        x = self.block2(x)
        x = self.conv2(x)

        x = self.maxpool3(x)
        x = self.block3(x)
        x = self.conv3(x)

        x = self.block4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, nIn, nOut=512, net='resnet'):
        super(ResNet, self).__init__()
        self.convnet = _resnet(nIn, nOut, residual_block, [1, 2, 5, 3])
        # TODO: added support for RCNN and VGG

    def forward(self, inputs):
        return self.convnet(inputs)
