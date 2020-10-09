import torch
from EfficientNet.EfficientNet import EfficientNet,MBConvBlock
from torch import nn
from torchvision import transforms
from EfficientNet.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

class LYCNet(nn.Module):
    def __init__(self,block_args=None,global_params = None):
        super().__init__()
        self._block_args = block_args
        self._global_params = global_params

        #BN的两个参数
        self.bn_mom = 1 - self._global_params.batch_norm_momentum
        self.bn_eps = self._global_params.batch_norm_epsilon

        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.scales = [[16,112],[24,56],[40,28],[112,14],[1280,7]]

        in_channels = 3
        out_channels = round_filters(32,self._global_params)
        self._conv_stem = Conv2d(in_channels,out_channels,kernel_size=3,stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels,eps=self.bn_eps,momentum=self.bn_mom)
        #计算输出图像大小 使用Conv2dSamePadding with a stride 2
        image_size = calculate_output_image_size(image_size,2)

        #Build blocks
        self._blocks = nn.ModuleList([])
        #第一个for循环是多少个BLOCK，第二个num_repeat是指一个block中是否重复添加MBConv
        for block_args in self._block_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args,self._global_params,image_size=image_size))
            image_size = calculate_output_image_size(image_size,block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters = block_args.output_filters, stride =1)
            for _ in range(block_args.num_repeat -1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

        # head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels,momentum=self.bn_mom,eps=self.bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
        naive_downsample_layers = self.add_adaptive_layers(self.scales)
        self.downsample_layers1 = naive_downsample_layers['reduction_1']
        self.downsample_layers2 = naive_downsample_layers['reduction_2']
        self.downsample_layers3 = naive_downsample_layers['reduction_3']
        self.downsample_layers4 = naive_downsample_layers['reduction_4']



    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    #这里是找到每次block后的那个特征，比如endponits['reduction_1']就是第一个block后提取到的特征
    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints
    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def add_adaptive_layers(self,scales):
        """
        主要用于增加第一层到第四层的适应层，该卷积核会将不同大小的feature map转换到
        同样大小[1280,7,7]的大小
        :param scales: [[16,112],[24,56],[40,28],[112,14],[1280,7]]
        :return: {'reduction_1':nn.ModuleList(),'reduction_2':nn.ModuleList()}
        """
        new_scales = list()
        for i in range(len(scales)-1):
            y = [x for x in scales[i:]]
            new_scales.append(y)
        blocks = {}
        blocks['reduction_1'] = NaiveDownSampleBlock(new_scales[0],global_params=self._global_params)
        blocks['reduction_2'] = NaiveDownSampleBlock(new_scales[1],global_params=self._global_params)
        blocks['reduction_3'] = NaiveDownSampleBlock(new_scales[2], global_params=self._global_params)
        blocks['reduction_4'] = NaiveDownSampleBlock(new_scales[3], global_params=self._global_params)
        return blocks

    """
    >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
    >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
    >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
    >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
    >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
    """
    def forward(self,inputs):
        endpoints = self.extract_endpoints(inputs)
        feature1,output1 = self.downsample_layers1(endpoints['reduction_1'])
        feature2,output2 = self.downsample_layers2(endpoints['reduction_2'])
        feature3,output3 = self.downsample_layers3(endpoints['reduction_3'])
        feature4,output4 = self.downsample_layers4(endpoints['reduction_4'])
        feature5 = endpoints['reduction_5']
        x = self._avg_pooling(feature5)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        output5 = self._fc(x)
        features = [feature1,feature2,feature3,feature4,feature5]
        outputs = [output1,output2,output3,output4,output5]
        return features,outputs



class NaiveDownSampleBlock(nn.Module):
    """
    要用于增加第一层到第四层的适应层，该卷积核会将不同大小的feature map转换到
        同样大小[1280,7,7]的大小
        :param scales: [[16,112],[24,56],[40,28],[112,14],[1280,7]]
        :return: {'reduction_1':nn.ModuleList(),'reduction_2':nn.ModuleList()}
    """
    def __init__(self,scales,global_params):
        super().__init__()
        self._global_params = global_params
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        self.block = nn.ModuleList()
        for i,scale in enumerate(scales):
            if i == len(scales)-1:
                break
            Conv2d = get_same_padding_conv2d(scale[1])
            depthwise_conv = Conv2d(in_channels=scale[0], out_channels=scale[0], groups=scale[0],
                                    kernel_size=3, stride=2, bias=False)
            bn1 = nn.BatchNorm2d(num_features=scale[0], momentum=bn_mom,
                                 eps=bn_eps)
            swish = MemoryEfficientSwish()
            project_conv = Conv2d(in_channels=scale[0], out_channels=scales[i+1][0], kernel_size=1, bias=False)
            bn2 = nn.BatchNorm2d(num_features=scales[i+1][0], momentum=bn_mom,
                                 eps=bn_eps)
            self.block.append(depthwise_conv)
            self.block.append(bn1)
            self.block.append(swish)
            self.block.append(project_conv)
            self.block.append(bn2)
        self.bn3 = nn.BatchNorm2d(num_features=scales[-1][0],momentum=bn_mom,eps=bn_eps)
        self.swish = MemoryEfficientSwish()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self._global_params.dropout_rate)
        self.fc = nn.Linear(1280,self._global_params.num_classes)

    def forward(self,inputs):
        x = inputs
        for idx,layer in enumerate(self.block):
            x = layer(x)
        feature_map = self.swish(self.bn3(x))
        x = self.avg_pooling(feature_map)
        x = x.flatten(start_dim = 1)
        x = self.dropout(x)
        x = self.fc(x)
        return feature_map,x
