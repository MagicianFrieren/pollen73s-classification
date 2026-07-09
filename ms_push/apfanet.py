"""
APFA-Net 基础版 (Attention-guided Pollen Features Aggregation Network)

论文来源:
    Mahmood T, Choi J, Park K R. "Artificial intelligence-based classification of pollen grains
    using attention-guided pollen features aggregation network."
    Journal of King Saud University - Computer and Information Sciences, 2023, 35: 740-756.
    DOI: https://doi.org/10.1016/j.jksuci.2023.01.013

核心架构特点 (与标准 CNN 的区别):
    1. 深度可分离卷积 (Depthwise Separable Convolution): 将标准卷积拆分为逐通道卷积 +
       逐点 1x1 卷积，参数量从 K^2*C_in*C_out 降至 K^2*C_in + C_in*C_out，
       降低约 8~9 倍，实现 1.88M 极轻量参数量。
    2. 双分支卷积块 (ConvBlock): 并行融合 3x3 和 5x5 双分支卷积输出，
       通过 Add 融合多尺度感受野信息（局部细节 + 稍大范围纹理）。
    3. 密集拼接 + Skip Connection: 每 Stage 内 ConvBlock 的输出进行 Concat 拼接
       (而非 Add)，Stage 间通过 skip connection 保留浅层特征。
       Stage 1 和 Stage 3 末尾引入 SE 通道注意力模块。
    4. 多尺度特征聚合 (APFA 核心): 将 Stage 2 的中间特征（240ch）通过池化对齐后
       与 Stage 4 的高层语义特征（1920ch）拼接，总计 2160 维特征向量。

网络结构 (原版 1.88M 参数):
    Stage 1 (CG-1): ConvBlock(3->16) + ConvBlock(16->32) -> Concat=48 -> SE -> Pool -> 112x112x48
    Stage 2 (CG-2): ConvBlock(48->64) + ConvBlock(64->128) -> Concat=192 + skip(48)=240 -> Pool -> 56x56x240
    Stage 3 (CG-3): ConvBlock(240->128) + ConvBlock(128->256) -> Concat=384 -> SE -> Pool -> 28x28x384
    Stage 4 (CG-4): ConvBlock(384->512) + ConvBlock(512->1024) -> Concat=1536 + skip(384)=1920
    特征聚合: Stage2 浅层(240ch) + Stage4 深层(1920ch) = 2160ch
    分类头: FinalPool(7x7) -> GAP -> Dropout(0.5) -> FC(2160, num_classes)

此文件为部署版 (ms_push/)，与训练版 (src/models/apfanet.py) 完全相同，
仅导入路径不同（部署版直接本地导入，无需 src.models 包结构）。

Wide 版本见 apfanet_wide.py，通道数翻倍至 2.92M 参数。
"""

# ===== 框架导入 =====
import torch                                      # PyTorch 核心库
import torch.nn as nn                             # 神经网络模块
import torch.nn.functional as F                   # 函数式 API（激活函数等）


class SeparableConv2d(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Convolution)

    将标准卷积分解为两步，极大减少参数量和计算量:
        Step 1 - Depthwise (逐通道): 每个输入通道独立进行空间卷积 (groups=in_ch)。
                 仅提取各通道的空间模式，不混合通道信息。
        Step 2 - Pointwise (逐点): 用 1x1 卷积混合通道信息，将 in_ch 映射到 out_ch。

    参数量对比 (以 3x3 卷积, C_in=64, C_out=128 为例):
        标准卷积: 3*3*64*128 = 73,728
        可分离卷积: 3*3*64 + 64*128 = 576 + 8,192 = 8,768
        压缩比: 73,728 / 8,768 ≈ 8.4 倍

    这是 APFA-Net 实现极低参数量的核心技术，等价于 TensorFlow 的 SeparableConv2D。
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding):
        """
        参数:
            in_ch (int):  输入通道数
            out_ch (int): 输出通道数
            kernel_size (int): 卷积核尺寸（正方形，如 3 表示 3x3）
            padding (int): 边缘填充量（保持空间尺寸不变时设为 kernel_size//2）
        """
        super().__init__()
        # Depthwise: groups=in_ch 使每个通道独立卷积
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size,
                                   padding=padding, groups=in_ch, bias=False)
        # Pointwise: 1x1 卷积用于混合和变换通道
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        """
        前向传播: depthwise -> pointwise

        参数:
            x: 输入张量 (B, in_ch, H, W)
        返回:
            输出张量 (B, out_ch, H, W)
        """
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """
    双分支可分离卷积块

    设计思路:
        并行的 3x3 和 5x5 深度可分离卷积分支分别捕获不同尺度的特征:
        - 3x3 分支: 关注局部细节（纹理、边缘走向）
        - 5x5 分支: 关注稍大范围的空间模式（轮廓、区域纹理）
        两分支输出通过 Add 融合，再经过 BatchNorm 归一化稳定训练。

    核心优势: 多尺度感受野融合带来更丰富的特征表达，
    原文消融实验 (Table 5) 显示比单一 3x3 卷积提升约 4% 准确率。

    等价于 TensorFlow 版本中的 conv_block 函数。
    """

    def __init__(self, in_ch, out_ch):
        """
        参数:
            in_ch (int):  输入通道数
            out_ch (int): 输出通道数（两分支输出通道数相同，以便 Add 操作）
        """
        super().__init__()
        # 3x3 分支: 小感受野，关注局部细节
        self.branch_3x3 = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 5x5 分支: 大感受野，关注区域纹理
        self.branch_5x5 = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        # BatchNorm 在 Add 融合后进行特征归一化
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """
        前向传播: 双分支并行 -> Add 融合 -> BN

        参数:
            x: 输入张量 (B, in_ch, H, W)
        返回:
            融合后的特征图 (B, out_ch, H, W)
        """
        x1 = self.branch_3x3(x)                    # 3x3 深度可分离卷积分支
        x2 = self.branch_5x5(x)                    # 5x5 深度可分离卷积分支
        return self.bn(x1 + x2)                     # Add 融合 + 批归一化


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    源自 SENet (Hu et al., CVPR 2018) 的通道注意力机制，用于自动学习各通道的重要性权重。

    三步流程:
        Step 1 - Squeeze (压缩): 对每个通道做全局平均池化 (HxW -> 1x1)，
                 将空间信息压缩为单个标量，获得通道级全局描述。
        Step 2 - Excitation (激励): 通过两层全连接网络（先降维再升维）学习
                 通道间依赖关系，经 Sigmoid 输出 [0,1] 范围的注意力权重。
        Step 3 - Scale (重标定): 将学习到的权重逐通道乘以原始特征图，
                 实现对重要通道的增强和不重要通道的抑制。

    reduction=4: 中间层维度 = C/4，原文推荐值。
    这个值在参数效率和信息保留之间取得平衡。

    原文效果 (Table 5): 加入 SE 模块后准确率从 93.14% 提升到 97.21%，
    验证了通道注意力在花粉识别任务中的有效性。
    """

    def __init__(self, channels, reduction=4):
        """
        参数:
            channels (int): 输入/输出通道数
            reduction (int): 降维比率，中间维度 = channels // reduction (默认 4)
        """
        super().__init__()
        reduced = channels // reduction             # 中间层通道数
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # Squeeze: 全局平均池化 (B,C,H,W) -> (B,C,1,1)
            nn.Flatten(),                            # 展平为 (B, C)
            nn.Linear(channels, reduced),            # 降维: C -> C/4
            nn.ReLU(inplace=True),                   # 非线性激活
            nn.Linear(reduced, channels),            # 升维: C/4 -> C
            nn.Sigmoid(),                            # 归一化到 [0,1]
        )

    def forward(self, x):
        """
        前向传播: Squeeze -> Excitation -> Scale

        参数:
            x: 输入特征图 (B, channels, H, W)
        返回:
            经注意力加权后的特征图 (B, channels, H, W)
        """
        # 计算通道注意力权重: (B, C) -> (B, C, 1, 1)
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale                            # 逐通道加权


class ProposedModel(nn.Module):
    """
    APFA-Net 完整模型 (Attention-Guided Pollen Features Aggregation Network)

    这是部署版，对应论文中提出的原始架构。四阶段下采样 + 多尺度特征聚合。

    完整数据流:

    [输入] RGB 图像 (B, 3, 224, 224)
        |
    Stage 1 (CG-1, 浅层特征):
        conv1: 3->16ch | conv2: 16->32ch
        c1 = Concat(16+32) = 48ch
        p1 = AvgPool(2): 224x224 -> 112x112
        se1 = SE(48ch): 通道注意力加权
        |
    Stage 2 (CG-2, 中层特征):
        conv3: 48->64ch | conv4: 64->128ch
        c2 = Concat(64+128) = 192ch
        c1c2 = Concat(se1=48, c2=192) = 240ch  (skip connection)
        p2 = AvgPool(2): 112x112 -> 56x56
        |
    Stage 3 (CG-3, 中高层特征):
        conv5: 240->128ch | conv6: 128->256ch
        c3 = Concat(128+256) = 384ch
        p3 = AvgPool(2): 56x56 -> 28x28
        se2 = SE(384ch): 通道注意力加权
        |
    Stage 4 (CG-4, 高层语义特征):
        conv7: 384->512ch | conv8: 512->1024ch
        c4 = Concat(512+1024) = 1536ch
        c3c4 = Concat(se2=384, c4=1536) = 1920ch  (skip connection)

    特征聚合 (APFA 核心):
        t1 = Pool2(p2): 将 Stage2 输出从 56x56 池化至 28x28，保持 240ch
        cnn_feat = Concat(t1=240, c3c4=1920) = 2160ch
        将浅层细节特征与深层语义特征融合

    分类头:
        FinalPool(4): 28x28 -> 7x7
        GAP: 7x7 -> 1x1  (全局平均池化)
        Flatten -> Dropout(0.5) -> FC(2160, num_classes)

    APFA 核心思想: 浅层特征提供空间细节（纹理、边缘），深层特征提供语义信息
    （形状、类别判别性特征），跨层拼接实现了多尺度信息的高效整合。

    参数:
        num_classes (int): 分类类别数（默认为 23，对应原始 AIpollen 数据集）
        input_size (int): 输入图像尺寸（默认 224）
    """

    def __init__(self, num_classes=23, input_size=224):
        super().__init__()
        self.num_classes = num_classes

        # ===== Stage 1: 浅层特征提取 (224x224 -> 112x112) =====
        # conv1: 3ch -> 16ch, conv2: 16ch -> 32ch
        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 32)
        self.pool1 = nn.AvgPool2d(2)                # 2倍下采样
        self.se1 = SEBlock(48, reduction=4)          # 48 = 16 + 32 (Concat)

        # ===== Stage 2: 中层特征提取 (112x112 -> 56x56) =====
        # conv3: 48ch -> 64ch, conv4: 64ch -> 128ch
        self.conv3 = ConvBlock(48, 64)
        self.conv4 = ConvBlock(64, 128)
        self.pool2 = nn.AvgPool2d(2)                # 2倍下采样

        # ===== Stage 3: 中高层特征提取 (56x56 -> 28x28) =====
        # conv5: 240ch -> 128ch, conv6: 128ch -> 256ch
        self.conv5 = ConvBlock(240, 128)
        self.conv6 = ConvBlock(128, 256)
        self.pool3 = nn.AvgPool2d(2)                # 2倍下采样
        self.se2 = SEBlock(384, reduction=4)         # 384 = 128 + 256 (Concat)

        # ===== Stage 4: 高层语义提取 (28x28, 不下采样) =====
        # conv7: 384ch -> 512ch, conv8: 512ch -> 1024ch
        self.conv7 = ConvBlock(384, 512)
        self.conv8 = ConvBlock(512, 1024)

        # ===== 最终分类层 =====
        # FinalPool(4): 28x28 -> 7x7  (4倍池化)
        self.final_pool = nn.AvgPool2d(4)
        self.gap = nn.AdaptiveAvgPool2d(1)           # 全局平均池化: 7x7 -> 1x1
        self.dropout = nn.Dropout(0.5)               # 50% Dropout 防止过拟合
        self.fc = nn.Linear(2160, num_classes)       # 2160 维 -> num_classes

    def _concat(self, a, b):
        """
        安全拼接函数: 自动对齐空间尺寸后做通道拼接

        用于 skip connection 场景，确保 Stage 间特征图尺寸一致。
        常规情况下 a 和 b 尺寸相同，直接 cat。
        若尺寸不一致（如池化层参数变化导致的边界情况），使用最近邻插值对齐。

        参数:
            a, b: 两个待拼接的张量 (B, C_a, H, W), (B, C_b, H, W)
        返回:
            拼接后的张量 (B, C_a+C_b, H, W)
        """
        if a.shape[2:] != b.shape[2:]:
            b = F.interpolate(b, size=a.shape[2:], mode='nearest')
        return torch.cat([a, b], dim=1)

    def forward(self, x):
        """
        前向传播: 完整的四阶段 + 特征聚合 + 分类

        数据流详情见类文档字符串。

        参数:
            x: 输入图像张量 (B, 3, 224, 224)
        返回:
            分类 logits (B, num_classes)
        """
        # Stage 1: 浅层特征 (224x224 -> 112x112)
        x1 = self.conv1(x)                          # (B, 16, 224, 224)
        x2 = self.conv2(x1)                         # (B, 32, 224, 224)
        c1 = torch.cat([x1, x2], dim=1)             # (B, 48, 224, 224) -- 双分支拼接
        p1 = self.pool1(c1)                         # (B, 48, 112, 112) -- 2倍池化
        se1 = self.se1(p1)                          # (B, 48, 112, 112) -- SE 通道注意力

        # Stage 2: 中层特征 (112x112 -> 56x56)
        x3 = self.conv3(se1)                        # (B, 64, 112, 112)
        x4 = self.conv4(x3)                         # (B, 128, 112, 112)
        c2 = torch.cat([x3, x4], dim=1)             # (B, 192, 112, 112) -- 双分支拼接
        c1c2 = torch.cat([se1, c2], dim=1)          # (B, 240, 112, 112) -- skip: se1+c2
        p2 = self.pool2(c1c2)                       # (B, 240, 56, 56)   -- 2倍池化

        # Stage 3: 中高层特征 (56x56 -> 28x28)
        x5 = self.conv5(p2)                         # (B, 128, 56, 56)
        x6 = self.conv6(x5)                         # (B, 256, 56, 56)
        c3 = torch.cat([x5, x6], dim=1)             # (B, 384, 56, 56)   -- 双分支拼接
        p3 = self.pool3(c3)                         # (B, 384, 28, 28)   -- 2倍池化
        se2 = self.se2(p3)                          # (B, 384, 28, 28)   -- SE 通道注意力

        # Stage 4: 高层语义 (28x28, 保持分辨率)
        x7 = self.conv7(se2)                        # (B, 512, 28, 28)
        x8 = self.conv8(x7)                         # (B, 1024, 28, 28)
        c4 = torch.cat([x7, x8], dim=1)             # (B, 1536, 28, 28)  -- 双分支拼接
        c3c4 = torch.cat([se2, c4], dim=1)          # (B, 1920, 28, 28)  -- skip: se2+c4

        # ===== APFA 核心: 多尺度特征聚合 =====
        # t1: Stage2 浅层特征 (240ch, 56x56 -> 28x28)
        t1 = self.pool2(p2)                         # (B, 240, 28, 28)   -- Stage2 降分辨率
        # cnn_feat: 浅层细节 + 深层语义 = 240+1920=2160ch
        cnn_feat = torch.cat([t1, c3c4], dim=1)     # (B, 2160, 28, 28)  -- 特征融合

        # ===== 分类头 =====
        out = self.final_pool(cnn_feat)              # (B, 2160, 7, 7)    -- 4倍池化
        out = self.gap(out)                          # (B, 2160, 1, 1)    -- 全局平均池化
        out = out.view(out.size(0), -1)              # (B, 2160)           -- 展平
        out = self.dropout(out)                      # (B, 2160)           -- Dropout 正则化
        out = self.fc(out)                           # (B, num_classes)    -- 全连接分类
        return out


def proposed_model(num_classes=23):
    """
    工厂函数: 创建 APFA-Net 模型实例

    兼容原 TensorFlow 版本的调用约定。

    参数:
        num_classes (int): 分类类别数
    返回:
        ProposedModel 实例
    """
    return ProposedModel(num_classes=num_classes)


# ===== 自检代码 =====
if __name__ == '__main__':
    # 创建模型并测试前向传播
    m = ProposedModel(num_classes=23)
    x = torch.randn(2, 3, 224, 224)                 # 模拟 2 张 RGB 224x224 图像
    y = m(x)                                         # 前向推理
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"APFA-Net (original) self-check:")
    print(f'Input:  {x.shape}')                      # (2, 3, 224, 224)
    print(f'Output: {y.shape}')                      # (2, 23)
    print(f'Params: {params:.2f}M')                  # ~1.88M
