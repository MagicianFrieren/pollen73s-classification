"""
ResNet34 基线 & ResNet34 + SE 注意力模型

包含两个模型变体:
    1. ResNet34 (标准版): ImageNet 预训练的 ResNet34，替换最后的全连接层。
    2. ResNet34 + SE: 在每个残差阶段后插入 Squeeze-and-Excitation 通道注意力模块。

模型对比 (POLLEN73S 测试集):
    - ResNet34 + SE: Acc=91.29%, F1=89.92%, Params=21.35M, Infer=6.88ms
    - ResNet34: 未记录（权重存在但缺 results.json）

设计选择:
    - 使用 torchvision 内置的 ResNet34 作为骨干网络
    - SE 模块插入在 layer1-4 之后，在 avgpool 之前
    - reduction=16 (标准 SE 配置，中间维度 = C/16)
    - 使用 ImageNet 预训练权重加速收敛

SE 模块原理 (Hu et al., CVPR 2018):
    Squeeze: 全局平均池化将每个通道的空间信息压缩为一个标量
    Excitation: 两层 FC + Sigmoid 生成 [0,1] 范围的通道注意力权重
    Scale: 权重逐通道乘以原始特征，增强重要通道，抑制次要通道
"""

# ===== 框架导入 =====
import torch                                      # PyTorch 核心
import torch.nn as nn                             # 神经网络模块
import torchvision.models as models               # torchvision 预训练模型库


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    源自 SENet 论文 (Hu et al., CVPR 2018), 用于自动学习通道级特征重要性权重。

    工作流程:
        1. AdaptiveAvgPool2d(1): 压缩空间维度 (B,C,H,W) -> (B,C,1,1)
        2. FC(C -> C/16) + ReLU: 降维，学习通道间依赖
        3. FC(C/16 -> C) + Sigmoid: 升维，输出 [0,1] 注意力权重
        4. 权重逐通道乘以原始特征: 选择性增强/抑制

    reduction=16: ResNet 标准配置。与 APFA-Net 的 reduction=4 不同，
    ResNet 通道数更大(最高 512ch)，更大的缩减比可保持参数效率。

    参数:
        channels (int): 输入/输出通道数
        reduction (int): 降维比率，中间维度 = C/reduction (默认 16)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = channels // reduction             # 中间层通道数
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # Squeeze: 全局平均池化 (B,C,H,W)->(B,C,1,1)
            nn.Flatten(),                            # 展平为 (B, C)
            nn.Linear(channels, reduced),            # 降维: C -> C/16
            nn.ReLU(inplace=True),                   # 非线性激活
            nn.Linear(reduced, channels),            # 升维: C/16 -> C
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
        # 逐通道加权: 重要通道的 scale 接近 1（保留），不重要通道接近 0（抑制）
        return x * scale


class ResNet34SE(nn.Module):
    """
    ResNet34 + Squeeze-and-Excitation

    在 ResNet34 的四个残差阶段后各插入一个 SE 模块:
        - layer1 -> SE(64):   浅层特征通道注意力
        - layer2 -> SE(128):  中层特征通道注意力
        - layer3 -> SE(256):  中高层特征通道注意力
        - layer4 -> SE(512):  深层语义通道注意力

    结构:
        conv1 (7x7, stride2) -> bn1 -> relu -> maxpool (stride2)
        -> layer1 (3 blocks) -> SE1(64)
        -> layer2 (4 blocks) -> SE2(128)
        -> layer3 (6 blocks) -> SE3(256)
        -> layer4 (3 blocks) -> SE4(512)
        -> avgpool -> flatten -> FC(512, num_classes)

    总参数量: ~21.35M (ResNet34 ~21.28M + SE 模块 ~0.07M)

    预训练策略: 使用 ImageNet1K_V1 权重初始化骨干网络,
    SE 模块和分类头随机初始化。

    参数:
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用 ImageNet 预训练权重 (默认 True)
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 加载预训练 ResNet34 骨干网络
        backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # 解构 backbone，以便在各阶段后插入 SE 模块
        self.conv1 = backbone.conv1                  # 7x7 卷积, stride=2
        self.bn1 = backbone.bn1                      # BatchNorm
        self.relu = backbone.relu                    # ReLU 激活
        self.maxpool = backbone.maxpool              # 3x3 MaxPool, stride=2

        # 四个残差阶段 (layer1-4)
        self.layer1 = backbone.layer1                # 3 个 BasicBlock, 64ch
        self.layer2 = backbone.layer2                # 4 个 BasicBlock, 128ch
        self.layer3 = backbone.layer3                # 6 个 BasicBlock, 256ch
        self.layer4 = backbone.layer4                # 3 个 BasicBlock, 512ch

        # 四个 SE 注意力模块，分别插入各阶段之后
        self.se1 = SEBlock(64)                       # layer1 后, 64ch
        self.se2 = SEBlock(128)                      # layer2 后, 128ch
        self.se3 = SEBlock(256)                      # layer3 后, 256ch
        self.se4 = SEBlock(512)                      # layer4 后, 512ch

        # 分类头
        self.avgpool = backbone.avgpool              # AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)  # 512 -> num_classes

    def forward(self, x):
        """
        ResNet34 前向传播 (含 SE 通道注意力)

        参数:
            x: 输入图像张量 (B, 3, 224, 224)
        返回:
            分类 logits (B, num_classes)
        """
        # 初始卷积: 7x7 + BN + ReLU + MaxPool
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 四个残差阶段，每阶段后接 SE 通道注意力
        x = self.layer1(x)
        x = self.se1(x)                             # 64ch 通道注意力
        x = self.layer2(x)
        x = self.se2(x)                             # 128ch 通道注意力
        x = self.layer3(x)
        x = self.se3(x)                             # 256ch 通道注意力
        x = self.layer4(x)
        x = self.se4(x)                             # 512ch 通道注意力

        # 分类头
        x = self.avgpool(x)                          # (B, 512, 1, 1)
        x = torch.flatten(x, 1)                      # (B, 512)
        x = self.fc(x)                               # (B, num_classes)
        return x


def resnet34(num_classes, pretrained=True):
    """
    创建标准 ResNet34 模型（无 SE），替换最后的全连接层

    直接用 torchvision 的 resnet34，仅修改分类头适配花粉类别数。

    参数:
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用 ImageNet 预训练权重 (默认 True)
    返回:
        torchvision ResNet34 模型实例
    """
    model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换分类头
    return model


def get_model(model_name, num_classes):
    """
    模型工厂函数: 根据名称返回对应的模型实例

    参数:
        model_name (str): 模型名称 ("resnet34" 或 "resnet34_se")
        num_classes (int): 分类类别数
    返回:
        nn.Module 模型实例
    抛出:
        ValueError: 当 model_name 不合法时
    """
    if model_name == "resnet34":
        return resnet34(num_classes, pretrained=True)
    elif model_name == "resnet34_se":
        return ResNet34SE(num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===== 自检代码 =====
if __name__ == "__main__":
    for name in ["resnet34", "resnet34_se"]:
        m = get_model(name, 36)                     # 36 类 (AIpollen)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        x = torch.randn(2, 3, 224, 224)             # 模拟 2 张 RGB 224x224 图像
        y = m(x)                                     # 前向推理
        print(f"{name}: params={params:.2f}M, input={list(x.shape)}, output={list(y.shape)}")
