"""
APFA-Net Wide 版 (通道加宽版, ~2.92M 参数)

基于 APFA-Net 基础版的通道翻倍变体，继承所有核心架构特性。

相比基础版 (1.88M) 的变化:
    - Stage 1 通道: 16/32 (concat=48) -> 32/64 (concat=96), 翻倍
    - Stage 3 通道: 128/256 (concat=384) -> 256/512 (concat=768), 翻倍
    - 特征聚合维度: 2160 -> 2784
    - 总参数: 1.88M -> 2.92M (增加 55%, 但仍仅为 ResNet34+SE 的 1/7)

设计动机:
    基础版 1.88M 在 POLLEN73S (73 类, 2523 张) 上的准确率仅 86.81%,
    远低于原文在 POLLEN23E (23 类) 上的 97%+, 说明基础版的表达能力
    不足以处理 73 类细粒度花粉识别。加宽 Stage 1 和 Stage 3 增加了
    浅层和中高层的特征容量, 提升准确率至 91.56%。

实验结果:
    - APFA-Wide (CE, 两阶段): Acc=91.56%, F1=91.52%, Params=2.92M, Infer=5.09ms
    - APFA-Wide (BFL, 两阶段): Acc=87.34%, F1=85.70%, Params=2.92M
    - 对比 ResNet34+SE (21.35M): Acc=91.29%, F1=89.92%
    - 结论: 仅用 13.7% 的参数量达到了相近甚至略优的准确率

不定性输出:
    当 uncertainty=True 时, forward 额外返回 log_var (对数方差),
    用于 Bayesian Focal Loss 的不定性加权机制。
    log_var 表示模型对每个样本预测的内在不确定性 (aleatoric uncertainty),
    借鉴 Kendall & Gal (NeurIPS 2017) 的异方差不确定性建模。

参考文献:
    [1] Mahmood et al., J. King Saud Univ. Comput. Inform. Sci., 2023 (APFA-Net 原文)
    [2] Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?",
        NeurIPS 2017 (aleatoric uncertainty 理论)
    [3] Khanzhina et al., CVPRW 2023 (Bayesian Focal Loss)
"""

import torch
import torch.nn as nn

# 从基础版导入可复用的构建块（深度可分离卷积、双分支卷积块、SE 注意力）
from .apfanet import SeparableConv2d, ConvBlock, SEBlock


class APFANetWide(nn.Module):
    """
    APFA-Net 通道加宽版

    网络结构 (Wide 版, 2.92M 参数):

    Stage 1 (CG-1, 浅层): ConvBlock(3->32) + ConvBlock(32->64) ->
                          Concat=96 -> SE -> AvgPool(2) -> 112x112x96
    Stage 2 (CG-2, 中层): ConvBlock(96->128) + ConvBlock(128->256) ->
                          Concat=384 + skip(96)=480 -> AvgPool(2) -> 56x56x480
    Stage 3 (CG-3, 中高层): ConvBlock(480->256) + ConvBlock(256->512) ->
                            Concat=768 -> SE -> AvgPool(2) -> 28x28x768
    Stage 4 (CG-4, 高层): ConvBlock(768->512) + ConvBlock(512->1024) ->
                           Concat=1536 + skip(768)=2304 -> 28x28x2304

    特征聚合: t1 (Pool2, Stage2, 480ch) + Stage4 (2304ch) = 2784ch
    分类头: FinalPool(4) -> GAP -> Dropout(0.5) -> FC(2784, num_classes)

    与基础版 (1.88M) 的维度对比:
        层        基础版        Wide版
        CG-1      48ch         96ch      (x2)
        CG-2      240ch        480ch     (x2, 受Stage1影响)
        CG-3      384ch        768ch     (x2)
        CG-4      1920ch       2304ch    (x1.2, 受Stage3影响)
        聚合      2160ch       2784ch    (x1.3)
    """

    def __init__(self, num_classes=73, uncertainty=False):
        """
        参数:
            num_classes (int): 分类类别数，默认 73 (POLLEN73S 类别数)
            uncertainty (bool): 是否启用不定性输出。True 时 forward 返回
                               (logits, log_var)，供 BFL 损失函数使用。
        """
        super().__init__()

        # ===== Stage 1: 浅层特征 (224x224 -> 112x112) =====
        # Wide 版: 32+64=96ch (基础版: 16+32=48ch, 翻倍)
        self.conv1 = ConvBlock(3, 32)               # RGB 3ch -> 32ch
        self.conv2 = ConvBlock(32, 64)              # 32ch -> 64ch
        self.pool1 = nn.AvgPool2d(2)                # 2倍下采样: 224 -> 112
        self.se1 = SEBlock(96, reduction=4)          # SE 通道注意力: 96=32+64

        # ===== Stage 2: 中层特征 (112x112 -> 56x56) =====
        # Wide 版: 128+256=384ch, skip 96 -> 480ch
        self.conv3 = ConvBlock(96, 128)             # 96ch -> 128ch
        self.conv4 = ConvBlock(128, 256)            # 128ch -> 256ch
        self.pool2 = nn.AvgPool2d(2)                # 2倍下采样: 112 -> 56

        # ===== Stage 3: 中高层特征 (56x56 -> 28x28) =====
        # Wide 版: 256+512=768ch (基础版: 128+256=384ch, 翻倍)
        self.conv5 = ConvBlock(480, 256)            # 480ch -> 256ch
        self.conv6 = ConvBlock(256, 512)            # 256ch -> 512ch
        self.pool3 = nn.AvgPool2d(2)                # 2倍下采样: 56 -> 28
        self.se2 = SEBlock(768, reduction=4)         # SE 通道注意力: 768=256+512

        # ===== Stage 4: 高层语义 (28x28, 不下采样) =====
        # Wide 版: 512+1024=1536ch, skip 768 -> 2304ch
        self.conv7 = ConvBlock(768, 512)            # 768ch -> 512ch
        self.conv8 = ConvBlock(512, 1024)           # 512ch -> 1024ch

        # ===== 最终分类层 =====
        # 特征聚合维度: 480 (Stage2浅层) + 2304 (Stage4深层) = 2784ch
        self.final_pool = nn.AvgPool2d(4)            # 4倍池化: 28x28 -> 7x7
        self.gap = nn.AdaptiveAvgPool2d(1)           # 全局平均池化: 7x7 -> 1x1
        self.dropout = nn.Dropout(0.5)               # 50% Dropout 防止过拟合
        self.fc = nn.Linear(2784, num_classes)       # 全连接层: 2784 -> num_classes

        # ===== 不定性输出 (仅 BFL 模式使用) =====
        self.uncertainty = uncertainty
        if uncertainty:
            # log_var_fc: 输出对数方差 log(sigma^2)，每个样本输出一个标量
            # 该方差表示模型对当前样本预测的 aleatoric uncertainty (数据固有噪声)
            # BFL 损失函数利用该方差自适应调整每个样本的 loss 权重
            self.log_var_fc = nn.Linear(2784, 1)

    def forward(self, x):
        """
        前向传播: 四阶段下采样 + APFA 多尺度特征聚合 + 分类

        Stage 1 (CG-1, 浅层特征):
            conv1: 3->32ch, conv2: 32->64ch
            c1 = Concat(32+64)=96ch, SE 通道注意力, Pool(224->112)

        Stage 2 (CG-2, 中层特征):
            conv3: 96->128ch, conv4: 128->256ch
            c2 = Concat(128+256)=384ch
            c1c2 = Concat(SE1_out=96, c2=384)=480ch (skip connection), Pool(112->56)

        Stage 3 (CG-3, 中高层特征):
            conv5: 480->256ch, conv6: 256->512ch
            c3 = Concat(256+512)=768ch, SE 通道注意力, Pool(56->28)

        Stage 4 (CG-4, 高层语义):
            conv7: 768->512ch, conv8: 512->1024ch
            c4 = Concat(512+1024)=1536ch
            c3c4 = Concat(SE2_out=768, c4=1536)=2304ch

        APFA 特征聚合 (核心):
            t1 = Pool2(p2): 480ch, 56x56->28x28 (Stage2 浅层特征降分辨率)
            cnn_feat = Concat(t1=480, c3c4=2304)=2784ch
            融合浅层细节 (Stage2) 与深层语义 (Stage4)

        分类头:
            FinalPool(28->7) -> GAP(7->1) -> Flatten -> Dropout -> FC

        不定性输出 (当 uncertainty=True 时):
            log_var = log_var_fc(features)  -- 每个样本的对数方差

        参数:
            x: 输入图像张量 (B, 3, 224, 224)
        返回:
            若 uncertainty=False: logits (B, num_classes)
            若 uncertainty=True:  (logits, log_var) 元组
        """
        # ===== Stage 1 =====
        x1 = self.conv1(x)                          # (B, 32, 224, 224)
        x2 = self.conv2(x1)                         # (B, 64, 224, 224)
        c1 = torch.cat([x1, x2], dim=1)             # (B, 96, 224, 224)  -- 双分支拼接
        p1 = self.pool1(c1)                         # (B, 96, 112, 112)  -- 2倍池化
        se1 = self.se1(p1)                          # (B, 96, 112, 112)  -- SE 通道注意力

        # ===== Stage 2 =====
        x3 = self.conv3(se1)                        # (B, 128, 112, 112)
        x4 = self.conv4(x3)                         # (B, 256, 112, 112)
        c2 = torch.cat([x3, x4], dim=1)             # (B, 384, 112, 112) -- 双分支拼接
        c1c2 = torch.cat([se1, c2], dim=1)          # (B, 480, 112, 112) -- skip: se1+c2
        p2 = self.pool2(c1c2)                       # (B, 480, 56, 56)   -- 2倍池化

        # ===== Stage 3 =====
        x5 = self.conv5(p2)                         # (B, 256, 56, 56)
        x6 = self.conv6(x5)                         # (B, 512, 56, 56)
        c3 = torch.cat([x5, x6], dim=1)             # (B, 768, 56, 56)   -- 双分支拼接
        p3 = self.pool3(c3)                         # (B, 768, 28, 28)   -- 2倍池化
        se2 = self.se2(p3)                          # (B, 768, 28, 28)   -- SE 通道注意力

        # ===== Stage 4 =====
        x7 = self.conv7(se2)                        # (B, 512, 28, 28)
        x8 = self.conv8(x7)                         # (B, 1024, 28, 28)
        c4 = torch.cat([x7, x8], dim=1)             # (B, 1536, 28, 28)  -- 双分支拼接
        c3c4 = torch.cat([se2, c4], dim=1)          # (B, 2304, 28, 28)  -- skip: se2+c4

        # ===== APFA 核心: 多尺度特征聚合 =====
        # t1: Stage2 浅层特征 (480ch) 池化对齐到 28x28
        t1 = self.pool2(p2)                         # (B, 480, 28, 28)   -- Stage2 降分辨率
        # cnn_feat: 浅层细节 + 深层语义 = 480+2304=2784ch
        cnn_feat = torch.cat([t1, c3c4], dim=1)     # (B, 2784, 28, 28)  -- 特征融合

        # ===== 分类头 =====
        out = self.final_pool(cnn_feat)              # (B, 2784, 7, 7)    -- 4倍池化
        out = self.gap(out)                          # (B, 2784, 1, 1)    -- 全局平均池化
        out = out.view(out.size(0), -1)              # (B, 2784)           -- 展平
        features = out                               # 保存特征向量（用于不定性分支）
        out = self.dropout(out)                      # (B, 2784)           -- Dropout 正则化
        logits = self.fc(out)                        # (B, num_classes)    -- 全连接分类

        # 不定性输出分支 (BFL 模式): 从同一特征向量预测对数方差
        if self.uncertainty:
            # log_var: 对数方差 log(sigma^2), 形状 (B, 1)
            log_var = self.log_var_fc(self.dropout(features))
            return logits, log_var                   # 返回元组
        return logits                                 # 仅返回分类结果


# ===== 自检代码 =====
if __name__ == "__main__":
    # 创建模型并测试前向传播
    m = APFANetWide(num_classes=73)
    x = torch.randn(2, 3, 224, 224)                 # 模拟 2 张 RGB 224x224 图像
    y = m(x)                                         # 前向推理
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"APFA-Net Wide self-check:")
    print(f"  Input:  {x.shape}")                    # (2, 3, 224, 224)
    print(f"  Output: {y.shape}")                    # (2, 73)
    print(f"  Params: {params:.2f}M")                # ~2.92M
