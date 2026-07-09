# 花粉识别 — APFA-Net & 网站部署

> 本科课程设计 小组项目  
> 本人负责部分：APFA-Net 模型训练 + 网站部署

---


## 致谢

由衷感谢 **Mahmood T.** 博士（APFA-Net 原作者），他在本项目过程中耐心解答了模型结构、两阶段训练策略等多个关键技术问题，给予了非常大的帮助。

> Mahmood T, Choi J, Park K R. *Artificial intelligence-based classification of pollen grains using attention-guided pollen features aggregation network.* Journal of King Saud University - Computer and Information Sciences, 2023, 35: 740-756.

---

## 目录结构

```
上交/
├── src/                          # 模型定义 + 训练脚本
│   ├── models/
│   │   ├── apfanet.py            # APFA-Net 基础版 (1.88M)
│   │   └── apfanet_wide.py       # APFA-Net Wide 版 (2.92M)
│   ├── train_apfanet.py          # APFA-Net Wide 两阶段训练入口
│   └── prepare_data.py           # 数据加载 / 预处理 / 划分
│
├── ms_push/                      # 网站部署 (Gradio + ModelScope)
│   ├── app.py                    # 网站主程序
│   ├── apfanet.py                # 部署版 APFA 基础模型
│   ├── apfanet_wide.py           # 部署版 APFA Wide 模型
│   ├── resnet.py                 # 部署版 ResNet34+SE 模型
│   ├── class_names_73.json       # 73 类花粉名称映射
│   ├── pollen_info.json          # 花粉物种中英文描述
│   └── requirements.txt          # Python 依赖
│
└── README.md                     # 本文件
```

---

## 模型说明

### APFA-Net

全称 Attention-guided Pollen Features Aggregation Network，出自：

> Mahmood T, Choi J, Park K R. *Artificial intelligence-based classification of pollen grains using attention-guided pollen features aggregation network.* Journal of King Saud University - Computer and Information Sciences, 2023, 35: 740-756.

核心特点：

- **深度可分离卷积** — 将标准卷积参数量降低约 8~9 倍，基础版仅 1.88M 参数
- **双分支多尺度感受野** — 3×3 + 5×5 并行的可分离卷积分支
- **密集拼接 + Skip Connection** — 四阶段分层特征提取，Concat 拼接代替 Add
- **SE 通道注意力** — Stage 1 / Stage 3 末尾插入，自适应学习通道权重
- **APFA 多尺度聚合** — 将 Stage 2 浅层细节 (空间纹理) 与 Stage 4 深层语义 (类别判别) 跨层拼接

### APFA-Net Wide

基础版在 POLLEN73S (73 类) 上准确率仅 86.81%，加宽 Stage 1 和 Stage 3 的通道数后提升至 91.56%，参数量仅增至 2.92M（仍为 ResNet34+SE 的 13.7%）。

---

## 训练方式

两阶段迁移学习：

1. **阶段一 (S1)**：在 AIpollen (36 类) 上预训练，学习花粉图像通用特征
2. **阶段二 (S2)**：加载 S1 权重（去除分类头），在 POLLEN73S (73 类) 上微调

```bash
# 标准交叉熵训练
python src/train_apfanet.py

# Bayesian Focal Loss 训练
python src/train_apfanet.py --loss bfl

# 消融实验：跳过预训练，直接训练
python src/train_apfanet.py --direct
```

关键超参数：

| 阶段 | 学习率 | Epochs | 早停 | 优化器 | 调度器 |
|------|--------|--------|------|--------|--------|
| S1 | 0.001 | 100 | patience=12 | AdamW | CosineAnnealingLR |
| S2 | 0.0003 | 100 | patience=12 | AdamW | CosineAnnealingLR |

硬件：NVIDIA RTX 4050 Laptop (6GB VRAM)，混合精度训练。

---

## 最终实验结果

全部结果均有 JSON 日志和权重文件可验证。

| 模型变体 | 训练策略 | 损失函数 | Acc | F1 |
|----------|----------|----------|-----|----|
| APFA-Net Wide | 两阶段 | CE | 91.56% | 91.52% |
| APFA-Net Wide | 两阶段 | BFL | 87.34% | 85.70% |
| APFA-Net Wide | direct | CE | 91.03% | 90.60% |
| APFA-Net 基础 | direct | CE | 86.81% | 86.33% |

---

## 网站部署

基于 Gradio 构建，部署于 ModelScope。支持三模型对比：

- **APFA-Net Wide** (2.92M) — 轻量级 CNN 注意力网络
- **ResNet34 + SE** (21.35M) — 经典残差 + 通道注意力
- **ViT-Small** (21.69M) — 视觉 Transformer

功能：图像上传 → 自动预处理（灰度化→缩放→裁剪→归一化）→ 模型推理 → Top-3 结果展示 + 物种信息描述。支持中英双语切换。

```bash
cd ms_push
pip install -r requirements.txt
python app.py
```

---

## 运行环境

- Python 3.12
- PyTorch 2.2.0 + CUDA 12.1
- 依赖见 `ms_push/requirements.txt`

---

## 注释说明

所有源码文件注释比 ≥ 1:1（符合任务书要求），使用中文注释。  
每个文件均包含模块级文档字符串、类/函数级 docstring、以及关键代码行的行内注释。


---

## 结果复现指南

以下步骤可完整复现 APFA-Net Wide（CE，两阶段）的 91.56% 结果。

### 前置条件

- Python 3.12，PyTorch 2.2.0+ CUDA 12.1
- GPU：NVIDIA RTX 4050 Laptop 6GB VRAM（或相当配置）
- 数据集放在项目根目录：
  - `POLLEN73S/` — 73 类花粉，每类一个子目录
  - `AIpollen-master/datasets/` — 36 类花粉预训练数据

### 步骤

```bash
# 1. 安装依赖
pip install -r ms_push/requirements.txt

# 2. 训练 APFA-Net Wide（CE，两阶段）— 约 110 分钟
python src/train_apfanet.py

# 3. 查看结果
# 训练完成后自动输出：Acc=91.56%, F1=91.52%
# 结果 JSON 保存在 outputs/apfanet_wide_results.json
# 权重保存在 outputs/apfanet_wide_stage2_pollen73s.pth
```

### 消融实验（可选）

```bash
# Bayesian Focal Loss 对比 — 约 40 分钟
python src/train_apfanet.py --loss bfl
# 预期：Acc≈87%, F1≈86%

# 跳过预训练，直接训练 — 约 60 分钟
python src/train_apfanet.py --direct
# 预期：Acc≈91%, F1≈90%
```

### 预期结果对照

| 指标 | 报告值 | 来源 |
|------|--------|------|
| Acc | 91.56% | `outputs/apfanet_wide_results.json` |
| F1 | 91.52% | 同上 |
| 参数量 | 2.92M | 同上 |
| 推理时间 | 5.09ms | 同上 |
| 训练耗时 | ~110 min | RTX 4050 Laptop 实测 |

### 消融结论

| 对比项 | 基线 | 变体 | 差异 |
|--------|------|------|------|
| BFL vs CE | CE: 91.56% | BFL: 87.34% | −4.22%，BFL 负提升 |
| 两阶段 vs direct | 两阶段: 91.56% | direct: 91.03% | +0.53%，迁移正向 |
| Wide vs 基础 | Wide: 91.03% | 基础: 86.81% | +4.22%，加宽有效 |

