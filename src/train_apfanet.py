"""
APFA-Net Wide 两阶段训练脚本

训练策略（两阶段迁移学习）：
    阶段一（S1）：在 AIpollen（36类）上预训练，学习花粉图像通用特征表示。
    阶段二（S2）：加载预训练权重（不含分类头），在 POLLEN73S（73类）上微调。
    
    这种策略对照 Mahmood et al. (2023) 原文设计，原文使用 POLLEN23E 预训练后
    在 POLLEN73S 上微调，我们改用 AIpollen 数据集作为预训练源。

支持的损失函数：
    - ce (默认): 标准交叉熵损失 (CrossEntropyLoss)
    - bfl: 贝叶斯焦点损失 (Bayesian Focal Loss，全局 log_var)
    - pbfl: 逐样本贝叶斯焦点损失 (Per-Sample Bayesian Focal Loss)(有一说一，这个效果更差)

使用方法：
    # CE loss（默认）
    python src/train_apfanet.py
    
    # BFL loss
    python src/train_apfanet.py --loss bfl

    # 跳过预训练，直接在 POLLEN73S 上训练（消融实验）
    python src/train_apfanet.py --direct

关键超参数：
    - 阶段一: lr=0.001, epochs=100, 早停 patience=12, AdamW + CosineAnnealing
    - 阶段二: lr=0.0003, epochs=100, 早停 patience=12, AdamW + CosineAnnealing
    - 混合精度: torch.cuda.amp (GradScaler + autocast)，适配 6GB VRAM

硬件需求：
    - GPU: RTX 4050 Laptop (6GB VRAM)
    - 阶段一耗时: ~10 分钟 (36 类，~800 样本)
    - 阶段二耗时: ~40-110 分钟 (73 类，1766 训练样本)
    - 推理速度测试: 500 次前向传播取均值（含 warmup），单位 ms/张

输出文件 (保存在 outputs/):
    - apfanet_wide_stage1_aipollen.pth                阶段一权重
    - apfanet_wide{_bfl}{_pbfl}_stage2_pollen73s.pth  阶段二权重
    - apfanet_wide{_bfl}{_pbfl}_results.json          训练结果 JSON
    - apfanet_wide_direct*.*                          直接训练模式（跳过预训练）输出

参考文献:
    [1] Mahmood et al., J. King Saud Univ. Comput. Inform. Sci., 2023 (APFA-Net 原文)
    [2] Khanzhina et al., CVPRW 2023 (Bayesian Focal Loss)
    [3] Kendall & Gal, NeurIPS 2017 (Aleatoric Uncertainty)
"""

# ===== 标准库导入 =====
import os, sys, time, json                     # 系统操作、时间记录、JSON 序列化
import torch, torch.nn as nn, torch.nn.functional as F  # PyTorch 核心库
import argparse                                  # 命令行参数解析
import numpy as np                              # 数值计算（用于早停最佳分数初始化）
from pathlib import Path                         # 路径管理（跨平台）

# ===== 项目模块导入 =====
# 将项目根目录加入 sys.path，确保无论在何处运行都能导入 src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prepare_data import get_aipollen, get_pollen73s  # 数据集加载函数
# 
# ===== Bayesian Focal Loss 类定义（内联，原 src/train.py） =====

class BayesianFocalLoss(nn.Module):
    """
    贝叶斯焦点损失 (Bayesian Focal Softmax Loss)

    论文来源:
        Khanzhina et al., "Bayesian Focal Loss: A New Loss for Imbalanced
        Classification", CVPRW 2023.

    数学公式:
        loss = exp(-log_var) * FocalLoss + log_var

    核心思想:
        将 aleatoric uncertainty 建模为可学习参数 log_var。
        - exp(-log_var): 精度，作为 FocalLoss 的加权系数
        - log_var: 正则化项，防止 log_var 无限增大
        全局 log_var: 所有样本共享同一不确定性参数。

    Focal Loss 部分:
        FL = alpha * (1 - pt)^gamma * CE
        alpha=0.25 类别平衡, gamma=2.0 聚焦难样本
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.log_var = nn.Parameter(torch.zeros(1))  # 全局对数方差

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")  # 逐样本CE
        pt = torch.exp(-ce)                      # 预测概率
        focal = self.alpha * (1 - pt) ** self.gamma * ce  # Focal Loss
        precision = torch.exp(-self.log_var)     # 精度 = 1/方差
        return precision * focal.mean() + self.log_var  # BFL


class PerSampleBayesianFocalLoss(nn.Module):
    """
    逐样本贝叶斯焦点损失 (Per-Sample BFL)

    与 BayesianFocalLoss 的区别:
        log_var 由模型为每样本单独预测 (模型输出 (logits, log_var))。

    数学公式:
        loss_i = exp(-log_var_i) * FocalLoss_i + log_var_i

    优势: 捕捉样本级别异方差不确定性
    缺点: 需模型额外输出 log_var，训练稍慢
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, log_var):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        lv = log_var.squeeze(-1)                # (B,1) -> (B,)
        precision = torch.exp(-lv)
        return (precision * focal + lv).mean()

from src.models.apfanet_wide import APFANetWide           # Wide 版 APFA-Net

# ===== 全局配置 =====
# 自动检测设备：有 CUDA 用 GPU，否则退化为 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 输出目录：项目根目录下的 outputs/
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)                 # 若不存在则创建
print(f"Device: {DEVICE}")                      # 打印当前使用的设备（训练前确认）


class EarlyStopping:
    """
    早停机制 (Early Stopping)

    当验证集准确率在连续 patience 个 epoch 内不再提升时，自动终止训练。
    每次验证分数创新高时保存当前最佳模型权重。

    原理:
        - 维护 best_score 记录历史最佳验证指标（初始 -inf）
        - counter 记录自上次提升以来的 epoch 数
        - counter >= patience 时设置 early_stop=True，外层训练循环检测后 break

    patience=12: 原文未明确说明，根据实验经验设定。
    12 epoch 约占总训练 100 epoch 的 12%，在不过早停止的前提下有效避免过拟合。

    属性:
        patience (int): 容忍轮数（默认 12）
        counter (int): 当前连续未提升的 epoch 计数
        best_score (float): 历史最佳验证分数
        early_stop (bool): 是否触发早停信号
    """
    def __init__(self, patience=12):
        """
        初始化早停器

        参数:
            patience (int): 早停容忍轮数。数值越大越不容易早停，默认 12。
        """
        # 三个核心状态变量
        self.patience, self.counter, self.best_score = patience, 0, -np.inf
        self.early_stop = False                    # 早停标志，初始未触发

    def __call__(self, score):
        """
        调用早停判断，记录最佳分数并返回是否保存权重。

        逻辑:
            - 如果当前分数 > 历史最佳: 重置 counter，返回 True（应保存）
            - 否则: counter++，若超 patience 则置 early_stop=True

        参数:
            score (float): 当前 epoch 的验证集准确率
        返回:
            bool: 如果当前是最佳分数返回 True（触发保存），否则 False
        """
        if score > self.best_score:
            # 新的最佳分数：保存并重置 counter
            self.best_score, self.counter = score, 0
            return True                              # True = 保存此权重
        # 未创新高：计数器递增
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True                   # 触发早停信号
        return False                                 # False = 不保存


@torch.no_grad()
def validate(model, loader):
    """
    验证/测试函数

    对数据加载器中的全部数据进行推理，计算两个核心指标：
    1. 准确率 (Accuracy): correct / total
    2. 宏观 F1 分数 (Macro F1): 各分类 F1 的未加权均值，适合类别不均衡场景

    使用 @torch.no_grad() 装饰器禁用梯度计算以：
    - 节省大量显存（不用存储中间激活和梯度图）
    - 加速推理（跳过反向传播计算图构建）

    注意: 模型若返回 (logits, log_var) 元组（BFL 模式），只取 logits 用于评估。

    参数:
        model (nn.Module): 待评估模型（函数内部会自动设为 eval 模式）
        loader (DataLoader): 验证/测试数据加载器
    返回:
        tuple: (准确率 float, 宏观 F1 分数 float)
    """
    model.eval()                                    # 切换到 eval 模式（关闭 BN/Dropout 的训练行为）
    correct, total = 0, 0                           # 累计正确预测数和总样本数
    all_preds, all_labels = [], []                   # 用于统一计算 F1 的预测和标签缓存
    for imgs, labels in loader:
        # 数据移动到 GPU/CPU 对应设备
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)                       # 前向推理
        # BFL 模型返回 (logits, log_var) 元组，标准模型直接返回 logits
        if isinstance(outputs, tuple):
            outputs = outputs[0]                    # 从元组中提取 logits
        preds = outputs.argmax(1)                   # 取最大概率的索引作为预测类别
        correct += (preds == labels).sum().item()   # 累加正确数
        total += labels.size(0)                     # 累加样本数
        # 收集所有预测和标签用于统一计算 F1（在循环外一次性计算更高效）
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    from sklearn.metrics import f1_score            # 延迟导入：仅验证时需要 sklearn
    # 宏观 F1: 每类 F1 的算术平均，不受类别样本量影响
    return correct / total, f1_score(all_labels, all_preds, average="macro")


def train_stage(model, train_ld, val_ld, epochs, lr, loss_fn, tag, save_path):
    """
    单个阶段的训练循环

    训练过程（每个 epoch）：
        1. 遍历训练数据，执行前向 + 反向 + 优化器更新
        2. 在验证集上评估，输出 Acc/F1
        3. 早停判断：若 val_acc 创新高则保存权重，若 patience 轮不提升则终止

    关键设计选择:
        - AdamW 优化器: 解耦权重衰减的 Adam 变体，比普通 Adam 更稳定
        - CosineAnnealingLR: 余弦退火学习率调度，从 lr 平滑递减至 0
        - 混合精度 (AMP): 前向用 float16 加速并省显存，GradScaler 防梯度下溢
        - 优化器包含 loss_fn.parameters(): BFL 的 log_var 参数也必须被训练

    参数:
        model (nn.Module): 待训练模型
        train_ld (DataLoader): 训练数据加载器
        val_ld (DataLoader): 验证数据加载器
        epochs (int): 最大训练轮数
        lr (float): 初始学习率（阶段一: 0.001, 阶段二: 0.0003）
        loss_fn (nn.Module): 损失函数（CE / BFL / PBFL）
        tag (str): 阶段标签（用于日志输出，如 "S1" / "S2"）
        save_path (Path): 最佳权重保存路径
    返回:
        tuple: (最佳验证准确率 float, 训练耗时秒数 float)
    """
    # AdamW 优化器：解耦权重衰减 (weight_decay=0.01 即 L2 正则化系数)
    # 关键：list(loss_fn.parameters()) 确保 BFL 的 log_var 也被优化器跟踪
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                            lr=lr, weight_decay=0.01)
    # CosineAnnealingLR: 学习率按余弦曲线从 lr 平滑衰减至 0
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    estop = EarlyStopping(patience=12)              # 早停器，patience=12
    # GradScaler: 混合精度训练的梯度缩放器（防止 float16 梯度下溢）
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best_acc, t0 = 0.0, time.time()                 # 最佳 val_acc 和起始时间

    for ep in range(epochs):
        model.train()                               # 切换到训练模式（启用 BN/Dropout）
        loss_sum = 0.0                              # 本轮累积损失（用于计算均值）
        for imgs, labels in train_ld:
            # 数据迁移到 GPU
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()                         # 清空梯度（PyTorch 默认累积梯度）
            # autocast: 启动混合精度上下文，自动将部分运算转为 float16
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                out = model(imgs)                    # 模型前向传播
                # BFL 模型返回 (logits, log_var)，需传入 log_var 给 loss
                if isinstance(out, tuple):
                    loss = loss_fn(out[0], labels, out[1])
                else:
                    loss = loss_fn(out, labels)
            # GradScaler: 缩放损失 -> 反向传播 -> 缩放梯度 -> 更新参数
            scaler.scale(loss).backward()           # 缩放后的损失反向传播
            scaler.step(opt)                        # 缩放后的梯度步进
            scaler.update()                         # 更新缩放因子
            # 累加损失：loss.item() * batch_size，确保最终除以总样本数
            loss_sum += loss.item() * imgs.size(0)
        sch.step()                                  # 每个 epoch 后更新学习率
        # 验证阶段：评估当前模型在验证集上的表现
        vacc, vf1 = validate(model, val_ld)
        elapsed = time.time() - t0                  # 已用时
        # 日志输出：阶段标签 / epoch 进度 / 损失 / 准确率 / F1 / 耗时
        print(f"[{tag}] E{ep+1:3d}/{epochs} | Loss:{loss_sum/len(train_ld.dataset):.4f} "
              f"| Acc:{vacc:.4f} F1:{vf1:.4f} | {elapsed:.0f}s")
        if estop(vacc):                             # 创新高 -> 保存权重
            torch.save(model.state_dict(), save_path)
            best_acc = vacc
            print(f"  >> Saved (Acc={vacc:.4f})")
        if estop.early_stop:                        # 早停触发 -> 退出训练循环
            break
    # 兜底保护：若从未触发保存（所有 epoch 分数都极低，极端情况），保存最后一 epoch
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
    return best_acc, time.time() - t0


def main(loss_name="ce", direct=False):
    """
    APFA-Net Wide 主训练流程

    执行顺序:
        1. 初始化模型并打印参数量（2.92M）
        2. 根据命令行参数选择损失函数（CE / BFL / PBFL）
        3. 阶段一 (S1): AIpollen 36 类预训练
           - 若权重已存在则跳过训练，直接载入评估
           - --direct 模式下完全跳过（消融实验用）
        4. 阶段二 (S2): 加载 S1 权重（去分类头 + log_var 头），在 POLLEN73S 73 类上微调
        5. 推理速度 Benchmark: 500 次前向传播取均值，含 30 次 warmup
        6. 将训练结果写入 JSON 文件（供报告和网站使用）

    两阶段迁移学习的设计理由:
        - 阶段一提供花粉图像的通用底层特征（纹理、边缘、形状等）
        - 阶段二在目标数据集上微调，学习 73 类的细粒度判别特征
        - 这种策略可有效缓解 POLLEN73S 的样本量不足问题（73 类仅 2523 张）

    参数:
        loss_name (str): 损失函数选择。"ce" (默认), "bfl", "pbfl"。
        direct (bool): 是否跳过阶段一预训练。用于消融实验，验证预训练的必要性。
    """
    # 计算并打印模型参数量（以 36 类为例，因阶段一为 36 类）
    params = sum(p.numel() for p in APFANetWide(num_classes=36).parameters()) / 1e6
    print(f"APFANetWide: {params:.2f}M params")

    # ===== 损失函数初始化 =====
    # use_uncertainty: 是否启用不定性输出（只有 pbfl 使用逐样本 log_var）
    # bfl 使用全局 log_var（在 loss 内部），不需要模型侧 uncertainty 分支
    use_uncertainty = (loss_name == "pbfl")
    if loss_name == "ce":
        loss_fn = nn.CrossEntropyLoss()             # 标准交叉熵
    elif loss_name == "bfl":
        loss_fn = BayesianFocalLoss()               # BFL：全局 log_var，loss 内部管理
    else:
        loss_fn = PerSampleBayesianFocalLoss()      # PBFL：每样本 log_var，模型侧输出
    loss_fn = loss_fn.to(DEVICE)                    # 损失函数移到 GPU
    # 损失函数标签（用于输出文件名区分，如 _bfl, _pbfl, 空=ce）
    loss_tag = "" if loss_name == "ce" else ("_pbfl" if loss_name == "pbfl" else "_bfl")

    # 阶段一权重保存路径
    stage1_path = OUTPUT_DIR / "apfanet_wide_stage1_aipollen.pth"

    # ============================================================
    # 阶段一 (S1): AIpollen 36 类预训练
    # ============================================================
    if direct:
        # --direct 模式：跳过预训练（消融实验，证明预训练的价值）
        print("\n[SKIP] Stage 1 (--direct mode)")
        test_acc1, t1, best_acc1 = 0, 0, 0          # 无 S1 结果
    elif stage1_path.exists():
        # 权重已存在：跳过训练，直接载入并在测试集上评估
        print("\n[SKIP] Stage 1 already done, using existing weights")
        model = APFANetWide(num_classes=36, uncertainty=use_uncertainty).to(DEVICE)
        # weights_only=True: 安全加载，仅允许张量数据（防止 pickle 代码注入）
        model.load_state_dict(torch.load(stage1_path, map_location=DEVICE,
                                         weights_only=True), strict=False)
        tl, vl, tsl, tn, vn, tsn, _ = get_aipollen()  # 加载 AIpollen 数据集
        test_acc1, test_f1_1 = validate(model, tsl)    # 用现有权重评估测试集
        print(f"Stage1 Test: Acc={test_acc1:.4f} F1={test_f1_1:.4f}")
        best_acc1, t1 = test_acc1, 0                   # 记录结果（耗时为 0）
    else:
        # 权重缺失：正常执行预训练
        print("\n" + "="*60)
        print("STAGE 1: AIpollen (36 classes)")
        print("="*60)
        tl, vl, tsl, tn, vn, tsn, _ = get_aipollen()  # 加载数据
        print(f"Train: {tn} Val: {vn} Test: {tsn}")    # 打印数据量统计
        model = APFANetWide(num_classes=36, uncertainty=use_uncertainty).to(DEVICE)
        # 阶段一训练: lr=0.001 (较大，从零开始学习)
        best_acc1, t1 = train_stage(model, tl, vl, 100, 0.001, loss_fn, "S1", stage1_path)
        # 重新加载最佳权重（train_stage 内部已保存）用于测试评估
        model.load_state_dict(torch.load(stage1_path, map_location=DEVICE,
                                         weights_only=True))
        test_acc1, test_f1_1 = validate(model, tsl)
        print(f"Stage1 Test: Acc={test_acc1:.4f} F1={test_f1_1:.4f}")

    # ============================================================
    # 阶段二 (S2): POLLEN73S 73 类微调
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 2: POLLEN73S (73 classes)")
    print("="*60)
    # 加载 POLLEN73S 完整数据（训练/验证/测试 + 73 类名称列表）
    tl2, vl2, tsl2, tn2, vn2, tsn2, names_73 = get_pollen73s()
    print(f"Train: {tn2} Val: {vn2} Test: {tsn2}")

    # 创建 S2 模型（73 分类头）
    model2 = APFANetWide(num_classes=73, uncertainty=use_uncertainty).to(DEVICE)
    if not direct:
        # 非 direct 模式：加载 S1 权重作为初始化（迁移学习核心步骤）
        state = torch.load(stage1_path, map_location=DEVICE, weights_only=True)
        # 移除分类头权重：S1 是 36 类，S2 是 73 类，维度不匹配
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        # 移除不定性头权重：log_var 层需要重新在 S2 数据集上训练
        state.pop("log_var_fc.weight", None)
        state.pop("log_var_fc.bias", None)
        # strict=False: 允许部分 key 缺失（已移除的 fc 和 log_var_fc）
        model2.load_state_dict(state, strict=False)

    # 确定权重保存路径（区分 direct 模式和损失函数）
    if direct:
        stage2_path = OUTPUT_DIR / f"apfanet_wide_direct{loss_tag}_stage2_pollen73s.pth"
    else:
        stage2_path = OUTPUT_DIR / f"apfanet_wide{loss_tag}_stage2_pollen73s.pth"
    # 阶段二训练: lr=0.0003 (较小，微调预训练权重)
    best_acc2, t2 = train_stage(model2, tl2, vl2, 100, 0.0003, loss_fn, "S2", stage2_path)

    # 加载 S2 最佳权重评估测试集（最终指标）
    model2.load_state_dict(torch.load(stage2_path, map_location=DEVICE, weights_only=True))
    test_acc2, test_f2 = validate(model2, tsl2)
    print(f"Stage2 Test: Acc={test_acc2:.4f} F1={test_f2:.4f}")

    # ============================================================
    # 推理速度 Benchmark
    # ============================================================
    # 创建单张 224x224 随机图像作为输入（模拟实际推理场景）
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)    # batch=1, 3 通道 RGB, 224x224
    model2.eval()                                      # 确保 eval 模式
    # Warmup: 30 次预热推理，消除 CUDA kernel JIT 编译的冷启动偏差
    for _ in range(30):
        with torch.no_grad():
            out = model2(dummy)
            _ = out[0] if isinstance(out, tuple) else out  # 统一处理元组/logits
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()                       # 确保所有 CUDA 操作完成
    # 正式计时: 500 次推理取平均，用 perf_counter 提供高精度计时
    t0 = time.perf_counter()
    for _ in range(500):
        with torch.no_grad():
            out = model2(dummy)
            _ = out[0] if isinstance(out, tuple) else out
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()                       # 确保最后一次推理完成
    # 平均推理时间（ms/张）= 总时间 / 500 * 1000
    infer_ms = (time.perf_counter() - t0) / 500 * 1000

    # ============================================================
    # 保存结果 JSON（供论文报告和网站 metrics 使用）
    # ============================================================
    results = {
        "tag": f"apfanet_wide_direct{loss_tag}" if direct else f"apfanet_wide{loss_tag}",
        "model": "apfanet_wide",                       # 模型标识
        "params_M": round(params, 2),                   # 参数量（百万）
        "loss": loss_name,                             # 损失函数名称
        "augment": True,                               # 是否启用数据增强
        "two_stage": False if direct else True,         # 是否两阶段训练
        "inference_ms": round(infer_ms, 2),             # 单张推理时间 (ms)
        "stage1_aipollen": {                            # 阶段一结果
            "test_acc": float(test_acc1),               #   测试准确率
            "test_f1": float(test_f1_1),                #   测试 F1
            "time_s": round(t1, 1)                      #   训练耗时 (秒)
        },
        "stage2_pollen73s": {                           # 阶段二结果（最终指标）
            "test_acc": float(test_acc2),               #   测试准确率
            "test_f1": float(test_f2),                  #   测试 F1
            "best_val_acc": float(best_acc2),           #   最佳验证准确率
            "time_s": round(t2, 1)                      #   训练耗时 (秒)
        },
    }
    # 保存路径同样区分 direct 模式
    if direct:
        results_file = OUTPUT_DIR / f"apfanet_wide_direct{loss_tag}_results.json"
    else:
        results_file = OUTPUT_DIR / f"apfanet_wide{loss_tag}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)                # indent=2 美化输出
    # 最终汇总打印
    print(f"\nDone! Acc={test_acc2:.4f} F1={test_f2:.4f} Params={params:.2f}M Infer={infer_ms:.1f}ms")


if __name__ == "__main__":
    # ===== 命令行参数解析 =====
    parser = argparse.ArgumentParser(
        description="APFA-Net Wide 两阶段训练脚本 - 花粉识别")
    # --loss: 损失函数选择
    parser.add_argument("--loss", choices=["ce", "bfl", "pbfl"], default="ce",
                        help="损失函数: ce=交叉熵(默认), bfl=贝叶斯焦点损失, "
                             "pbfl=逐样本贝叶斯焦点损失")
    # --direct: 跳过预训练（消融实验用）
    parser.add_argument("--direct", action="store_true",
                        help="跳过阶段一预训练，直接在 POLLEN73S 上训练（消融实验用）")
    args = parser.parse_args()                         # 解析命令行参数
    main(loss_name=args.loss, direct=args.direct)      # 启动主训练流程
