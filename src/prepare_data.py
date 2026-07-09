"""
数据加载与预处理模块

负责三个数据集的加载、划分和预处理:
    - AIpollen (36 类): 指导老师论文参考项目的数据集，用于阶段一预训练
    - POLLEN73S (73 类): 主实验数据集，2523 张 224x224 光学显微镜图像
    - POLLEN23E (23 类): 辅助小数据集（对应原文 Table 1 的 POLLEN23E）

数据划分策略:
    使用 StratifiedShuffleSplit 分层随机划分，保证每类在各子集中的比例一致。
    - 训练集: 70%
    - 验证集: 15%
    - 测试集: 15%
    固定 seed=42 确保划分结果可复现。

预处理管线:
    训练阶段: 灰度化 -> 缩放 256 -> 裁剪 224 -> 随机水平/垂直翻转 ->
             RandAugment(2, m=9) -> ToTensor -> RandomErasing(p=0.3) -> ImageNet 归一化
    评估阶段: 灰度化 -> 缩放 256 -> 裁剪 224 -> ToTensor -> ImageNet 归一化
    （评估阶段不启用任何数据增强）

关键设计决策:
    - 灰度化 (3 通道输出): 花粉显微图像本身是灰度图，分离颜色信息减少过拟合
    - RandAugment: 自动搜索最优增强策略，比手动设计增强更鲁棒
    - RandomErasing: 随机遮挡图像区域，模拟真实显微图像中可能的遮挡/污渍
    - ImageNet 归一化: 使用标准统计量，兼容预训练模型的输入分布

使用方法:
    from src.prepare_data import get_pollen73s
    train_ld, val_ld, test_ld, n_train, n_val, n_test, class_names = get_pollen73s()
"""

# ===== 标准库/框架导入 =====
import os                                          # 文件系统操作
import torch                                       # PyTorch 核心
from torch.utils.data import Dataset, DataLoader    # 数据集和数据加载器
from torchvision import transforms                  # 图像变换
from sklearn.model_selection import StratifiedShuffleSplit  # 分层随机划分
import numpy as np                                  # 数值计算
from PIL import Image                               # PIL 图像库

# ===== 全局配置常量 =====
BATCH_SIZE = 32                                     # 批大小（适配 6GB VRAM 的 RTX 4050）
IMG_SIZE = 224                                      # 模型输入尺寸
NUM_WORKERS = 0                                      # DataLoader 子进程数（0 = 主进程加载，兼容 Windows）
SEED = 42                                           # 随机种子（确保划分可复现，42 是 ML 社区惯例）


def _walk_dataset(root):
    """
    遍历数据集目录，收集所有图像路径和对应标签

    扫描逻辑:
        1. 按字母序遍历 root 下的所有子目录，每个子目录代表一个类别
        2. 收集每个子目录内的图像文件（支持 .jpg/.jpeg/.png/.tiff/.tif/.bmp）
        3. 标签按子目录遍历顺序自动分配（0, 1, 2, ...）

    参数:
        root (str): 数据集根目录路径，其下每个子目录为一个类别
    返回:
        tuple: (图像路径列表, 标签列表, 类别名称列表)
    """
    paths, labels, class_names = [], [], []          # 初始化三个空列表
    # 按字母序遍历确保每次运行结果一致（sorted 保证确定性）
    for cls_name in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue                                 # 跳过非目录文件（如 README.txt）
        class_names.append(cls_name)                 # 记录类别名称
        label = len(class_names) - 1                 # 标签 = 当前类别索引
        # 遍历类别目录内所有图像文件
        for fname in sorted(os.listdir(cls_dir)):
            low = fname.lower()
            # 仅处理支持的图像格式
            if not low.endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")):
                continue
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label)
    return paths, labels, class_names


class PollenDataset(Dataset):
    """
    PyTorch 花粉图像数据集

    继承 torch.utils.data.Dataset，实现标准的 __len__ 和 __getitem__ 接口。
    支持 RGBA 格式（合成到白色背景）和任意 RGB 兼容格式。

    参数:
        paths (list): 图像文件绝对路径列表
        labels (list): 对应的整数标签列表
        transform (callable): torchvision 预处理变换（可为 None）
    """

    def __init__(self, paths, labels, transform=None):
        self.paths = paths                           # 图像路径列表
        self.labels = labels                         # 标签列表
        self.transform = transform                   # 预处理变换（训练用强增强，评估用基础变换）

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        按索引获取单个样本

        处理流程:
            1. PIL 加载图像
            2. RGBA -> RGB（白色背景合成，消除透明通道干扰）
            3. 应用预处理变换
            4. 返回 (图像张量, 标签)

        参数:
            idx (int): 样本索引
        返回:
            tuple: (预处理后的图像张量, 整数标签)
        """
        img = Image.open(self.paths[idx])
        # 透明格式处理: RGBA/P -> 白色背景合成
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])       # Alpha 通道作为遮罩
            img = bg
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")                 # 其他格式统一转 RGB
        if self.transform:
            img = self.transform(img)                # 应用预处理变换
        return img, self.labels[idx]


def _split(paths, labels, train_r=0.7, val_r=0.15, seed=SEED):
    """
    分层随机划分数据集

    划分策略 (两步法):
        Step 1: 用 StratifiedShuffleSplit 分出训练集 (train_r) 和临时集 (val_r + test_r)
        Step 2: 在临时集内再次分层划分出验证集和测试集

    分层 (Stratified) 的意义:
        确保每类的样本在各子集中比例近似相同，避免某类全部落入测试集。
        对 73 类花粉识别任务尤其重要——部分类别样本量极少，随机划分可能导致
        某些类的测试样本数为 0，使得 F1 计算失效。

    参数:
        paths (list): 图像路径列表
        labels (list): 对应的标签列表
        train_r (float): 训练集比例 (默认 0.7)
        val_r (float): 验证集比例 (默认 0.15)
        seed (int): 随机种子 (默认 SEED=42)
    返回:
        tuple: (训练集索引, 验证集索引, 测试集索引)
    """
    labels = np.array(labels)
    idx = np.arange(len(paths))
    test_r = 1.0 - train_r - val_r                   # 测试集比例 = 剩余部分

    # Step 1: 分出训练集 + 临时集(验证+测试)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=val_r + test_r,
                                random_state=seed)
    tr_idx, tmp_idx = next(s1.split(idx, labels))    # 训练索引 + 临时索引
    tmp_lab = labels[tmp_idx]

    # Step 2: 从临时集中分出验证集和测试集
    # vf = 验证集在临时集中的比例
    vf = val_r / (val_r + test_r)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - vf, random_state=seed)
    v_rel, t_rel = next(s2.split(tmp_idx, tmp_lab))
    # 使用相对索引映射回原始索引
    return tr_idx, tmp_idx[v_rel], tmp_idx[t_rel]


def _train_tf():
    """
    训练阶段数据增强变换

    增强策略 (从轻到重):
        1. Grayscale (3ch): 灰度化，移除色彩信息
        2. Resize(256) + CenterCrop(224): 标准缩放裁剪
        3. RandomHorizontalFlip(p=0.5): 随机水平翻转（显微镜图像可能有不同朝向）
        4. RandomVerticalFlip(p=0.5): 随机垂直翻转
        5. RandAugment(num_ops=2, magnitude=9): 自动数据增强
           - 每次随机选择 2 种增强操作（如亮度、对比度、旋转等）
           - magnitude=9 表示中等强度，兼顾多样性和真实性
        6. ToTensor: 转为 [0,1] 浮点张量
        7. RandomErasing(p=0.3): 随机遮挡矩形区域（模拟图像局部污损）
        8. Normalize: ImageNet 标准归一化

    设计考量:
        RandAugment 比手动 AutoAugment 更简单高效，且 magnitude=9 对细粒度任务
        不会过度变形花粉的判别性特征。RandomErasing 增强模型的鲁棒性。
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),      # 灰度化（3 通道，值相同）
        transforms.Resize((256, 256)),                     # 等比缩放
        transforms.CenterCrop(IMG_SIZE),                   # 中心裁剪 224x224
        transforms.RandomHorizontalFlip(p=0.5),            # 50% 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),              # 50% 概率垂直翻转
        transforms.RandAugment(num_ops=2, magnitude=9),    # 自动增强（2 操作，强度 9）
        transforms.ToTensor(),                             # HWC -> CHW, [0,255] -> [0,1]
        transforms.RandomErasing(p=0.3),                   # 30% 概率随机擦除
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 均值
                             std=[0.229, 0.224, 0.225]),   # ImageNet 标准差
    ])


def _noaug_tf():
    """
    无数据增强的预处理变换（消融实验用）

    仅包含基本预处理: 灰度化 -> 缩放 -> 裁剪 -> ToTensor -> 归一化
    用于验证数据增强对模型性能的贡献。
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),      # 灰度化
        transforms.Resize((256, 256)),                     # 缩放
        transforms.CenterCrop(IMG_SIZE),                   # 裁剪
        transforms.ToTensor(),                             # 转张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 归一化
                             std=[0.229, 0.224, 0.225]),
    ])


def _eval_tf():
    """
    验证/测试阶段预处理变换

    与 _noaug_tf 相同，仅做基础预处理，不包含任何数据增强。
    评估阶段必须保持确定性和一致性，避免随机性干扰指标计算。
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),      # 灰度化
        transforms.Resize((256, 256)),                     # 缩放
        transforms.CenterCrop(IMG_SIZE),                   # 裁剪
        transforms.ToTensor(),                             # 转张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 归一化
                             std=[0.229, 0.224, 0.225]),
    ])


def _build(all_p, all_l, tr_i, v_i, ts_i, train_tf):
    """
    构建 DataLoader 的三个子集（训练/验证/测试）

    参数:
        all_p (list): 所有图像路径
        all_l (list): 所有标签
        tr_i (ndarray): 训练集索引
        v_i (ndarray): 验证集索引
        ts_i (ndarray): 测试集索引
        train_tf (callable): 训练阶段预处理变换
    返回:
        tuple: (train_loader, val_loader, test_loader, n_train, n_val, n_test)
    """
    # 创建三个 PollenDataset 实例，训练集使用增强变换
    tr = PollenDataset([all_p[i] for i in tr_i], [all_l[i] for i in tr_i], train_tf)
    va = PollenDataset([all_p[i] for i in v_i],  [all_l[i] for i in v_i],  _eval_tf())
    te = PollenDataset([all_p[i] for i in ts_i], [all_l[i] for i in ts_i], _eval_tf())
    # 返回 DataLoader: 训练集 shuffle=True，验证/测试集 shuffle=False
    # pin_memory=True: 加速 CPU->GPU 数据传输（仅 CUDA 环境生效）
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS,
                   pin_memory=True),
        DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                   pin_memory=True),
        DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                   pin_memory=True),
        len(tr), len(va), len(te),                  # 返回各级样本数（用于日志输出）
    )


# 项目根目录（prepare_data.py 在 src/ 下，根目录在上一级）
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_aipollen(augment=True):
    """
    加载 AIpollen 数据集（36 类）

    数据来源: AIpollen-master/datasets/
    过滤规则: 排除以 "mask_" 或 "augmented_" 开头的文件（避免重复/非标准样本）

    参数:
        augment (bool): 训练集是否启用数据增强 (默认 True)
    返回:
        tuple: (train_loader, val_loader, test_loader, n_train, n_val, n_test, class_names)
    """
    root = os.path.join(BASE, "AIpollen-master", "datasets")
    ap, al, cn = _walk_dataset(root)
    # 过滤增强/掩码文件: 这些文件是原始数据集的变体，不应纳入训练
    keep = [i for i, p in enumerate(ap)
            if not os.path.basename(p).lower().startswith(("mask_", "augmented_"))]
    ap, al = [ap[i] for i in keep], [al[i] for i in keep]
    ti, vi, tsi = _split(ap, al)
    return _build(ap, al, ti, vi, tsi, _train_tf() if augment else _noaug_tf()) + (cn,)


def get_pollen73s(augment=True):
    """
    加载 POLLEN73S 数据集（73 类，主实验数据集）

    数据来源: POLLEN73S/
    总计 2523 张 224x224 光学显微镜花粉图像。
    默认划分: 训练 1766 / 验证 378 / 测试 379。

    参数:
        augment (bool): 训练集是否启用数据增强 (默认 True)
    返回:
        tuple: (train_loader, val_loader, test_loader, n_train, n_val, n_test, class_names)
    """
    root = os.path.join(BASE, "POLLEN73S")
    ap, al, cn = _walk_dataset(root)
    ti, vi, tsi = _split(ap, al)
    return _build(ap, al, ti, vi, tsi, _train_tf() if augment else _noaug_tf()) + (cn,)


def get_pollen23e(augment=True):
    """
    加载 POLLEN23E 数据集（23 类，辅助小数据集）

    对应 APFA-Net 原文 (Mahmood et al., 2023) 的 POLLEN23E。
    用于消融实验和小规模验证。

    参数:
        augment (bool): 训练集是否启用数据增强 (默认 True)
    返回:
        tuple: (train_loader, val_loader, test_loader, n_train, n_val, n_test, class_names)
    """
    root = os.path.join(BASE, "POLLEN23E")
    ap, al, cn = _walk_dataset(root)
    ti, vi, tsi = _split(ap, al)
    return _build(ap, al, ti, vi, tsi, _train_tf() if augment else _noaug_tf()) + (cn,)


# ===== 自检代码 =====
if __name__ == "__main__":
    # 依次加载三个数据集，验证数据加载管道正常工作
    for nm, fn in [("AIpollen", get_aipollen),
                   ("POLLEN73S", get_pollen73s),
                   ("POLLEN23E", get_pollen23e)]:
        tl, vl, tsl, tn, vn, tsn, names = fn()
        x, y = next(iter(tl))                        # 获取一个 batch
        print(f"{nm}: train={tn} val={vn} test={tsn} "
              f"cls={len(names)} batch={list(x.shape)}")
    print("All OK")
