"""
花粉识别系统 - Gradio Web 应用

功能概述:
    提供基于三种深度学习模型的花粉显微图像在线识别服务。
    - APFA-Net Wide (2.92M): 轻量级 CNN 注意力网络，深度可分离卷积
    - ResNet34 + SE (21.35M): 经典残差网络 + 通道注意力机制
    - ViT-Small (21.69M): 视觉 Transformer，全局自注意力

预处理管线:
    原始图像 -> 灰度化(保持3通道) -> 缩放到256x256 -> 中心裁剪224x224
    -> 转Tensor [0,1] -> ImageNet标准化

输出:
    - Top-3 预测结果（含概率百分比和奖牌图标）
    - 物种信息描述（中英双语）
    - 预处理步骤可视化
    - 推理耗时显示

UI 框架: Gradio
部署平台: ModelScope
"""

# ===== 标准库/第三方库导入 =====
import json, os, io, time, sys, base64          # JSON解析、文件操作、时间、base64编码
from pathlib import Path                          # 路径管理

import torch, torch.nn as nn, torch.nn.functional as F  # PyTorch 深度学习框架
import numpy as np                                # 数值计算（图像数组处理）
from PIL import Image                             # PIL 图像处理库
from torchvision import transforms                # torchvision 图像变换
import timm                                       # PyTorch Image Models 库（ViT等预训练模型）
import gradio as gr                               # Gradio Web UI 框架

# ===== 本地模型模块导入 =====
from apfanet_wide import APFANetWide              # Wide 版 APFA-Net（2.92M参数）
from resnet import ResNet34SE                     # ResNet34 + Squeeze-and-Excitation

# ===== 路径配置 =====
# HERE: 当前脚本所在目录（ms_push/），所有路径以此为基准
HERE = os.path.dirname(os.path.abspath(__file__))
# 权重目录：存放 .pth 模型文件
WEIGHTS_DIR = os.path.join(HERE, "weights")
# 类名映射: 73 个花粉类别索引 -> 拉丁学名
CLASS_NAMES = json.load(open(os.path.join(HERE, "class_names_73.json"), "r", encoding="utf-8"))
# 花粉信息: 拉丁学名 -> {中英文描述}
POLLEN_INFO = json.load(open(os.path.join(HERE, "pollen_info.json"), "r", encoding="utf-8"))
NUM_CLASSES = len(CLASS_NAMES)                    # 分类类别数: 73

# ===== 图像预处理变换定义 =====
# ImageNet 标准归一化参数（RGB三通道均值和标准差）
# 注意：虽然输入被灰度化了（三通道值相同），但使用 ImageNet 统计量不影响模型行为
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# TF_VIZ: 可视化用预处理（不包含 ToTensor 和归一化，保留 0-255 uint8 图像）
# 这样可以在网页上直接显示预处理中间结果
TF_VIZ = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 灰度化：将RGB转为灰度，输出3通道（值相同）
    transforms.Resize((256, 256)),                 # 等比缩放至 256x256
    transforms.CenterCrop(224),                    # 中心裁剪至 224x224（模型输入尺寸）
])

# TF_DEFAULT: 模型推理用完整预处理（含 ToTensor + Normalize）
TF_DEFAULT = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 灰度化（3通道）
    transforms.Resize((256, 256)),                 # 缩放
    transforms.CenterCrop(224),                    # 裁剪到模型输入尺寸
    transforms.ToTensor(),                         # 转为 Tensor [0, 1] 浮点型
    NORM,                                          # ImageNet 标准化
])

# ===== 模型注册表 =====
# 每个模型包含三个要素: weight_path, builder函数, transform
# 注意: APFA-Net 权重文件名为 "apfanet_direct" 是历史遗留，
# 实际加载的是 Wide 变体 (2.92M 参数, ~12MB)
MODEL_REGISTRY = {
    "apfanet": {
        "path": os.path.join(WEIGHTS_DIR, "apfanet_direct_stage2_pollen73s.pth"),
        "builder": lambda: APFANetWide(num_classes=NUM_CLASSES),
        "transform": TF_DEFAULT,
    },
    "resnet34_se": {
        "path": os.path.join(WEIGHTS_DIR, "resnet34_se_stage2_pollen73s.pth"),
        "builder": lambda: ResNet34SE(NUM_CLASSES, pretrained=False),
        "transform": TF_DEFAULT,
    },
    "vit_small": {
        "path": os.path.join(WEIGHTS_DIR, "vit_small_stage2_pollen73s.pth"),
        # 使用 timm 库创建 ViT-Small: patch_size=16, 输入 224x224
        "builder": lambda: timm.create_model("vit_small_patch16_224", pretrained=False,
                                              num_classes=NUM_CLASSES),
        "transform": TF_DEFAULT,
    },
}

# 设备检测: 有 CUDA 用 GPU，否则退化为 CPU
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 模型缓存与加载 =====
# _model_cache: 已加载模型的缓存字典 {model_name: (model, class_names, transform)}
# 避免每次预测都重新加载模型，显著降低响应延迟
_model_cache = {}

def _load_model(model_name):
    """
    加载并缓存模型

    加载流程:
        1. 检查缓存，若命中则直接返回
        2. 根据 MODEL_REGISTRY 构建模型并加载权重
        3. 处理权重 key 前缀（兼容 DataParallel 的 "module." 前缀）
        4. 通过 5 次 dummy 前向传播预热 CUDA kernel
        5. 存入缓存并返回

    参数:
        model_name (str): 模型名称，必须存在于 MODEL_REGISTRY
    返回:
        tuple: (model, class_names, transform)
    """
    # 缓存命中: 直接返回已加载模型
    if model_name in _model_cache:
        return _model_cache[model_name]
    cfg = MODEL_REGISTRY[model_name]
    # 创建模型实例并加载权重
    model = cfg["builder"]().to(DEV)
    state = torch.load(cfg["path"], map_location=DEV, weights_only=True)
    # 兼容保存格式: 有些权重用 {"state_dict": ...} 包装
    if "state_dict" in state:
        state = state["state_dict"]
    # 去除 DataParallel 的 "module." 前缀（单GPU推理不需要）
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()                                    # 切换到推理模式
    # 预热: 执行几次虚拟推理，确保 CUDA kernel 编译缓存
    dummy = torch.randn(1, 3, 224, 224).to(DEV)     # 创建模拟 224x224 输入
    for _ in range(5):
        with torch.no_grad():
            out = model(dummy)
            _ = out[0] if isinstance(out, tuple) else out
    if DEV.type == "cuda":
        torch.cuda.synchronize()                     # 确保 CUDA 操作完成
    # 存入缓存
    _model_cache[model_name] = (model, CLASS_NAMES, cfg["transform"])
    return _model_cache[model_name]

# 当前选中的模型状态（全局变量，避免重复加载）
current_model_name = None                            # 当前模型名称
current_model = None                                 # 当前模型实例
current_cls = None                                   # 当前类别名称列表
current_tf = None                                    # 当前预处理变换

def _ensure_model(model_name):
    """
    确保指定模型已加载到全局变量中

    仅在模型切换时才重新加载，避免不必要的 I/O 和 GPU 操作。
    使用 global 变量缓存模型跨请求复用。

    参数:
        model_name (str): 需要加载的模型名称
    """
    global current_model_name, current_model, current_cls, current_tf
    if model_name != current_model_name:
        # 模型切换: 重新加载
        current_model, current_cls, current_tf = _load_model(model_name)
        current_model_name = model_name

# ===== 模型性能指标加载 =====
# 从 results.json 读取当前选中模型的评估指标
# 若文件不存在或格式异常，metrics 为空字典，页面指标区不显示
try:
    metrics = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
except:
    metrics = {}

# ===== 图像加载器（格式兼容） =====
def load_any_image(fp):
    """
    通用图像加载函数，兼容多种格式

    处理逻辑:
        - RGBA/P/LA 格式: 先转为 RGBA
        - RGBA 格式: 合成到白色背景（消除透明通道）
        - 非 RGB 格式: 统一转为 RGB

    参数:
        fp: 图像文件路径或类路径对象
    返回:
        np.ndarray: RGB 格式的 uint8 数组 (H, W, 3)
    """
    img = Image.open(fp)
    # 处理调色板模式和透明模式
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        # RGBA -> RGB: 在白色背景上合成
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])          # Alpha 通道作为遮罩
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


# ===== 预处理管线可视化辅助函数 =====
def _pil_to_b64uri(img_array, size=(150, 150)):
    """
    将 numpy 图像数组转为 base64 Data URI

    用于在 HTML 中内嵌预览图像（无需额外的 HTTP 请求）。
    生成格式: data:image/png;base64,...

    参数:
        img_array (np.ndarray): uint8 图像数组
        size (tuple): 缩略图目标尺寸
    返回:
        str: base64 编码的 Data URI 字符串
    """
    img = Image.fromarray(img_array)
    img.thumbnail(size, Image.LANCZOS)              # 高质量缩略图
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_pipeline_html(orig, gray, resized, cropped):
    """
    构建预处理流程的 HTML 可视化

    展示四步预处理管线（带箭头连接）:
        原始图像 -> 灰度化 -> 缩放 256 -> 裁剪 224
    之后用文字标注: Normalize -> Model

    参数:
        orig (np.ndarray): 原始 RGB 图像
        gray (np.ndarray): 灰度化结果
        resized (np.ndarray): 缩放结果
        cropped (np.ndarray): 裁剪结果
    返回:
        str: 完整的 HTML 字符串
    """
    # 四个图像步骤（原始、灰度、缩放、裁剪）
    imgs = [
        (orig, 'Original'),
        (gray, 'Grayscale'),
        (resized, 'Resize 256'),
        (cropped, 'Crop 224'),
    ]
    steps_html = []
    for arr, label in imgs:
        b64 = _pil_to_b64uri(arr)
        tag = ('<div class="pipe-step"><img src="' + b64 + '" alt="' + label
               + '"/><span>' + label + '</span></div>')
        steps_html.append(tag)
    # 用箭头连接各步骤
    arrows = '<div class="pipe-arrow">&rarr;</div>'
    img_flow = arrows.join(steps_html)
    # 后续步骤（Normalize 和 Model）用文字图标表示
    norm_step = ('<div class="pipe-step pipe-text"><div class="pipe-icon">&mu;,&sigma;'
                 '</div><span>Normalize</span></div>')
    model_step = ('<div class="pipe-step pipe-text"><div class="pipe-icon">&#x1F9E0;'
                  '</div><span>Model</span></div>')
    return ('<div class="pipeline-row"><div class="pipeline-title">Preprocessing Pipeline'
            '</div><div class="pipeline-flow">' + img_flow + arrows + norm_step + arrows
            + model_step + '</div></div>')


# ===== 核心分类函数 =====
_LAST_FP = None  # 记录最近一次上传的文件路径（模型切换时复用）

def classify(fp, model_name, lang):
    """
    图像分类主函数

    完整预测管线:
        1. 加载图像（支持多种格式）
        2. 预处理: 灰度化 -> 缩放256 -> 裁剪224 -> Tensor -> 归一化
        3. 模型前向推理 + Softmax
        4. 提取 Top-3 预测结果
        5. 获取物种信息描述
        6. 构建预处理可视化 HTML
        7. 计时推理耗时

    参数:
        fp: 上传的文件路径（Gradio File 组件提供）
        model_name (str): 选中的模型名称
        lang (str): 当前语言 ("en" / "zh")
    返回:
        tuple: (预览图像, 结果HTML, 描述HTML, 预处理图像, 原始图像, 管线HTML)
    """
    global _LAST_FP
    # 处理文件路径: 切换模型时 fp 可能为 None，复用上次路径
    if fp is not None:
        _LAST_FP = fp
    elif _LAST_FP is not None:
        fp = _LAST_FP
    if fp is None:
        # 未上传任何文件: 返回空状态
        return None, ("<div style='color:#999;text-align:center;padding:3rem;'>"
                      "Upload to begin</div>"), "", None, None, ""

    # 确保模型已加载
    _ensure_model(model_name)
    # 加载原始 RGB 图像
    rgb = load_any_image(fp)
    # 生成预处理可视化图像（不含归一化步骤，因为归一化后为浮点数，不可直接显示）
    preproc_img = np.array(TF_VIZ(Image.fromarray(rgb)))

    # 构建预处理步骤可视化（分步显示在管线上）
    pil_for_pipe = Image.fromarray(rgb)
    step_gray = np.array(transforms.Grayscale(num_output_channels=3)(pil_for_pipe))
    step_resize = np.array(transforms.Resize((256, 256))(Image.fromarray(step_gray)))
    step_crop = np.array(transforms.CenterCrop(224)(Image.fromarray(step_resize)))
    pipeline_html = build_pipeline_html(rgb, step_gray, step_resize, step_crop)

    # 模型推理用完整预处理（含 ToTensor + Normalize）
    t = current_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEV)

    # ===== 带计时的推理 =====
    if DEV.type == "cuda":
        torch.cuda.synchronize()                     # 确保所有 CUDA 操作完成再计时
    t0 = time.perf_counter()                         # 高精度计时开始
    with torch.no_grad():
        # Softmax 归一化，输出各类别概率分布 (1, 73)
        p = F.softmax(current_model(t), 1)
    if DEV.type == "cuda":
        torch.cuda.synchronize()                     # 等待推理完成
    infer_ms = (time.perf_counter() - t0) * 1000     # 推理耗时（毫秒）

    # 提取 Top-3 预测结果
    # tp: Top-3 概率值, ti: Top-3 类别索引
    tp, ti = torch.topk(p, 3)
    # 奖牌 emoji: 金牌、银牌、铜牌
    emoji = ["\U0001F947", "\U0001F948", "\U0001F949"]
    rows = []
    for i in range(3):
        idx = ti[0, i].item()                        # 类别索引
        # 获取类别名称（兜底用 "Class X" 防止越界）
        nm = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        pc = tp[0, i].item() * 100                   # 概率百分比
        rows.append(
            f"<tr><td style='font-weight:700;width:36px;text-align:center;"
            f"font-size:1.2rem'>{emoji[i]}</td>"
            f"<td style='font-weight:600'>{nm}</td>"
            f"<td style='text-align:right;font-weight:700;color:#2d5016'>{pc:.1f}%</td></tr>"
        )
    # Top-3 结果表格
    tbl = f"<table style='width:100%;border-collapse:collapse'>{"".join(rows)}</table>"

    # 最佳预测结果展示
    top_name = CLASS_NAMES[ti[0, 0].item()]          # Top-1 花粉拉丁学名
    top_pct = tp[0, 0].item() * 100                  # Top-1 置信度
    # 绿色高亮徽章: 显示花粉名称 + 置信度
    badge = (
        f"<div style='display:flex;align-items:center;gap:12px;"
        f"margin-bottom:20px'>"
        f"<span style='display:inline-block;background:linear-gradient(135deg,#5a8a3c,#7ab648);"
        f"color:#fff;padding:10px 22px;border-radius:28px;font-weight:700;font-size:1.15rem'>"
        f"{top_name}</span>"
        f"<span style='font-size:1.3rem;font-weight:800;color:#2d5016'>{top_pct:.1f}%</span></div>"
    )

    # 推理速度标签
    speed_badge = (
        f"<div style='font-size:0.78rem;color:#8aab6e;text-align:right;"
        f"margin-bottom:8px'>Inference: {infer_ms:.1f}ms</div>"
    )

    # 获取花粉物种信息（中英文描述）
    info = POLLEN_INFO.get(top_name, {"zh": "暂无资料", "en": "No info available."})
    desc = (
        f"<div style='background:linear-gradient(135deg,#fafdf7,#f2f8ec);"
        f"border:1px solid #d8ecc8;border-radius:12px;padding:14px 18px;margin-top:8px'>"
        f"<div style='font-weight:700;color:#2d5016;margin-bottom:6px'>{top_name}</div>"
        f"<p style='font-size:0.88rem;color:#555;line-height:1.6;margin:0'>{info[lang]}</p></div>"
    )

    # 返回多组输出: 预览图, 结果, 物种信息, 预处理图, 原始图, 预处理管线
    return rgb, speed_badge + badge + tbl, desc, preproc_img, rgb, pipeline_html


# ===== 自定义 CSS 样式 =====
# 整体风格: 以绿色系为主题色，与花粉识别/植物学主题呼应
# 使用 Inter 字体，柔和的阴影和圆角，营造清新专业的视觉感受
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.gradio-container{font-family:'Inter',system-ui,sans-serif!important;max-width:1060px!important;margin:0 auto}
.header{text-align:center;padding:2.5rem 0 1rem}
.header h1{font-size:2.4rem;font-weight:800;color:#1a3a08;margin-bottom:0.1rem;letter-spacing:-0.02em}
.header .subtitle{font-size:1.05rem;color:#5a8a3c;font-weight:500}
.divider{height:3px;background:linear-gradient(90deg,#c8d8b8,#5a8a3c,#c8d8b8);margin:1rem 0 1.5rem;border-radius:2px}
.metrics{display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;padding:1.2rem 2rem;margin:0 0 1.5rem;background:linear-gradient(135deg,#fafdf7,#f2f8ec);border:1px solid #d8ecc8;border-radius:14px}
.met{text-align:center;min-width:100px}
.met-val{font-size:1.5rem;font-weight:800;color:#2d5016}
.met-lbl{font-size:0.75rem;color:#6b8a50;text-transform:uppercase;letter-spacing:0.06em;margin-top:2px}
.lang-toggle label{font-weight:600!important}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem}
.info-card{background:linear-gradient(135deg,#fafdf7,#f2f8ec);border:1px solid #d8ecc8;border-radius:14px;padding:1.2rem 1.5rem}
.info-card h4{font-size:0.95rem;font-weight:700;color:#2d5016;margin:0 0 0.6rem;text-transform:uppercase;letter-spacing:0.04em}
.info-card p{font-size:0.88rem;color:#555;line-height:1.6;margin:0}
.info-card ul{margin:0;padding-left:1.2rem;font-size:0.88rem;color:#555;line-height:1.7}
.info-card code{background:#e8f4dc;padding:1px 6px;border-radius:4px;font-size:0.82rem;color:#3d6b1e}
.footer{text-align:center;padding:2rem 0 1rem;color:#bbb;font-size:0.78rem;border-top:1px solid #eef3e8;margin-top:2rem}
.upload-box{border:2px dashed #c8d8b8!important;border-radius:16px!important;min-height:80px!important;max-height:120px!important}
.upload-box:hover{border-color:#7ab648!important}
.model-select label{font-weight:600!important;color:#2d5016!important;font-size:0.9rem!important}
.preproc-toggle label{font-size:0.82rem!important;color:#6b8a50!important}
.pipeline-row{background:linear-gradient(135deg,#fafdf7,#f2f8ec);border:1px solid #d8ecc8;border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.5rem}
.pipeline-title{font-size:0.85rem;font-weight:700;color:#2d5016;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;text-align:center}
.pipeline-flow{display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap}
.pipe-step{display:flex;flex-direction:column;align-items:center;gap:6px;padding:0 8px}
.pipe-step img{width:110px;height:110px;object-fit:cover;border-radius:8px;border:2px solid #d8ecc8;box-shadow:0 2px 6px rgba(0,0,0,0.04)}
.pipe-step span{font-size:0.72rem;color:#6b8a50;font-weight:600;text-transform:uppercase;letter-spacing:0.04em}
.pipe-step.pipe-text{min-width:90px}
.pipe-icon{width:66px;height:66px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#e8f4dc,#d8ecc8);border-radius:10px;font-size:1.2rem;color:#2d5016;font-weight:700}
.pipe-arrow{font-size:1.3rem;color:#b8d4a0;font-weight:700;padding:0 4px;margin-bottom:18px}
"""

# ===== 国际化文本 (i18n) =====
# 支持英文和中文两种语言切换
# 页面标题、模型介绍、数据集说明、特性列表、使用说明
TEXT = {
    "en": {
        "title": "Pollen Grain Recognition",
        "subtitle": "3-Model Comparison | 73 Species | POLLEN73S",
        "info_arch_title": "Models",
        "info_arch": ("<b>APFA-Net Wide</b> (2.92M) | <b>ResNet34+SE</b> (21.37M) | "
                      "<b>ViT-Small</b> (21.69M). CNN + CNN-Attention + Transformer comparison."),
        "info_data_title": "Dataset",
        "info_data": ("<b>POLLEN73S</b>: 73 pollen species, 2523 optical micrographs at "
                      "224x224. Train 1766 / Val 378 / Test 379."),
        "info_feat_title": "Key Features",
        "info_feat_1": "3-model comparison (CNN + CNN-Attention + Transformer)",
        "info_feat_2": "SE channel attention + ViT self-attention + Depthwise separable conv",
        "info_feat_3": "TIFF/JPG/PNG all supported",
        "info_feat_4": "Real-time inference ~5ms",
        "info_usage_title": "Usage",
        "info_usage": ("Select model, upload pollen micrograph. Auto Grayscale+Resize+Crop+Normalize."),
    },
    "zh": {
        "title": "花粉颗粒智能识别",
        "subtitle": "三模型对比 | 73个物种 | POLLEN73S基准",
        "info_arch_title": "模型",
        "info_arch": ("<b>APFA-Net Wide</b> (2.92M) | <b>ResNet34+SE</b> (21.37M) | "
                      "<b>ViT-Small</b> (21.69M)。CNN + CNN注意力 + Transformer 三模型对比。"),
        "info_data_title": "数据集",
        "info_data": ("<b>POLLEN73S</b>：73种花粉，2523张光学显微镜图像 224x224。"
                      "训练1766 / 验证378 / 测试379。"),
        "info_feat_title": "核心特性",
        "info_feat_1": "CNN + CNN注意力 + Transformer 三模型对比",
        "info_feat_2": "SE通道注意力 + ViT自注意力 + 可分离卷积",
        "info_feat_3": "支持TIFF/JPG/PNG全格式",
        "info_feat_4": "实时推理 ~5ms",
        "info_usage_title": "使用说明",
        "info_usage": "选择模型，上传花粉显微图像。自动灰度化+缩放+裁剪+归一化。"
    }
}

# 指标标签的国际化（页面上方的四格指标显示）
met_labels = {
    "en": ["Best Acc", "F1 Score", "Params", "Speed"],
    "zh": ["最佳准确率", "F1 分数", "参数量", "推理速度"]
}

def build_metrics_html(lang):
    """
    构建模型指标展示 HTML

    从全局 metrics 字典中提取四个关键指标并格式化展示:
        - test_accuracy: 测试准确率 (百分比)
        - test_f1: 宏观 F1 分数 (百分比)
        - params_M: 参数量 (百万)
        - inference_ms: 推理速度 (毫秒)

    参数:
        lang (str): 当前语言 ("en" / "zh")
    返回:
        str: 指标 HTML 字符串，若 metrics 为空则返回空字符串
    """
    if not metrics:
        return ""
    items = []
    keys = ["test_accuracy", "test_f1", "params_M", "inference_ms"]
    # 四个格式化函数分别对应四个指标
    fmts = [
        lambda v: f"{v*100:.1f}%",                   # 准确率: 0.95 -> 95.0%
        lambda v: f"{v*100:.1f}%",                   # F1: 0.95 -> 95.0%
        lambda v: f"{v:.1f}M",                       # 参数量: 21.35 -> 21.4M
        lambda v: f"{v:.1f}ms",                      # 推理速度: 5.1 -> 5.1ms
    ]
    for k, fmt, lbl in zip(keys, fmts, met_labels[lang]):
        if k in metrics:
            items.append(
                f"<div class='met'><div class='met-val'>{fmt(metrics[k])}</div>"
                f"<div class='met-lbl'>{lbl}</div></div>"
            )
    return f"<div class='metrics'>{"".join(items)}</div>" if items else ""


def build_info_section(lang):
    """
    构建页面底部信息卡片区 HTML

    四个卡片:
        - 模型架构介绍
        - 数据集说明
        - 核心特性列表
        - 使用说明

    参数:
        lang (str): 当前语言 ("en" / "zh")
    返回:
        str: 完整的四卡片 HTML 字符串
    """
    t = TEXT[lang]
    return f"""<div class='info-grid'>
<div class='info-card'><h4>{t["info_arch_title"]}</h4><p>{t["info_arch"]}</p></div>
<div class='info-card'><h4>{t["info_data_title"]}</h4><p>{t["info_data"]}</p></div>
<div class='info-card'><h4>{t["info_feat_title"]}</h4><ul><li>{t["info_feat_1"]}</li><li>{t["info_feat_2"]}</li><li>{t["info_feat_3"]}</li><li>{t["info_feat_4"]}</li></ul></div>
<div class='info-card'><h4>{t["info_usage_title"]}</h4><p>{t["info_usage"]}</p></div>
</div>"""


def update_lang(lang_state, new_lang):
    """
    语言切换回调函数

    当用户切换语言单选按钮时触发，更新页面上所有国际化文本:
        - metrics 标签文字
        - info section 内容

    参数:
        lang_state (str): 当前语言状态
        new_lang (str): 新选择的语言 (Gradio Radio 组件值)
    返回:
        tuple: (新语言, 新指标HTML, 新信息区HTML)
    """
    lang = "en" if new_lang == "EN" else "zh"
    return lang, build_metrics_html(lang), build_info_section(lang)


def toggle_view(show, orig, preproc):
    """
    预处理视图切换回调

    当用户勾选/取消 "Show model input view" 时切换预览显示:
        - 勾选: 显示预处理后的图像（模型实际输入视角）
        - 取消: 显示原始上传图像

    参数:
        show (bool): 是否显示预处理视图
        orig (np.ndarray): 原始图像
        preproc (np.ndarray): 预处理后的图像
    返回:
        np.ndarray: 要显示的图像
    """
    if show and preproc is not None:
        return preproc
    return orig


# ===== Gradio Web 界面构建 =====
# 使用 gr.Blocks 低级别 API 构建完全自定义的布局
# gr.themes.Soft(): Gradio 内置现代柔和主题
with gr.Blocks(title="Pollen Recognition", css=css, theme=gr.themes.Soft()) as demo:
    # 不可见状态变量: 用于在组件间传递数据而不渲染
    lang_state = gr.State("en")                      # 当前语言
    preproc_state = gr.State(None)                   # 预处理后的图像
    orig_state = gr.State(None)                      # 原始上传图像

    # 页面标题区域
    gr.HTML("""<div class='header'><h1>Pollen Grain Recognition</h1>
<div class='subtitle'>Multi-Model | 73 Species | POLLEN73S System</div></div>
<div class='divider'></div>""")

    # 语言切换 + 指标展示行
    with gr.Row():
        # 语言切换单选按钮（EN / 中文）
        lang_toggle = gr.Radio(choices=["EN", "中文"], value="EN", label="Language",
                               interactive=True, elem_classes="lang-toggle", scale=1)
        # 模型性能指标（Acc, F1, Params, Speed）
        metrics_display = gr.HTML(value=build_metrics_html("en"), scale=6)

    # 主内容区: 左侧上传/预览，右侧结果
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            # 图像上传组件：支持 JPG/PNG/TIFF
            upload_file = gr.File(label="Upload Image (JPG/PNG/TIFF)", file_types=["image"],
                                  elem_classes="upload-box")
            # 图像预览区
            preview_img = gr.Image(label="Preview", interactive=False, height=420)
            # 预处理视图切换复选框
            show_preprocess = gr.Checkbox(label="Show model input view", value=False,
                                          elem_classes="preproc-toggle")
        with gr.Column(scale=5):
            with gr.Group():
                # 模型选择下拉菜单
                model_selector = gr.Dropdown(
                    choices=["apfanet", "resnet34_se", "vit_small"],
                    value="vit_small", label="Select Model", interactive=True,
                    elem_classes="model-select")
                # 预测结果展示区（空状态: 提示上传）
                result_html = gr.HTML(
                    value="<div style='color:#999;text-align:center;padding:3rem;"
                          "font-size:1.05rem'>Upload a pollen micrograph to begin</div>")
                # 物种信息描述区
                desc_html = gr.HTML(value="")

    # 预处理管线可视化（页面底部）
    pipeline_html = gr.HTML(value="")
    # 模型/数据集/特性信息卡片
    info_section = gr.HTML(value=build_info_section("en"))

    # ===== 事件绑定 =====
    # 上传文件或切换模型时触发分类
    upload_file.change(fn=classify,
                       inputs=[upload_file, model_selector, lang_state],
                       outputs=[preview_img, result_html, desc_html,
                                preproc_state, orig_state, pipeline_html])
    # 模型切换时同样触发分类（复用上次上传的文件）
    model_selector.change(fn=lambda m, l: classify(None, m, l),
                          inputs=[model_selector, lang_state],
                          outputs=[preview_img, result_html, desc_html,
                                   preproc_state, orig_state, pipeline_html])
    # 预处理视图切换
    show_preprocess.change(fn=toggle_view,
                           inputs=[show_preprocess, orig_state, preproc_state],
                           outputs=[preview_img])
    # 语言切换
    lang_toggle.change(fn=update_lang,
                       inputs=[lang_state, lang_toggle],
                       outputs=[lang_state, metrics_display, info_section])

    # 页脚
    gr.HTML("""<div class='footer'>APFA-Net Wide | ResNet34+SE | ViT-Small \u00B7 PyTorch \u00B7 Gradio</div>""")


# ===== 应用启动入口 =====
if __name__ == "__main__":
    # 启动 Gradio 服务器:
    #   server_name="0.0.0.0": 监听所有网络接口（局域网可访问）
    #   server_port=7860: Gradio 默认端口
    demo.launch(server_name="0.0.0.0", server_port=7860)
