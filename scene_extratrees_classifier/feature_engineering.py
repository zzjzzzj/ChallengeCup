from __future__ import annotations

"""手工视觉特征提取。

针对 air / sea / urban / forest 场景分类，统一将图像转为灰度图并缩放到
128×128，然后提取 30 维特征：灰度统计、梯度和边缘、局部标准差、
LBP、GLCM 同质性以及频谱熵。
"""

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern


SCENES: Final[list[str]] = ["air", "sea", "urban", "forest"]
IMAGE_SIZE: Final[tuple[int, int]] = (128, 128)
LBP_POINTS: Final[int] = 8
LBP_RADIUS: Final[int] = 1
LBP_BINS: Final[int] = LBP_POINTS + 2  # uniform LBP 共 P+2 个取值

# 30 个特征。顺序会写入模型元数据，训练和推理必须保持一致。
FEATURE_NAMES: Final[list[str]] = [
    "int_std",
    "int_p70",
    "int_p75",
    "int_p80",
    "int_p85",
    "int_p90",
    "int_p95",
    "int_p97",
    "int_p99",
    "int_dynamic_range",
    "int_entropy",
    "grad_p90",
    "edge_density",
    "local_std_mean",
    "local_std_std",
    "local_std_p90",
    "lbp_entropy",
    *[f"lbp_hist_{i}" for i in range(LBP_BINS)],
    "glcm_homogeneity_d2",
    "glcm_homogeneity_d4",
    "spectrum_entropy",
]

if len(FEATURE_NAMES) != 30:
    raise RuntimeError(f"特征数量应为 30，当前为 {len(FEATURE_NAMES)}")


def _read_image(path: Path) -> np.ndarray:
    """读取图像，并兼容包含中文的 Windows 路径。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图片或格式不受支持: {path}")
    return image


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """转灰度并归一化到 [0, 1]，尽量保留不同图像的亮度差异。"""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[..., 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"不支持的图像形状: {image.shape}")

    original_dtype = gray.dtype
    gray = gray.astype(np.float32, copy=False)

    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        gray = gray / max_value
    else:
        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            raise ValueError("图像不包含有效像素")

        min_value = float(finite.min())
        max_value = float(finite.max())
        if 0.0 <= min_value and max_value <= 1.0:
            pass
        elif 0.0 <= min_value and max_value <= 255.0:
            gray = gray / 255.0
        else:
            # 浮点遥感图像可能不是固定量化范围，此时使用稳健归一化。
            low, high = np.percentile(finite, [0.5, 99.5])
            if high <= low:
                gray = np.zeros_like(gray, dtype=np.float32)
            else:
                gray = (gray - float(low)) / float(high - low)

    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.clip(gray, 0.0, 1.0)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32, copy=False)


def _entropy_from_probabilities(probabilities: np.ndarray, *, normalize: bool = False) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0

    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    if normalize and probabilities.size > 1:
        entropy /= float(np.log2(probabilities.size))
    return entropy


def _gray_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0
    return _entropy_from_probabilities(hist / total)


def _gradient_and_edges(gray: np.ndarray) -> tuple[float, float]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    grad_p90 = float(np.percentile(magnitude, 90))

    # 固定阈值可让不同图片间的边缘密度有可比性。
    edge_map = canny(gray, sigma=1.2, low_threshold=0.08, high_threshold=0.18)
    edge_density = float(edge_map.mean())
    return grad_p90, edge_density


def _local_std_features(gray: np.ndarray, window_size: int = 9) -> tuple[float, float, float]:
    kernel = (window_size, window_size)
    local_mean = cv2.blur(gray, kernel)
    local_square_mean = cv2.blur(gray * gray, kernel)
    variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(variance)
    return (
        float(local_std.mean()),
        float(local_std.std()),
        float(np.percentile(local_std, 90)),
    )


def _lbp_features(gray: np.ndarray) -> tuple[float, np.ndarray]:
    gray_u8 = np.round(gray * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(
        gray_u8,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    hist, _ = np.histogram(lbp, bins=np.arange(LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype(np.float64)
    hist /= max(float(hist.sum()), 1.0)
    entropy = _entropy_from_probabilities(hist)
    return entropy, hist.astype(np.float32)


def _glcm_homogeneity(gray: np.ndarray, distance: int) -> float:
    # 量化为 16 个灰度级，避免 256×256 GLCM 过于稀疏。
    quantized = np.floor(gray * 15.999).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[distance],
        angles=[0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
        levels=16,
        symmetric=True,
        normed=True,
    )
    values = graycoprops(matrix, "homogeneity")
    return float(values.mean())


def _spectrum_entropy(gray: np.ndarray) -> float:
    centered = gray.astype(np.float64) - float(gray.mean())
    power = np.abs(np.fft.fft2(centered)) ** 2
    power = np.fft.fftshift(power)
    flat = power.ravel()
    total = float(flat.sum())
    if total <= 1e-15:
        return 0.0

    probabilities = flat / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log2(positive)))
    # 按所有频率单元数归一化到约 [0, 1]。
    return entropy / float(np.log2(flat.size))


def extract_from_gray(gray: np.ndarray) -> dict[str, float]:
    """从已经归一化并缩放后的灰度图提取 30 维特征。"""
    if gray.shape != IMAGE_SIZE[::-1]:
        gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = np.clip(gray.astype(np.float32), 0.0, 1.0)

    percentiles = np.percentile(gray, [1, 70, 75, 80, 85, 90, 95, 97, 99])
    p1, p70, p75, p80, p85, p90, p95, p97, p99 = [float(x) for x in percentiles]

    grad_p90, edge_density = _gradient_and_edges(gray)
    local_mean, local_std, local_p90 = _local_std_features(gray)
    lbp_entropy, lbp_hist = _lbp_features(gray)

    features: dict[str, float] = {
        "int_std": float(gray.std()),
        "int_p70": p70,
        "int_p75": p75,
        "int_p80": p80,
        "int_p85": p85,
        "int_p90": p90,
        "int_p95": p95,
        "int_p97": p97,
        "int_p99": p99,
        "int_dynamic_range": p99 - p1,
        "int_entropy": _gray_entropy(gray),
        "grad_p90": grad_p90,
        "edge_density": edge_density,
        "local_std_mean": local_mean,
        "local_std_std": local_std,
        "local_std_p90": local_p90,
        "lbp_entropy": lbp_entropy,
        **{f"lbp_hist_{i}": float(value) for i, value in enumerate(lbp_hist)},
        "glcm_homogeneity_d2": _glcm_homogeneity(gray, 2),
        "glcm_homogeneity_d4": _glcm_homogeneity(gray, 4),
        "spectrum_entropy": _spectrum_entropy(gray),
    }

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"特征提取结果不完整: {missing}")
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_one(image_path: str | Path) -> dict[str, float]:
    """读取一张图片并返回按 FEATURE_NAMES 排序的特征字典。"""
    image = _read_image(Path(image_path))
    gray = _to_gray_float(image)
    return extract_from_gray(gray)


def feature_vector(image_path: str | Path) -> np.ndarray:
    """返回 shape=(30,) 的特征向量。"""
    features = extract_one(image_path)
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
