import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 配置区
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

EVM_DIR = os.path.join(PROJECT_ROOT, "data", "evm")
PATTERN = "evm*.bin"
USE_TRIM_95 = True          # 开关：True=启用去掉最大5%，False=不处理
ASSUME_DTYPE = np.float32   # 默认按 float32 读取（和你 cell 一致）
# =========================


def empirical_cdf(data: np.ndarray):
    data = np.asarray(data)
    data = data[np.isfinite(data)]  # 防 NaN/Inf
    if data.size == 0:
        return np.array([]), np.array([])
    data_sorted = np.sort(data)
    cdf = np.arange(1, data_sorted.size + 1) / data_sorted.size
    return data_sorted, cdf


def extract_label(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"^evm_?(.*)\.bin$", base, flags=re.IGNORECASE)
    if m:
        inner = m.group(1)
        return inner if inner else "default"
    return os.path.splitext(base)[0]



def read_evm_file(path: str, dtype=np.float32) -> np.ndarray:
    return np.fromfile(path, dtype=dtype).astype(float)


def maybe_trim_95(evm: np.ndarray, enable: bool):
    if not enable:
        return evm, None
    p95 = np.percentile(evm, 95)
    evm_trim = evm[evm <= p95]
    return evm_trim, p95


def main():
    files = sorted(glob.glob(os.path.join(EVM_DIR, PATTERN)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(EVM_DIR, PATTERN)}")

    plt.figure(figsize=(8, 5))

    # 如果开启 trim，我们也把每条曲线的 p95 记一下（可选：打印）
    p95_dict = {}

    for f in files:
        label = extract_label(f)

        # 读取
        evm = read_evm_file(f, dtype=ASSUME_DTYPE)

        # 基本信息（你也可以注释掉这些 print）
        print("=" * 60)
        print("file:", f)
        print("size (bytes):", os.path.getsize(f))
        print("num samples :", evm.size)
        print("min/max     :", float(np.min(evm)), float(np.max(evm)))
        print("mean/std    :", float(np.mean(evm)), float(np.std(evm)))

        # trim 开关
        evm_use, p95 = maybe_trim_95(evm, USE_TRIM_95)
        if USE_TRIM_95:
            p95_dict[label] = float(p95)
            print("95th percentile:", float(p95))
            print("kept ratio     :", evm_use.size / evm.size)

        # CDF
        x, cdf = empirical_cdf(evm_use)
        if x.size == 0:
            print("WARNING: empty/invalid data after filtering:", f)
            continue

        # 画到同一张图
        # y 轴用百分比显示（0~100）
        plt.plot(x, 100 * cdf, label=label)

    title = "EVM CDF (OTA)"
    if USE_TRIM_95:
        title += " - Trimmed (<=95th percentile)"

    plt.title(title)
    plt.xlabel("EVM (%)")
    plt.ylabel("CDF (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(EVM_DIR, "evm_cdf.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    # 可选：打印每个文件的 p95
    if USE_TRIM_95 and p95_dict:
        print("\nPer-curve 95th percentile thresholds:")
        for k in sorted(p95_dict.keys()):
            print(f"  {k}: {p95_dict[k]:.6f}")


if __name__ == "__main__":
    main()
