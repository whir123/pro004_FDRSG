"""
从 STL 粗糙面计算表面分形维数 D | 批量版本（输出汇总 Excel）

步骤：
1) 读取 STL，表示为单值高度场 z=f(x,y)；
2) 将散点投影/插值到规则网格 Z(x,y)；
3) 去趋势（拟合并移除全局平面 ax+by+c）；
4) 2D FFT -> 各向同性功率谱 P(k)；
5) 在 log10 P(k) – log10 k 的中间频段线性拟合，得到斜率 s；
6) 依据自仿射关系： D = (8 + s) / 2；
7) 对读取到的 STL_LIST 中的多个 STL 逐个计算，并把结果写入一个 Excel。
"""

from pathlib import Path
import numpy as np
import trimesh
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Excel 导出
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ====================== 全局参数 =========================
HERE = Path(__file__).resolve().parent

# 指定要读取 STL 的目录
STL_DIR = HERE / "3_生成面"
if not STL_DIR.exists():
    raise FileNotFoundError(f"指定的 STL 目录不存在：{STL_DIR}")

# 读取该目录下所有 .stl 文件（按文件名排序），作为批量输入列表
STL_LIST = sorted(
    [p for p in STL_DIR.glob("*.stl") if p.is_file()],
    key=lambda x: x.name
)
if not STL_LIST:
    raise RuntimeError(f"在目录中没有找到任何 .stl 文件：{STL_DIR}")

# 网格分辨率（越高幂律段越稳，计算也更慢）
NX = 64
NY = 64

USE_WINDOW = True   # 是否使用 2D 汉宁窗
USE_DETREND = True  # 是否做平面去趋势
NBINS = 60          # 径向对数分箱的个数
FIT_LO = 0.15       # 拟合频段的下边界（按分位比例）
FIT_HI = 0.75       # 拟合频段的上边界（按分位比例）

# 汇总结果的 Excel 输出路径
OUT_EXCEL = HERE / "output.xlsx"
# ============================================================

# ------------------------- 工具函数 -------------------------
def load_stl_height(stl_path, nx=512, ny=512):
    """
    读取 STL 并采样为规则网格 (Xi, Yi, Zi)，同时返回物理尺寸 Lx, Ly。
    - 若 STL 包含多个子网面，则合并为一个整体；
    - 先用线性插值生成网格高度；若存在空洞，再以最近邻补齐。
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        # 将 Scene/多子网面合并
        mesh = mesh.dump().sum()
    v = mesh.vertices.astype(np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # 物理范围与网格
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    Lx, Ly = xmax - xmin, ymax - ymin

    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # 线性插值 + 最近邻补洞
    lin = LinearNDInterpolator(np.c_[x, y], z, fill_value=np.nan)
    Zi = lin(Xi, Yi)
    if np.isnan(Zi).any():
        nea = NearestNDInterpolator(np.c_[x, y], z)
        m = np.isnan(Zi)
        Zi[m] = nea(Xi[m], Yi[m])

    return Xi, Yi, Zi.astype(np.float64), float(Lx), float(Ly)

def detrend_plane(Z, Xi=None, Yi=None):
    """平面去趋势：最小二乘拟合 ax+by+c 并从 Z 中移除。"""
    ny, nx = Z.shape
    if Xi is None or Yi is None:
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    else:
        xx, yy = Xi, Yi
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(nx * ny)]
    coefs, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    trend = (A @ coefs).reshape(ny, nx)
    return Z - trend

def radial_psd(Z, Lx, Ly, window=True, detrend=True, nbins=60):
    """
    计算各向同性 PSD：2D FFT 后按半径对数分箱做环向平均。
    返回 (k_centers, Pk)。k 的单位为 rad/length，斜率与单位无关。
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - np.nanmean(Z)
    Z[~np.isfinite(Z)] = 0.0

    if detrend:
        Z = detrend_plane(Z)

    ny, nx = Z.shape
    if window:
        wy = np.hanning(ny)[:, None]
        wx = np.hanning(nx)[None, :]
        Z = Z * (wy * wx)

    # 2D FFT 与周期图估计
    F = np.fft.fft2(Z)
    S = (np.abs(F) ** 2) / (nx * ny)

    # 波数坐标（含 2π 因子）：单位 rad/length
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / (nx - 1))
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / (ny - 1))
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX ** 2 + KY ** 2)

    # 拉平成径向序列，并剔除 k=0 与非数
    kr = KR.ravel()
    sr = S.ravel()
    mask = np.isfinite(kr) & np.isfinite(sr) & (kr > 0)
    kr, sr = kr[mask], sr[mask]

    # 取 1%~99% 的范围做对数均匀分箱，避免极端值干扰
    kmin, kmax = np.percentile(kr, [1, 99])
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    idx = np.digitize(kr, edges) - 1

    Pk = np.zeros(nbins)
    kc = np.zeros(nbins)
    for i in range(nbins):
        m = idx == i
        if np.any(m):
            Pk[i] = np.mean(sr[m])
            kc[i] = np.exp(np.mean(np.log(kr[m])))  # 几何平均中心
        else:
            Pk[i] = np.nan
            kc[i] = np.nan

    good = np.isfinite(Pk) & np.isfinite(kc) & (Pk > 0) & (kc > 0)
    return kc[good], Pk[good]

def fit_midband(logk, logP, lo=0.15, hi=0.75):
    """在光谱的中间比例段 (lo, hi) 上做线性拟合，返回斜率/截距/R²。"""
    n = len(logk)
    i0 = max(int(np.floor(n * lo)), 0)
    i1 = min(int(np.ceil(n * hi)), n - 1)
    if i1 <= i0 + 2:
        i0, i1 = 1, max(n - 2, 2)

    x = logk[i0:i1]
    y = logP[i0:i1]
    A = np.c_[x, np.ones_like(x)]
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]

    yhat = s * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(s), float(b), float(r2)


def estimate_fractal_dimension_2dpsd(
    stl_path,
    nx=512,
    ny=512,
    window=True,
    detrend=True,
    nbins=60,
    fit_lo=0.15,
    fit_hi=0.75,
):
    """
    从单个 STL 以 2D-PSD 路线估计表面分形维数。
    返回一个 dict，包括 Lx/Ly、斜率 s、R² 与 D 等。
    """
    # 1) 读取 STL -> 规则网格
    Xi, Yi, Zi, Lx, Ly = load_stl_height(stl_path, nx=nx, ny=ny)

    # 2) 计算径向各向同性 PSD
    k, Pk = radial_psd(Zi, Lx, Ly, window=window, detrend=detrend, nbins=nbins)
    logk, logP = np.log10(k), np.log10(Pk)

    # 3) 在中间频段拟合斜率
    s, b, r2 = fit_midband(logk, logP, lo=fit_lo, hi=fit_hi)

    # 4) 2D 表面自仿射关系：D = (8 + s) / 2
    D = (8.0 + s) / 2.0

    return {
        "stl": str(stl_path),
        "nx": int(nx),
        "ny": int(ny),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "slope": float(s),
        "R2": float(r2),
        "D": float(D),
    }

# --------------------------- 主程序入口 ---------------------------
def main():
    if not HAS_PANDAS:
        raise RuntimeError("需要 pandas 才能导出 Excel，请先安装：pip install pandas openpyxl")

    results = []
    for idx, p in enumerate(STL_LIST, start=1):
        p = p.resolve()
        if not p.exists():
            print(f"[Warning] 第 {idx} 个文件不存在，跳过：{p}")
            continue

        print(f"[Info] ({idx}/{len(STL_LIST)}) 处理：{p}")
        res = estimate_fractal_dimension_2dpsd(
            p,
            nx=NX,
            ny=NY,
            window=USE_WINDOW,
            detrend=USE_DETREND,
            nbins=NBINS,
            fit_lo=FIT_LO,
            fit_hi=FIT_HI,
        )
        results.append(res)

    if not results:
        raise SystemExit("未成功处理任何 STL 文件，请检查 STL_LIST 中的路径。")

    # 按 STL_LIST 顺序写入 Excel
    rows = []
    for res in results:
        fname = Path(res["stl"]).name   # 如 "0988-E008-D229.stl"
        stem = Path(res["stl"]).stem    # 如 "0988-E008-D229"

        # 从文件名中解析 输入参数的D：Dxxx -> xxx/100
        D_true = np.nan
        try:
            parts = stem.split("-")
            # 找到以 D/d 开头的那一段（比如 "D229"）
            d_part = next(p for p in parts if p.startswith("D") or p.startswith("d"))
            D_true = int(d_part[1:]) / 100.0
        except Exception:
            # 解析失败就留 NaN
            D_true = np.nan

        rows.append(
            {
                "D_true": D_true,       # 第 1 列：文件名里解析出 输入参数 D”
                "D_psd": res["D"],      # 第 2 列：PSD 算法 -> D计算值
                "R2": res["R2"],        # 第 3 列：拟合 R²
                "file": fname,          # 第 4 列：文件名
            }
        )

    df = pd.DataFrame(rows)
    df.to_excel(OUT_EXCEL, index=False)
    print(f"[Info] 已将 {len(rows)} 个 STL 的结果写入：{OUT_EXCEL}")

    # 终端也简单输出一下
    print("---- PSD-based fractal dimension (2D surface) ----")
    for row in rows:
        print(
            f"{row['file']}: D_true={row['D_true']:.2f}, "
            f"D_psd={row['D_psd']:.3f}, R^2={row['R2']:.3f}"
        )

if __name__ == "__main__":
    main()
