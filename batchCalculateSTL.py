"""
从 STL 粗糙面计算表面分形维数 D | 批量版本（输出汇总 Excel）

【保证】：
- 输入 STL 是单值高度场 z=f(x,y)
- 网格尽量是生成时的 64×64 | 否则二次插值引入谱形变
- 无整体倾角 | 否则开启 detrend
- 各向同性 | 可做径向平均 PSD

【步骤】：
1) STL -> 规则网格 Z(x,y)（优先直接回填 不插值）
2) 2D FFT -> 径向平均 PSD：P(k)
3) 在指定 k 带宽内拟合 logP(k)-logk 的斜率 s
4) D = (8 + s) / 2
"""

from pathlib import Path
import numpy as np
import trimesh
# fallback 插值 | 无法直接回填网格时采用
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
# Excel 导出
import pandas as pd

# ====================== 全局参数 =========================
HERE = Path(__file__).resolve().parent
STL_DIR = HERE / "../3_生成面" # 读取 STL 的目录
OUT_EXCEL = HERE / "../output.xlsx"

# 网格分辨率
NX = 64
NY = 64

USE_DETREND = False   # 保证无倾角 => 关闭平面去趋势
USE_WINDOW  = False   # 周期面一般不需要窗 若边界不连续再开
NBINS = 30            # 把波数模长|k|=sqrt(kx^2+ky^2)分成多少个径向分箱 | 64×64建议25~35

# 拟合带宽：用 kmin + Nyquist 来选
# kmin = 2π/L, knyq = π/dx
FIT_KLO_MULT = 3.0    # k_lo = 3*kmin（避开域尺度/低频）
FIT_KHI_MULT = 0.6    # k_hi = 0.6*knyq（避开高频端/插值/离散化 roll-off）
# ========================================================


"""
==========【读 STL -> 网格】==========
【优先】：从 STL 顶点直接重建规则网格（避免插值改变谱）
【失败】：退回 LinearNDInterpolator + NearestNDInterpolator
【返回】：Xi, Yi, Z, dx, dy | dx,dy 为网格步长（用于 k 轴计算）
"""
def load_stl_height(stl_path: Path, nx=64, ny=64, decimals=10):
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    v = mesh.vertices.astype(np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # 尝试直接回填规则网格
    xr = np.round(x, decimals=decimals)
    yr = np.round(y, decimals=decimals)

    xu = np.unique(xr)  # 默认已排序
    yu = np.unique(yr)

    if len(xu) == nx and len(yu) == ny:
        ix = np.searchsorted(xu, xr)
        iy = np.searchsorted(yu, yr)

        sumZ = np.zeros((ny, nx), dtype=np.float64)
        cntZ = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(sumZ, (iy, ix), z)
        np.add.at(cntZ, (iy, ix), 1.0)

        Z = sumZ / np.maximum(cntZ, 1.0)
        Xi, Yi = np.meshgrid(xu, yu)

        dx = float(np.median(np.diff(xu)))
        dy = float(np.median(np.diff(yu)))
        return Xi, Yi, Z, dx, dy

    # fallback：插值（尽量不用）
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xi = np.linspace(xmin, xmax, nx, endpoint=False)
    yi = np.linspace(ymin, ymax, ny, endpoint=False)
    Xi, Yi = np.meshgrid(xi, yi)

    lin = LinearNDInterpolator(np.c_[x, y], z, fill_value=np.nan)
    Zi = lin(Xi, Yi)
    if np.isnan(Zi).any():
        nea = NearestNDInterpolator(np.c_[x, y], z)
        m = np.isnan(Zi)
        Zi[m] = nea(Xi[m], Yi[m])

    dx = float(np.median(np.diff(xi)))
    dy = float(np.median(np.diff(yi)))
    return Xi, Yi, Zi.astype(np.float64), dx, dy


"""
==========【平面去趋势】==========
最小二乘拟合 ax+by+c 并从 Z 中移除
"""
def detrend_plane(Z, Xi=None, Yi=None):
    ny, nx = Z.shape
    if Xi is None or Yi is None:
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    else:
        xx, yy = Xi, Yi
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(nx * ny)]
    coefs, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    trend = (A @ coefs).reshape(ny, nx)
    return Z - trend


"""
==========【PSD计算】==========
2D FFT -> periodogram -> 径向对数分箱平均，得到各向同性 PSD: P(k)
k 的单位：rad/length（由 dx,dy 决定）
"""
def radial_psd(Z, dx, dy, window=False, detrend=False, Xi=None, Yi=None, nbins=30):
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - np.mean(Z)

    if detrend:
        Z = detrend_plane(Z, Xi=Xi, Yi=Yi)

    ny, nx = Z.shape

    if window:
        wy = np.hanning(ny)[:, None]
        wx = np.hanning(nx)[None, :]
        Z = Z * (wy * wx)

    F = np.fft.fft2(Z)
    S = (np.abs(F) ** 2) / (nx * ny)

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX**2 + KY**2)

    kr = KR.ravel()
    sr = S.ravel()
    mask = (kr > 0) & np.isfinite(sr) & (sr > 0)
    kr, sr = kr[mask], sr[mask]

    # 对数均匀分箱
    kmin, kmax = np.min(kr), np.max(kr)
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    idx = np.digitize(kr, edges) - 1

    Pk = np.full(nbins, np.nan, dtype=np.float64)
    kc = np.full(nbins, np.nan, dtype=np.float64)

    for i in range(nbins):
        m = idx == i
        if np.any(m):
            Pk[i] = np.mean(sr[m])
            kc[i] = np.exp(np.mean(np.log(kr[m])))

    good = np.isfinite(Pk) & np.isfinite(kc) & (Pk > 0) & (kc > 0)
    return kc[good], Pk[good]


"""==========【拟合（按 k 带宽）】=========="""
def fit_band_by_k(k, Pk, k_lo, k_hi):
    m = (k >= k_lo) & (k <= k_hi) & (Pk > 0)
    if np.count_nonzero(m) < 6:
        raise RuntimeError(
            f"可拟合点太少：count={np.count_nonzero(m)}，"
            f"请放宽 k_lo/k_hi 或减少 NBINS"
        )

    x = np.log10(k[m])
    y = np.log10(Pk[m])

    A = np.c_[x, np.ones_like(x)]
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]

    yhat = s * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(s), float(b), float(r2), int(np.count_nonzero(m))


"""==========【单文件估计】=========="""
def estimate_fractal_dimension_2dpsd(
    stl_path: Path,
    nx=64,
    ny=64,
    window=False,
    detrend=False,
    nbins=30,
    fit_klo_mult=3.0,
    fit_khi_mult=0.6,
):
    Xi, Yi, Z, dx, dy = load_stl_height(stl_path, nx=nx, ny=ny)

    k, Pk = radial_psd(
        Z, dx, dy,
        window=window,
        detrend=detrend,
        Xi=Xi, Yi=Yi,
        nbins=nbins
    )

    # 物理带宽选择
    Lx_eff = nx * dx
    Ly_eff = ny * dy
    kmin = 2 * np.pi / min(Lx_eff, Ly_eff)
    knyq = np.pi / min(dx, dy)

    k_lo = fit_klo_mult * kmin
    k_hi = fit_khi_mult * knyq

    s, b, r2, nfit = fit_band_by_k(k, Pk, k_lo, k_hi)
    D = (8.0 + s) / 2.0

    return {
        "stl": str(stl_path),
        "nx": int(nx), "ny": int(ny),
        "dx": float(dx), "dy": float(dy),
        "Lx_eff": float(Lx_eff), "Ly_eff": float(Ly_eff),
        "k_lo": float(k_lo), "k_hi": float(k_hi),
        "n_fit": int(nfit),
        "slope": float(s),
        "R2": float(r2),
        "D": float(D),
    }


"""==========【主程序】=========="""
def main():
    if not STL_DIR.exists():
        raise FileNotFoundError(f"指定的 STL 目录不存在：{STL_DIR}")

    stl_list = sorted([p for p in STL_DIR.glob("*.stl") if p.is_file()], key=lambda x: x.name)
    if not stl_list:
        raise RuntimeError(f"在目录中没有找到任何 .stl 文件：{STL_DIR}")

    rows = []
    for idx, p in enumerate(stl_list, start=1):
        print(f"[Info] ({idx}/{len(stl_list)}) 处理：{p.name}")

        res = estimate_fractal_dimension_2dpsd(
            p,
            nx=NX, ny=NY,
            window=USE_WINDOW,
            detrend=USE_DETREND,
            nbins=NBINS,
            fit_klo_mult=FIT_KLO_MULT,
            fit_khi_mult=FIT_KHI_MULT,
        )

        # 从文件名解析 D_true：Dxxx -> xxx/100
        D_true = np.nan
        try:
            stem = p.stem
            parts = stem.split("-")
            d_part = next(t for t in parts if t.startswith("D") or t.startswith("d"))
            D_true = int(d_part[1:]) / 100.0
        except Exception:
            D_true = np.nan

        rows.append({
            "D_true": D_true,
            "D_psd": res["D"],
            "slope": res["slope"],
            "R2": res["R2"],
            "n_fit": res["n_fit"],
            "k_lo": res["k_lo"],
            "k_hi": res["k_hi"],
            "dx": res["dx"],
            "file": p.name,
        })

    df = pd.DataFrame(rows)
    df.to_excel(OUT_EXCEL, index=False)
    print(f"[Info] 已将 {len(rows)} 个 STL 的结果写入：{OUT_EXCEL}")

    print("---- PSD-based fractal dimension (2D surface) ----")
    for row in rows:
        print(
            f"{row['file']}: D_true={row['D_true']:.2f}, "
            f"D_psd={row['D_psd']:.3f}, slope={row['slope']:.3f}, "
            f"R^2={row['R2']:.3f}, n_fit={row['n_fit']}"
        )

if __name__ == "__main__":
    main()