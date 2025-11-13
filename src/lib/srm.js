// ===== RNGs =====
class ParkMiller {
  constructor(s = 1) {
    this.m = 2147483647;
    this.a = 16807;
    this.state = (Math.abs(Math.floor(s)) % (this.m - 1)) + 1;
  }
  next() {
    this.state = (this.a * this.state) % this.m;
    return this.state / this.m;
  }
}
class BaysDurham {
  constructor(s = 1, n = 32) {
    this.pm = new ParkMiller(s);
    this.n = n;
    this.tab = Array.from({ length: n }, () => this.pm.next());
  }
  next() {
    const j = Math.floor(this.pm.next() * this.n);
    const r = this.tab[j];
    this.tab[j] = this.pm.next();
    return r;
  }
}
class Lecuyer {
  constructor(s = 1) {
    this.IM1 = 2147483563;
    this.IM2 = 2147483399;
    this.AM = 1 / this.IM1;
    this.IA1 = 40014;
    this.IA2 = 40692;
    this.IQ1 = 53668;
    this.IQ2 = 52774;
    this.IR1 = 12211;
    this.IR2 = 3791;
    this.NTAB = 32;
    this.NDIV = Math.floor((this.IM1 - 1) / this.NTAB);
    this.EPS = 1.2e-7;
    this.RNMX = 1.0 - this.EPS;
    let id = Math.abs(Math.floor(s)) + 1;
    this.idum2 = 123456789;
    this.iy = 0;
    this.iv = new Array(this.NTAB).fill(0);
    for (let j = this.NTAB + 7; j >= 0; j--) {
      let k = Math.floor(id / this.IQ1);
      id = this.IA1 * (id - k * this.IQ1) - k * this.IR1;
      if (id < 0) id += this.IM1;
      if (j < this.NTAB) this.iv[j] = id;
    }
    this.iy = this.iv[0];
    this.idum = id;
  }
  next() {
    let k = Math.floor(this.idum / this.IQ1);
    this.idum = this.IA1 * (this.idum - k * this.IQ1) - k * this.IR1;
    if (this.idum < 0) this.idum += this.IM1;
    let k2 = Math.floor(this.idum2 / this.IQ2);
    this.idum2 = this.IA2 * (this.idum2 - k2 * this.IQ2) - k2 * this.IR2;
    if (this.idum2 < 0) this.idum2 += this.IM2;
    const j = Math.floor(this.iy / this.NDIV);
    this.iy = this.iv[j] - this.idum2;
    this.iv[j] = this.idum;
    if (this.iy < 1) this.iy += this.IM1 - 1;
    let t = this.AM * this.iy;
    return t > this.RNMX ? this.RNMX : t;
  }
}
function makeRNG(kind, seed) {
  switch ((kind || "parkmiller").toLowerCase()) {
    case "parkmiller":
      return new ParkMiller(seed);
    case "bays-durham":
    case "baysdurham":
      return new BaysDurham(seed);
    case "l'ecuyer":
    case "lecuyer":
      return new Lecuyer(seed);
    default:
      return new ParkMiller(seed);
  }
}

// ===== utils =====
const create2D = (ny, nx, v = 0) =>
  Array.from({ length: ny }, () => new Array(nx).fill(v));

const lin = (n, s) => {
  const a = new Array(n),
    h = Math.floor(n / 2);
  for (let i = 0; i < n; i++) a[i] = (i <= h ? i : i - n) * s;
  return a;
};

function rotk2(kx, ky, th, ax, ay) {
  const c = Math.cos(th),
    s = Math.sin(th);
  const xr = c * kx + s * ky;
  const yr = -s * kx + c * ky;
  const xa = xr / ax,
    ya = yr / ay;
  return xa * xa + ya * ya;
}

function enforceHermitian(Re, Im) {
  const ny = Re.length,
    nx = Re[0].length;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const jj = (ny - j) % ny,
        ii = (nx - i) % nx;
      if (j > 0 || i > 0) {
        Re[jj][ii] = Re[j][i];
        Im[jj][ii] = -Im[j][i];
      }
    }
  }
  Re[0][0] = 0;
  Im[0][0] = 0; // 零均值
}

function rescale(z, targetStd) {
  const ny = z.length,
    nx = z[0].length;
  let s = 0,
    s2 = 0,
    n = nx * ny;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      s += z[j][i];
      s2 += z[j][i] * z[j][i];
    }
  const m = s / n,
    sd = Math.sqrt(Math.max(1e-20, s2 / n - m * m));
  const r = targetStd / (sd || 1);
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) z[j][i] = (z[j][i] - m) * r;
  return z;
}

// ===== radix-2 FFT/IFFT =====
function bitrev(n) {
  const r = new Array(n),
    b = Math.log2(n) | 0;
  for (let i = 0; i < n; i++) {
    let x = i,
      y = 0;
    for (let k = 0; k < b; k++) {
      y = (y << 1) | (x & 1);
      x >>= 1;
    }
    r[i] = y;
  }
  return r;
}
function fft1d(re, im, inv = false) {
  const n = re.length,
    rev = bitrev(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inv ? -2 : 2) * Math.PI) / len;
    const wr = Math.cos(ang),
      wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let ur = 1,
        ui = 0;
      for (let j = 0; j < len >> 1; j++) {
        const ar = re[i + j],
          ai = im[i + j];
        const br = re[i + j + (len >> 1)] * ur - im[i + j + (len >> 1)] * ui;
        const bi = re[i + j + (len >> 1)] * ui + im[i + j + (len >> 1)] * ur;
        re[i + j] = ar + br;
        im[i + j] = ai + bi;
        re[i + j + (len >> 1)] = ar - br;
        im[i + j + (len >> 1)] = ai - bi;
        const tr = ur * wr - ui * wi;
        ui = ur * wi + ui * wr;
        ur = tr;
      }
    }
  }
  if (inv) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}
function ifft2(Re, Im) {
  const ny = Re.length,
    nx = Re[0].length;
  const tRe = create2D(ny, nx),
    tIm = create2D(ny, nx);
  for (let j = 0; j < ny; j++) {
    const re = Re[j].slice(),
      im = Im[j].slice();
    fft1d(re, im, true);
    for (let i = 0; i < nx; i++) {
      tRe[j][i] = re[i];
      tIm[j][i] = im[i];
    }
  }
  const z = create2D(ny, nx),
    reCol = new Array(ny),
    imCol = new Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      reCol[j] = tRe[j][i];
      imCol[j] = tIm[j][i];
    }
    fft1d(reCol, imCol, true);
    for (let j = 0; j < ny; j++) z[j][i] = reCol[j];
  }
  return z;
}

// ===== 单面 SRM 生成 =====
function makeSpectrumSingle(nx, ny, Lx, Ly, H, ax, ay, thetaDeg, rng) {
  const th = (thetaDeg * Math.PI) / 180;
  const dkx = (2 * Math.PI) / Lx,
    dky = (2 * Math.PI) / Ly;
  const kx = lin(nx, dkx),
    ky = lin(ny, dky);
  const Re = create2D(ny, nx),
    Im = create2D(ny, nx);
  const alpha = H + 1; // 振幅 ~ k^{-(H+1)} = k^{-(4-D)}
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k2 = rotk2(kx[i], ky[j], th, ax, ay);
      if (k2 < 1e-30) continue; // 跳过 DC
      const k = Math.sqrt(k2);
      const amp = Math.pow(k, -alpha);
      const phi = 2 * Math.PI * (rng ? rng.next() : Math.random());
      Re[j][i] = amp * Math.cos(phi);
      Im[j][i] = amp * Math.sin(phi);
    }
  }
  enforceHermitian(Re, Im);
  return { Re, Im };
}

/**
 * 生成单张自仿射粗糙表面（经典 SRM）
 * @param {Object} opts
 *  - nx, ny: 网格
 *  - L: 物理尺寸（正方域）
 *  - D: 分形维数 (2~3)
 *  - sigma: 目标 RMS 高度
 *  - anisotropy: X/Y 各向异性（ax = anisotropy, ay = 1）
 *  - thetaDeg: 旋转角（度）；若无需角度，设为 0
 *  - rngKind: "parkmiller" | "baysdurham" | "lecuyer"
 *  - seed: 随机种子
 * @returns {{ Z: number[][], meta: Object }}
 */
export function generateAUPGSingle({
  nx = 512,
  ny = 512,
  L = 0.1,
  D = 2.1,
  sigma = 0.01,
  anisotropy = 1.0,
  thetaDeg = 0,
  rngKind = "parkmiller",
  seed = 799753397,
} = {}) {
  const H = 3 - D;
  const ax = anisotropy,
    ay = 1.0;
  const Lx = L,
    Ly = L;
  const rng = makeRNG(rngKind, seed);

  const { Re, Im } = makeSpectrumSingle(
    nx,
    ny,
    Lx,
    Ly,
    H,
    ax,
    ay,
    thetaDeg,
    rng
  );
  let Z = ifft2(Re, Im);
  Z = rescale(Z, sigma);

  return {
    Z,
    meta: { nx, ny, L, D, H, sigma, anisotropy, thetaDeg, rngKind, seed },
  };
}

export const generateAUPG = (opts) => generateAUPGSingle(opts);