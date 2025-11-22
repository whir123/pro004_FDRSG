/**
 * 【批量随机生成面】
 * 随机数生成器 分辨率 物理边长 各向异性保持默认值不变
 * 高程落差取值在 5~30 范围内随机（整数）
 * 分形维数取值在 2.01~2.99 范围内随机（两位小数）
 * 随机种子取随机六位数
 *
 * 生成面命名为：序号-Exxx（落差值）-Dxxx（分形维数值）.stl
 * 同时输出一个与之对应的csv 以记录这些随机面的输入参数
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
// 导入生成面函数
import { generateSurface } from "./src/lib/srm.js";

// 沿用 stl.js 逻辑
function gridToSTL(z, Lx, Ly, name = "surface") {
  const ny = z.length;
  const nx = z[0].length;
  const dx = Lx / (nx - 1);
  const dy = Ly / (ny - 1);
  const lines = [`solid ${name}`];

  const tri = (p1, p2, p3) => {
    const ux = p2[0] - p1[0];
    const uy = p2[1] - p1[1];
    const uz = p2[2] - p1[2];

    const vx = p3[0] - p1[0];
    const vy = p3[1] - p1[1];
    const vz = p3[2] - p1[2];

    const nx_n = uy * vz - uz * vy;
    const ny_n = uz * vx - ux * vz;
    const nz_n = ux * vy - uy * vx;

    lines.push(`  facet normal ${nx_n} ${ny_n} ${nz_n}`);
    lines.push(`    outer loop`);
    lines.push(`      vertex ${p1[0]} ${p1[1]} ${p1[2]}`);
    lines.push(`      vertex ${p2[0]} ${p2[1]} ${p2[2]}`);
    lines.push(`      vertex ${p3[0]} ${p3[1]} ${p3[2]}`);
    lines.push(`    endloop`);
    lines.push(`  endfacet`);
  };

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const x0 = i * dx - Lx / 2;
      const x1 = (i + 1) * dx - Lx / 2;
      const y0 = j * dy - Ly / 2;
      const y1 = (j + 1) * dy - Ly / 2;

      const p00 = [x0, y0, z[j][i]];
      const p10 = [x1, y0, z[j][i + 1]];
      const p01 = [x0, y1, z[j + 1][i]];
      const p11 = [x1, y1, z[j + 1][i + 1]];

      tri(p00, p10, p11);
      tri(p00, p11, p01);
    }
  }

  lines.push(`endsolid ${name}`);
  return lines.join("\n");
}

//========== 小工具函数准备 ==========

// [min,max]之间的整数
function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// [min,max]之间的两位小数
function randFloat2(min, max) {
  const v = Math.random() * (max - min) + min;
  return Math.round(v * 100) / 100; // Math.round->四舍五入
}

// 左侧补零
const pad = (n, width) => String(n).padStart(width, "0");

// 根据 elevationDrop 把 Z 按峰谷差缩放成指定落差
function rescaleZToElevationDrop(Z, targetDrop) {
  let zmin = Infinity;
  let zmax = -Infinity;
  const ny = Z.length;
  const nx = Z[0].length;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = Z[j][i];
      if (v < zmin) zmin = v;
      if (v > zmax) zmax = v;
    }
  }
  const currentDrop = zmax - zmin || 1e-12;
  const scale = targetDrop / currentDrop;

  // 把表面居中到 0，上下各一半落差：[-drop/2, +drop/2]
  const center = (zmax + zmin) / 2;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      Z[j][i] = (Z[j][i] - center) * scale;
    }
  }
  // 返回实际落差，方便记录
  let newMin = Infinity;
  let newMax = -Infinity;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = Z[j][i];
      if (v < newMin) newMin = v;
      if (v > newMax) newMax = v;
    }
  }
  return newMax - newMin;
}

// ============ 主逻辑 生成 N 个 STL + CSV 参数表 ============
const __filename = fileURLToPath(import.meta.url);
// console.log(__filename); // 到此模块的完整 URL，包括查询参数和片段标识符（在 ? 和 # 之后） // xx\xx\xx\batchGenerateSTL.mjs
// CommonJS 里 __filename 是 Node 内置的全局变量，但在 ESModule 里没有，所以要用 import.meta.url + fileURLToPath 计算
const __dirname = path.dirname(__filename);
// console.log(__dirname); // path 是 Node 的路径工具库 // path.dirname(...) 会返回路径的目录部分 = 去掉最后那个文件名，只留下文件夹路径

// 输出目录
const OUTPUT_DIR = path.join(__dirname, "batch_stl_output");
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const N = 1000;

// CSV 头
const rows = [
  [
    "index",
    "filename",
    "elevationDrop",
    "D",
    "seed",
    "nx",
    "ny",
    "L",
    "anisotropy",
    "actualDrop",
  ].join(","),
];

for (let idx = 1; idx <= N; idx++) {
  // 1) 随机参数
  const elevationDrop = randInt(5, 30); // 5~30 整数
  const D = randFloat2(2.01, 2.99); // 2.01~2.99，两位小数
  const seed = randInt(100000, 999999); // 6位随机种子

  // 2) 调用现有生成函数
  const { Z, meta } = generateSurface({
    D,
    seed,
    // 不传 nx, ny, L, anisotropy, rngKind, sigma 等 -> 走默认
  });

  // 3) 用 elevationDrop 对 Z 缩放
  const actualDrop = rescaleZToElevationDrop(Z, elevationDrop);

  // 4) 命名规则：序号-E***-D***.stl
  //    这里：
  //    - 序号：4 位补零，如 0001
  //    - E：三位整数，如 E005, E023
  //    - D：把 D*100 取整，如 2.53 → D253
  const indexStr = pad(idx, 4);
  const E_str = pad(elevationDrop, 3);
  const D_str = pad(Math.round(D * 100), 3);

  const filename = `${indexStr}-E${E_str}-D${D_str}.stl`;
  const filepath = path.join(OUTPUT_DIR, filename);

  // 5) 生成 STL 文本并写文件
  const stlText = gridToSTL(Z, meta.L, meta.L, filename);
  fs.writeFileSync(filepath, stlText, "utf8");

  // 6) 把这一行参数写进 CSV
  rows.push(
    [
      idx,
      filename,
      elevationDrop,
      D.toFixed(2),
      seed,
      meta.nx,
      meta.ny,
      meta.L,
      meta.anisotropy,
      actualDrop,
    ].join(",")
  );

  // 进度提示
  if (idx % 50 === 0) {
    console.log(`Generated ${idx}/${N} surfaces...`);
  }
}

// 7) 写出参数表
const csvPath = path.join(OUTPUT_DIR, "surface_params.csv");
fs.writeFileSync(csvPath, rows.join("\n"), "utf8");

console.log(`Done! STL files + CSV saved in: ${OUTPUT_DIR}`);
