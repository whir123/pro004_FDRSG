<template>
  <!-- 最外层：两列 CSS Grid 布局：左列参数面板，右列预览 -->
  <div class="grid">
    <div class="panel">
      <h2 style="margin:0 0 8px">分形维数单元基底生成</h2>
      <hr />

      <!-- 随机数发生器 | 单选组 + 两个种子 -->
      <div class="panel" style="padding:10px; margin:10px 0">
        <div style="font-weight:600; margin-bottom:6px">随机数生成器</div>
        <div class="flex">
          <label class="checkbox"><input type="radio" name="rng" value="parkmiller" v-model="rng">Park&Miller</label>
          <label class="checkbox"><input type="radio" name="rng" value="baysdurham" v-model="rng">BaysDurham</label>
          <label class="checkbox"><input type="radio" name="rng" value="lecuyer" v-model="rng">L'Ecuyer</label>
        </div>
        <div class="row" style="margin-top:6px">
          <div><label>Seed</label><input type="number" v-model.number="seed1" /></div>
          <!-- <div><label>Seed 2</label><input type="number" v-model.number="seed2" /></div> -->
        </div>
      </div>

      <div class="panel" style="padding:10px; margin:10px 0">
        <!-- 分辨率选择器 -->
        <label>生成面分辨率</label>
        <!-- name="res" 让 5 个 radio 互斥 | v-model.number="res" 把字符值强制转成数字 -->
        <div class="flex">
          <label class="checkbox"><input type="radio" name="res" value="64" v-model.number="res"> 0064×0064 </label>
          <label class="checkbox"><input type="radio" name="res" value="128" v-model.number="res"> 0128×0128 </label>
          <label class="checkbox"><input type="radio" name="res" value="256" v-model.number="res"> 0256×0256 </label>
          <label class="checkbox"><input type="radio" name="res" value="512" v-model.number="res"> 0512×0512 </label>
          <label class="checkbox"><input type="radio" name="res" value="1024" v-model.number="res"> 1024×1024 </label>
        </div>

        <!-- 生成面配置参数 Parameters -->
        <!-- 部分旧参数废弃 -->
        <div class="row">
          <div><label>Physical size (mm)</label><input type="number" v-model.number="sizeMM" step="1" /></div>
          <div><label>Standard deviation (mm)</label><input type="number" v-model.number="stdMM" step="0.01" /></div>
        </div>
        <!-- <div class="row"> -->
          <!-- <div><label>Mismatch length (mm)</label><input type="number" v-model.number="mismatchMM" step="1" /></div> -->
          <!-- <div><label>Transition length (mm)</label><input type="number" v-model.number="transitionMM" step="1" /></div> -->
        <!-- </div> -->
        <div class="row">
          <div><label>Anisotropy factor</label><input type="number" v-model.number="aniso" step="0.01" /></div>
          <div><label>Fractal dimension</label><input type="number" v-model.number="D" step="0.01" /></div>
        </div>
        <!-- <div class="row"> -->
          <!-- <div><label>Max matching fraction</label><input type="number" v-model.number="mfmax" min="0" max="1" step="0.01" /></div> -->
          <!-- <div><label>Min matching fraction</label><input type="number" v-model.number="mfmin" min="0" max="1" step="0.01" /></div> -->
        <!-- </div> -->
      </div>

      <div class="panel" style="padding:10px; margin:10px 0">
        <div class="flex" style="margin-top:10px">
          <!-- 调用 run() 生成一次基底面 -->
          <button class="primary" @click="run">[ 生成结果 ]</button>
          <!-- 导出 Bottom 的 STL -->
          <button class="ghost" @click="exportSTL">[ 保存文件 ]</button>
          <!-- 显示分辨率、D、耗时等信息 -->
          <span class="badge">{{ info }}</span>
        </div>
      </div>
    </div>

    <div class="panel">
      <!-- 只有在 ZB（底面网格高程矩阵）生成后才渲染 -->
      <SurfaceViewer v-if="ZB" :z="ZB" :Lx="L" :Ly="L" label="Bottom" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { generateAUPG } from './lib/srm' // 核心算法（频域自仿射 + AUPG 匹配），返回 { Z, meta }。
import { gridToSTL } from './lib/stl'
import SurfaceViewer from './components/SurfaceViewer.vue'

// 部分响应式参数的默认值
const res = ref(64) // 分辨率
const rng = ref('parkmiller') // 随机数发生器类型
const seed1 = ref(799753397) // A 面随机种子
const seed2 = ref(737375979) // 噪声 N 的随机种子

const sizeMM = ref(100) // 物理尺寸（mm）
const mismatchMM = ref(15)   // ML （mm）
const transitionMM = ref(10) // TL （mm）
const stdMM = ref(10.0) // σz (mm)
const aniso = ref(1.0) // 各向异性 x相对于y
const D = ref(2.5) // 分形维数
const mfmax = ref(1.0) // 长波极限相关
const mfmin = ref(0.0) // 短波极限相关

const nx = computed(() => res.value)
const ny = computed(() => res.value)
const L = computed(() => sizeMM.value / 1000) // mm → m

const ZB = ref(null) // 生成的 Bottom 面高程矩阵（二维数组 ny × nx）
const info = ref('')

// ========= 把 z 重标定为标准差的指定范围 =========
function remapZ_toRangeZero(z, targetRange) {
  const ny = z.length, nx = z[0].length
  let zmin = Infinity, zmax = -Infinity
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = z[j][i]
    if (v < zmin) zmin = v
    if (v > zmax) zmax = v
  }
  const range0 = zmax - zmin
  if (!(targetRange > 0)) return z.map(row => row.slice())
  if (!(range0 > 0) || !isFinite(range0)) return z.map(row => row.map(() => 0))
  const mid0 = 0.5 * (zmax + zmin)
  const s = targetRange / range0
  const out = Array.from({ length: ny }, (_, j) => new Array(nx))
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out[j][i] = (z[j][i] - mid0) * s
  return out
}

// ========= 生成后立刻重标定 预览为“重标定后” =========
function run() {
  const t0 = performance.now()
  const { Z } = generateAUPG({
    nx: nx.value, ny: ny.value, L: L.value,
    D: D.value,
    sigma: stdMM.value / 1000, // SRM 阶段仍然按 σ 定标
    anisotropy: aniso.value,
    thetaDeg: 0, // 暂时没旋转需求 → 固定 0
    rngKind: rng.value,
    seed: seed1.value // 单 seed
    // 其余旧参数（mfmin/mfmax/TL/ML/seed2）已废弃
  })
  // 立刻按 “Standard deviation (mm)” 作为上下峰值范围进行重标定，并把中面移到 0
  const targetRange_m = (stdMM.value || 0) / 1000
  ZB.value = remapZ_toRangeZero(Z, targetRange_m)
  info.value = `${res.value}×${res.value} | D=${D.value.toFixed(2)} | ${(performance.now() - t0).toFixed(0)} ms`
}

// ========= 保存：仅导出一次最终结果 =========
function saveBlob(blob, filename) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

function exportSTL() {
  if (!ZB.value) return
  const finalBlob = gridToSTL(ZB.value, L.value, L.value, 'surface_bottom')
  saveBlob(finalBlob, 'surface_bottom.stl')
}
</script>

