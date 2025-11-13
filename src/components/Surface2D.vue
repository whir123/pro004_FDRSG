<template>
  <!-- 2D 像素图画布 -->
  <canvas ref="grayCanvas" style="width:715px; height:715px"></canvas>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'

const props = defineProps({
  z:  { type: Array,  required: true },  // ny x nx
  Lx: { type: Number, required: true },
  Ly: { type: Number, required: true },
})

const grayCanvas = ref(null);

// ---------------- 2D 部分 ----------------
function zMinMax(z){
  let mn=Infinity, mx=-Infinity;
  const ny=z.length, nx=z[0].length;
  for(let j=0;j<ny;j++) {
    for(let i=0;i<nx;i++){ 
      const v=z[j][i];
      if(v<mn) mn=v;
      if(v>mx) mx=v;
    };
  };
  return { mn, mx, range: mx - mn }
}

function draw2D() {
  if (!grayCanvas.value) return;
  const ctx = grayCanvas.value.getContext('2d');

  // 读取 CSS 尺寸（715 兜底）
  let W = grayCanvas.value.clientWidth;
  let H = grayCanvas.value.clientHeight;
  if (!W || !H) {
    const cs = getComputedStyle(grayCanvas.value)
    W = parseInt(cs.width)  || 715;
    H = parseInt(cs.height) || 715;
  };
  // 设置画布实际像素尺寸
  if (grayCanvas.value.width !== W) grayCanvas.value.width = W;
  if (grayCanvas.value.height !== H) grayCanvas.value.height = H;
  const ny = props.z.length, nx = props.z[0].length;

  // 先在离屏 nx×ny 画布生成像素
  const off = document.createElement('canvas');
  off.width = nx; off.height = ny;
  const octx = off.getContext('2d');
  const img = octx.createImageData(nx, ny);
  const buf = img.data;

  const { mn, range } = zMinMax(props.z);
  const safeRange = (range && isFinite(range)) ? range : 1;
  const toGray = (v) => {
    if (!range || !isFinite(range)) return 128 // 常数面：中灰
    const g = (v - mn) / safeRange;
    const t = Math.max(0, Math.min(1, g));
    return Math.round(255 * t);
  };
  // 写每个栅格像素的 RGBA（灰度 v、alpha=255）
  for (let j=0;j<ny;j++){
    for (let i=0;i<nx;i++){
      const v = toGray(props.z[j][i])
      const p = (j*nx + i) * 4
      buf[p+0] = v
      buf[p+1] = v
      buf[p+2] = v
      buf[p+3] = 255
    }
  }
  octx.putImageData(img, 0, 0)

  // 最近邻放大贴到可见画布
  ctx.save();
  ctx.imageSmoothingEnabled = false
  ctx.clearRect(0,0,W,H);

  // 缩放 居中（偏右上）
  const scale = 0.85;
  const offsetX = (W * (1.0 - scale)) / 2;
  const offsetY = (H * (1.0 - scale)) / 2;
  const amendX = W*0.025; // 加上标尺后 整体修正位移
  const amendY = -H*0.02;
  ctx.translate(offsetX+amendX, offsetY+amendY);
  ctx.scale(scale, scale);

  ctx.drawImage(off, 0, 0, W, H);

  // --- 绘制白色网格（仅64*64和128*128时显示） ---
  if ((nx === 64 && ny === 64) || (nx === 128 && ny === 128)) {
    ctx.save();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 0.5;
    const cellW = W / nx;
    const cellH = H / ny;
    // 画竖线
    for (let i = 0; i <= nx; i++) {
      const x = Math.round(i * cellW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    // 画横线
    for (let j = 0; j <= ny; j++) {
      const y = Math.round(j * cellH) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  // --- 绘制XY轴刻度标尺 ---
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0); // 保证标尺不受缩放影响
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1;

  // 标尺参数
  const x0 = offsetX+amendX, x1 = W-offsetX+amendX;
  const y0 = offsetY+amendY, y1 = H-offsetY+amendY;
  const tickLen = 8;
  const nxTick = Math.min(nx, 5);
  const nyTick = Math.min(ny, 5);

  // X轴标尺（底部）
  for (let i = 0; i <= nxTick; i++) {
    const frac = i / nxTick;
    const x = x0 + frac * (x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x, y1 + 4);
    ctx.lineTo(x, y1 + 4 + tickLen);
    ctx.stroke();
    // 标注
    const label = (frac * props.Lx).toFixed(2);
    ctx.fillText(label, x - 18, y1 + 4 + tickLen + 20);
  }
  // Y轴标尺（左侧）
  for (let j = 0; j <= nyTick; j++) {
    const frac = j / nyTick;
    const y = y1 - frac * (y1 - y0);
    ctx.beginPath();
    ctx.moveTo(x0 - 4, y);
    ctx.lineTo(x0 - 4 - tickLen, y);
    ctx.stroke();
    // 标注
    const label = (frac * props.Ly).toFixed(2);
    ctx.fillText(label, x0 - 4 - tickLen - 48, y + 8);
  }
  // 轴名
  // ctx.font = "bold 13px sans-serif";
  // ctx.fillStyle = "#fff";
  // ctx.fillText("X (m)", (x0 + x1) / 2 - 18, y1 + 4 + tickLen + 32);
  // ctx.save();
  // ctx.translate(x0 - 4 - tickLen - 44, (y0 + y1) / 2 + 16);
  // ctx.rotate(-Math.PI / 2);
  // ctx.fillText("Y (m)", 0, 0);
  // ctx.restore();

  // ctx.restore();
}

onMounted(async () => {
  await nextTick();
  requestAnimationFrame(() => draw2D());
});
onBeforeUnmount(() => {});
watch(() => props.z, async () => {
  await nextTick();
  requestAnimationFrame(() => draw2D());
});
</script>
