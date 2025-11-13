<template>
  <div ref="wrap" style="width:100%; height:100%; position:relative">
    <!-- 3D/2D 视图切换，子组件负责具体渲染 -->
    <Surface3D v-if="mode==='3d'" ref="s3d" :z="z" :Lx="Lx" :Ly="Ly" />
    <Surface2D v-if="mode==='2d'" :z="z" :Lx="Lx" :Ly="Ly" />
    <div class="toolbar">
      <button v-if="mode==='3d'" @click="resetView" title="重置相机视角" style="width: 80px;">重置视角</button>
      <button :class="{ active: mode==='3d' }" @click="setMode('3d')">3D</button>
      <button :class="{ active: mode==='2d' }" @click="setMode('2d')">2D</button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import Surface2D from './Surface2D.vue'
import Surface3D from './Surface3D.vue'

const props = defineProps({
  z:  { type: Array,  required: true },  // ny x nx
  Lx: { type: Number, required: true },
  Ly: { type: Number, required: true },
});
const mode = ref('3d');
const s3d = ref(null);

// ---------------- 2D/3D 模式切换 ----------------
function setMode(m){
  if (m === mode.value) return;
  mode.value = m;
  if (mode.value === '2d') {
    nextTick();
  };
  if (mode.value === '3d') {
    nextTick(() => { s3d.value?.resetView?.() });
  };
}
function resetView() {
  if (mode.value !== '3d') return;
  s3d.value?.resetView?.();
};
</script>