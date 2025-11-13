<template>
  <!-- 3D WebGL 画布 -->
  <canvas ref="glCanvas" style="width:715px; height:715px"></canvas>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const props = defineProps({
  z:  { type: Array,  required: true },  // ny x nx
  Lx: { type: Number, required: true },
  Ly: { type: Number, required: true },
})

const glCanvas = ref(null);

// ---------------- 3D 部分 ----------------
let renderer, scene;
let camera3D, activeCamera;
let controls, mesh3D, gridHelper;
let raf = 0;

function buildGeometry3D() {
  const ny = props.z.length, nx = props.z[0].length;
  const geo = new THREE.PlaneGeometry(props.Lx, props.Ly, nx - 1, ny - 1);
  const pos = geo.attributes.position;
  for (let j=0;j<ny;j++) {
    for (let i=0;i<nx;i++) {
      pos.setZ(j*nx+i, props.z[j][i]);
    };
  };
  pos.needsUpdate = true;
  geo.computeVertexNormals()
  return geo;
};

function init3D() {
  // 创建场景 设置背景
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x222222)

  // 高亮旋转中心球体
  let highlightSphere = null;
  let isRotating = false;

  renderer = new THREE.WebGLRenderer({ canvas: glCanvas.value, antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio);
  const w = glCanvas.value.clientWidth || 715;
  const h = glCanvas.value.clientHeight || 715;
  renderer.setSize(w, h);

  camera3D = new THREE.PerspectiveCamera(50, w / h, 0.01, 2000);
  const maxL = Math.max(props.Lx, props.Ly);
  const fit = (maxL) / (2 * Math.tan(THREE.MathUtils.degToRad(camera3D.fov) / 2));
  const dist = fit * 0.9;
  // 初始相机位置
  camera3D.position.set(0, -dist, dist*1.4);
  camera3D.up.set(0,0,1); camera3D.lookAt(0,0,0);
  activeCamera = camera3D;

  // 鼠标交互设置
  controls = new OrbitControls(camera3D, renderer.domElement)
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enableZoom = true;
  controls.enablePan = true; // 允许右键平移
  controls.target.set(0,0,0);
  controls.update();


  const geo3D = buildGeometry3D();
  const mat3D = new THREE.MeshStandardMaterial({
    color: 0xdddddd, metalness: 0.0, roughness: 0.95, flatShading: true, side: THREE.DoubleSide
  });
  mesh3D = new THREE.Mesh(geo3D, mat3D);
  scene.add(mesh3D);

  gridHelper = new THREE.GridHelper(maxL * 1.2, 10);
  gridHelper.rotateX(Math.PI / 2);
  scene.add(gridHelper);


  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(1, -1, 2);
  scene.add(dir);

  const loop = () => {
    controls.update()
    renderer.render(scene, activeCamera)
    raf = requestAnimationFrame(loop)
  };
  loop();

  // ========== 鼠标缩放基于鼠标位置 ==========
  renderer.domElement.addEventListener('wheel', function (event) {
    event.preventDefault();
    // 获取鼠标在canvas内的归一化坐标（-1~1）
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    // 屏幕点反投影到世界坐标
    const mouse = new THREE.Vector2(x, y);
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera3D);
    // 与z=0平面求交点
    const planeZ = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const intersect = new THREE.Vector3();
    raycaster.ray.intersectPlane(planeZ, intersect);
    // 计算缩放因子
    const delta = event.deltaY < 0 ? 0.9 : 1.1;
    // 缩放前目标点到相机的向量
    const camToTarget = new THREE.Vector3().subVectors(intersect, camera3D.position);
    camToTarget.multiplyScalar(1 - delta);
    // 相机位置向目标点移动
    camera3D.position.add(camToTarget);
    // controls.target 也跟随目标点移动
    controls.target.add(camToTarget);
    controls.update();
  }, { passive: false });
};

function resetView() {
  if (!camera3D) return;
  // 读取画布尺寸（715×715 兜底）
  const w = glCanvas.value?.clientWidth || 715;
  const h = glCanvas.value?.clientHeight || 715;
  camera3D.aspect = w / h;
  camera3D.updateProjectionMatrix();

  const maxL = Math.max(props.Lx, props.Ly);
  const fit = maxL / (2 * Math.tan(THREE.MathUtils.degToRad(camera3D.fov) / 2));
  const dist = fit * 0.9

  // 相机回到默认位置 & 姿态（围绕原点俯视）
  camera3D.position.set(0, -dist, dist*1.4);
  camera3D.up.set(0, 0, 1);
  camera3D.lookAt(0, 0, 0);

  // OrbitControls 的旋转中心固定原点，并立即生效
  if (controls) {
    controls.target.set(0, 0, 0);
    controls.update();
  };
};
defineExpose({ resetView });

onMounted(() => {
  init3D();
});
onBeforeUnmount(() => {
  cancelAnimationFrame(raf);
  controls && controls.dispose();
  renderer && renderer.dispose();
});
watch(() => props.z, () => {
  if (mesh3D) {
    const g3 = buildGeometry3D();
    mesh3D.geometry.dispose();
    mesh3D.geometry = g3;
  };
});
</script>
