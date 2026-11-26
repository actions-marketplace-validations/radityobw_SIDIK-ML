import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const KEYFRAMES = [
  { p: 0.0, rotY: -0.2, rotX: 0.06, rotZ: 0.0, y: 0.0, camZ: 4.15, explode: 0.0 },
  { p: 0.25, rotY: 0.56, rotX: -0.07, rotZ: 0.02, y: 0.03, camZ: 3.7, explode: 0.9 },
  { p: 0.55, rotY: -0.46, rotX: 0.12, rotZ: -0.02, y: 0.0, camZ: 3.55, explode: 1.0 },
  { p: 0.82, rotY: 0.34, rotX: -0.05, rotZ: 0.01, y: 0.02, camZ: 3.82, explode: 0.45 },
  { p: 1.0, rotY: 0.06, rotX: 0.03, rotZ: 0.0, y: 0.0, camZ: 4.05, explode: 0.0 },
];

const CHAPTER_LABELS = {
  overview: '01 Overview',
  architecture: '02 Non-ANN Engine',
  benchmarks: '03 Empirical Table',
  cli: '04 CLI',
  releases: '05 Releases',
};

const MESH_EXPLODE_OFFSETS = {
  'outer-blue': [-0.018, -0.028, 0.0],
  'outer-yellow': [0.018, -0.028, 0.0],
  magnifier: [0.0, 0.014, 0.0],
  'graph-lines-solid': [0.0, 0.044, 0.0],
  diamond: [0.0, 0.072, -0.008],
  'node-blue': [-0.008, 0.064, 0.0],
  'node-green': [0.0, 0.08, 0.006],
  'node-orange': [0.008, 0.064, 0.006],
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function samplePose(p) {
  if (p <= KEYFRAMES[0].p) return KEYFRAMES[0];
  for (let i = 0; i < KEYFRAMES.length - 1; i++) {
    const a = KEYFRAMES[i];
    const b = KEYFRAMES[i + 1];
    if (p <= b.p) {
      const t = smoothstep(clamp((p - a.p) / Math.max(1e-4, b.p - a.p), 0, 1));
      return {
        rotY: lerp(a.rotY, b.rotY, t),
        rotX: lerp(a.rotX, b.rotX, t),
        rotZ: lerp(a.rotZ, b.rotZ, t),
        y: lerp(a.y, b.y, t),
        camZ: lerp(a.camZ, b.camZ, t),
        explode: lerp(a.explode, b.explode, t),
      };
    }
  }
  return KEYFRAMES[KEYFRAMES.length - 1];
}

function initShowcase() {
  const stageEl = document.getElementById('stage-3d');
  const canvas = document.getElementById('canvas-3d');
  if (!stageEl || !canvas) return;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
  } catch {
    return;
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.04;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.52;

  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 40);
  camera.position.set(0, 0.02, 4.15);

  const keyLight = new THREE.DirectionalLight(0xfffcf7, 1.35);
  keyLight.position.set(3.2, 4.2, 5.0);
  scene.add(keyLight);

  const amberRim = new THREE.SpotLight(0xd97706, 6.5, 18, Math.PI / 3.0, 0.55, 1.1);
  amberRim.position.set(2.8, 2.2, -3.0);
  scene.add(amberRim);

  const cobaltRim = new THREE.DirectionalLight(0x2563eb, 1.15);
  cobaltRim.position.set(-2.9, -1.2, -2.6);
  scene.add(cobaltRim);

  function applyTheme() {
    const isLight = document.documentElement.dataset.theme === 'light';
    if (isLight) {
      renderer.toneMappingExposure = 1.04;
      scene.environmentIntensity = 0.52;
      keyLight.intensity = 1.35;
      amberRim.intensity = 6.0;
      cobaltRim.intensity = 0.75;
    } else {
      renderer.toneMappingExposure = 1.1;
      scene.environmentIntensity = 0.82;
      keyLight.intensity = 2.0;
      amberRim.intensity = 8.5;
      cobaltRim.intensity = 1.15;
    }
  }
  applyTheme();
  new MutationObserver(applyTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  const pivot = new THREE.Group();
  scene.add(pivot);

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let dragYaw = 0;
  let dragPitch = 0;
  let hoverX = 0;
  let hoverY = 0;
  let isDragging = false;
  let prevX = 0;
  let prevY = 0;

  stageEl.addEventListener('pointerdown', (e) => {
    isDragging = true;
    prevX = e.clientX;
    prevY = e.clientY;
    stageEl.setPointerCapture(e.pointerId);
  });

  stageEl.addEventListener('pointermove', (e) => {
    const rect = stageEl.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      hoverX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      hoverY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    }
    if (!isDragging) return;
    const dx = e.clientX - prevX;
    const dy = e.clientY - prevY;
    prevX = e.clientX;
    prevY = e.clientY;
    dragYaw += dx * 0.011;
    dragPitch = clamp(dragPitch + dy * 0.008, -0.58, 0.58);
  });

  const endDrag = () => {
    isDragging = false;
  };
  stageEl.addEventListener('pointerup', endDrag);
  stageEl.addEventListener('pointercancel', endDrag);
  stageEl.addEventListener('pointerleave', () => {
    hoverX = 0;
    hoverY = 0;
  });

  function resize() {
    const w = Math.max(1, stageEl.clientWidth);
    const h = Math.max(1, stageEl.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  if ('ResizeObserver' in window) {
    new ResizeObserver(resize).observe(stageEl);
  }
  window.addEventListener('resize', resize, { passive: true });

  const chapterEl = document.getElementById('stage-chapter');
  const explodeEl = document.getElementById('stage-explode');
  const progressBar = document.getElementById('scroll-progress-bar');
  const chapterEls = Array.from(document.querySelectorAll('[data-chapter]'));

  function computeScrollProgress() {
    const doc = document.documentElement;
    const maxScroll = Math.max(1, doc.scrollHeight - window.innerHeight);
    return clamp(window.scrollY / maxScroll, 0, 1);
  }

  function updateHud(p, exp) {
    if (progressBar) {
      progressBar.style.transform = `scaleX(${p.toFixed(4)})`;
    }
    if (explodeEl) {
      explodeEl.textContent = exp > 0.12 ? `DECISION TREE DEPTH ${Math.round(exp * 100)}%` : 'ASSEMBLED EMBLEM';
    }
    if (!chapterEl || chapterEls.length === 0) return;
    const triggerY = window.innerHeight * 0.36;
    let activeKey = chapterEls[0].dataset.chapter;
    for (const el of chapterEls) {
      if (el.getBoundingClientRect().top <= triggerY) {
        activeKey = el.dataset.chapter;
      }
    }
    const label = CHAPTER_LABELS[activeKey];
    if (label && chapterEl.textContent !== label) {
      chapterEl.textContent = label;
    }
  }

  const partRecords = [];
  let curRotY = KEYFRAMES[0].rotY;
  let curRotX = KEYFRAMES[0].rotX;
  let curRotZ = KEYFRAMES[0].rotZ;
  let curY = KEYFRAMES[0].y;
  let curCamZ = KEYFRAMES[0].camZ;
  let curExplode = 0.0;
  let curHoverX = 0;
  let curHoverY = 0;
  let elapsed = 0;
  let lastTime = performance.now();
  let rafId = 0;
  let loaded = false;

  const loader = new GLTFLoader();
  loader.load(
    './models/sidik-ml.glb',
    (gltf) => {
      const model = gltf.scene;
      model.traverse((obj) => {
        if (obj.isMesh) {
          const offset = MESH_EXPLODE_OFFSETS[obj.name];
          if (offset) {
            partRecords.push({
              obj,
              basePos: obj.position.clone(),
              offset,
            });
          }
        }
      });

      const upright = new THREE.Group();
      upright.rotation.x = Math.PI / 2;
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const normScale = 2.15 / maxDim;

      model.position.sub(center);
      upright.add(model);
      upright.scale.setScalar(normScale);
      pivot.add(upright);

      loaded = true;
      stageEl.classList.add('is-ready');
      resize();
    },
    undefined,
    () => {
      stageEl.classList.add('is-fallback');
    }
  );

  function frame(now) {
    rafId = requestAnimationFrame(frame);
    const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) * 0.001));
    lastTime = now;

    const p = computeScrollProgress();
    const pose = samplePose(p);

    if (!isDragging) {
      dragYaw *= Math.exp(-1.4 * dt);
      dragPitch *= Math.exp(-1.6 * dt);
    }

    const alpha = reducedMotion ? 1 : 1 - Math.exp(-8.5 * dt);
    curRotY = lerp(curRotY, pose.rotY, alpha);
    curRotX = lerp(curRotX, pose.rotX, alpha);
    curRotZ = lerp(curRotZ, pose.rotZ, alpha);
    curY = lerp(curY, pose.y, alpha);
    curCamZ = lerp(curCamZ, pose.camZ, alpha);
    curExplode = lerp(curExplode, reducedMotion ? 0 : pose.explode, alpha);
    curHoverX = lerp(curHoverX, hoverX, alpha);
    curHoverY = lerp(curHoverY, hoverY, alpha);

    updateHud(p, curExplode);

    elapsed += dt;
    const idleFloat = reducedMotion ? 0 : Math.sin(elapsed * 1.1) * 0.02;
    const idleYaw = reducedMotion ? 0 : Math.sin(elapsed * 0.65) * 0.035;

    if (loaded) {
      for (const rec of partRecords) {
        rec.obj.position.set(
          rec.basePos.x + rec.offset[0] * curExplode,
          rec.basePos.y + rec.offset[1] * curExplode,
          rec.basePos.z + rec.offset[2] * curExplode
        );
      }
      pivot.rotation.y = curRotY + dragYaw + curHoverX * 0.14 + idleYaw;
      pivot.rotation.x = curRotX + dragPitch + curHoverY * 0.08;
      pivot.rotation.z = curRotZ;
      pivot.position.y = curY + idleFloat;
      camera.position.z = curCamZ;
      renderer.render(scene, camera);
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    } else if (!rafId) {
      lastTime = performance.now();
      rafId = requestAnimationFrame(frame);
    }
  });

  lastTime = performance.now();
  rafId = requestAnimationFrame(frame);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initShowcase, { once: true });
} else {
  initShowcase();
}
