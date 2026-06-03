import { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Stage } from '@react-three/drei';
import * as THREE from 'three';
import type { EditorEntry, ModelRecord } from './types';

/* ── SCAD helpers ── */

interface ParamMeta {
  value: number;
  min: number;
  max: number;
  step: number;
}

function parseScadParams(code: string): Record<string, number> {
  return Object.fromEntries(
    Object.entries(parseScadParamsMeta(code)).map(([k, m]) => [k, m.value])
  );
}

function parseScadParamsMeta(code: string): Record<string, ParamMeta> {
  const result: Record<string, ParamMeta> = {};
  // Match:  varname = number;  // [min:max]  or  // [min:max:step]
  const re = /^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\d.]+)\s*;(?:.*\/\/\s*\[([^\]]+)\])?/gm;
  let m;
  while ((m = re.exec(code)) !== null) {
    const value = parseFloat(m[2]);
    if (isNaN(value)) continue;
    let min = 0.1, max = value * 4 + 20, step = value >= 10 ? 0.5 : 0.1;
    if (m[3]) {
      const parts = m[3].split(':').map(Number);
      if (parts.length === 2 && !parts.some(isNaN)) {
        [min, max] = parts;
        step = Number.isInteger(value) && (max - min) <= 30 ? 1 : (max - min) / 100;
        step = parseFloat(step.toFixed(2));
      } else if (parts.length === 3 && !parts.some(isNaN)) {
        [min, max, step] = parts;
      }
    }
    result[m[1]] = { value, min: Math.max(0.1, min), max, step };
  }
  return result;
}

function substituteScadParams(code: string, params: Record<string, number>): string {
  return code.replace(
    /^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\d.]+)\s*;/gm,
    (match, key) => params[key] !== undefined ? `${key} = ${params[key]};` : match,
  );
}

function highlightScad(code: string) {
  return code.split('\n').map((line, i) => {
    if (line.trim().startsWith('//')) return <div key={i}><span className="scad-com">{line}</span></div>;
    const re = /(\bmodule\b|\bunion\b|\btranslate\b|\bcylinder\b|\bsphere\b|\bcube\b|\bfor\b|\b\$fn\b|\bdifference\b|\bintersection\b|\blinear_extrude\b|\brotate_extrude\b)|(\b\d+\.?\d*\b)/g;
    const parts: React.ReactNode[] = [];
    let last = 0; let mm;
    while ((mm = re.exec(line)) !== null) {
      if (mm.index > last) parts.push(<span key={`t${last}`}>{line.slice(last, mm.index)}</span>);
      if (mm[1]) parts.push(<span key={`k${mm.index}`} className="scad-kw">{mm[1]}</span>);
      else if (mm[2]) parts.push(<span key={`n${mm.index}`} className="scad-num">{mm[2]}</span>);
      last = mm.index + mm[0].length;
    }
    if (last < line.length) parts.push(<span key={`r${last}`}>{line.slice(last)}</span>);
    return <div key={i}>{parts.length ? parts : (line || ' ')}</div>;
  });
}

/* ── Thumbnail capture (inside Canvas) ── */
function ThumbnailCapture({ onCapture }: { onCapture: (data: string) => void }) {
  const { gl, scene, camera } = useThree();
  const fired = useRef(false);
  useEffect(() => {
    if (fired.current) return;
    const id = setTimeout(() => {
      fired.current = true;
      gl.render(scene, camera);
      const offscreen = document.createElement('canvas');
      offscreen.width = 320; offscreen.height = 200;
      offscreen.getContext('2d')?.drawImage(gl.domElement, 0, 0, 320, 200);
      onCapture(offscreen.toDataURL('image/jpeg', 0.7));
    }, 1200);
    return () => clearTimeout(id);
  }, []);
  return null;
}

/* ── Organic mesh loader ── */
function OrganicMesh({ model, onStats, wireframe }: { model: ModelRecord; onStats: (s: MeshStats) => void; wireframe: boolean }) {
  const [object, setObject] = useState<THREE.Group | null>(null);
  const statsRef = useRef(false);

  useEffect(() => {
    const url = model.objUrl || model.plyUrl;
    if (!url) return;
    let cancelled = false;
    Promise.all([
      import('three/examples/jsm/loaders/OBJLoader.js'),
      import('three/examples/jsm/loaders/PLYLoader.js'),
    ]).then(([{ OBJLoader }, { PLYLoader }]) => {
      if (cancelled) return;
      if (model.objUrl) {
        new OBJLoader().load(url, (obj) => {
          if (cancelled) return;
          let tris = 0;
          obj.traverse(child => {
            if ((child as THREE.Mesh).isMesh) {
              const mesh = child as THREE.Mesh;
              mesh.material = new THREE.MeshStandardMaterial({ color: '#10b981', roughness: 0.45, metalness: 0.15 });
              mesh.castShadow = true;
              const g = mesh.geometry;
              tris += g.index ? g.index.count / 3 : g.attributes.position.count / 3;
            }
          });
          const box = new THREE.Box3().setFromObject(obj);
          const s = box.getSize(new THREE.Vector3());
          normalizeObject(obj);
          setObject(obj);
          if (!statsRef.current) {
            statsRef.current = true;
            onStats({ triangles: Math.round(tris), bbox: `${s.x.toFixed(0)} × ${s.y.toFixed(0)} × ${s.z.toFixed(0)} mm` });
          }
        });
      } else {
        new PLYLoader().load(url, (geo) => {
          if (cancelled) return;
          geo.computeVertexNormals();
          const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#10b981', roughness: 0.45, metalness: 0.15, side: THREE.DoubleSide }));
          mesh.castShadow = true;
          const box = new THREE.Box3().setFromObject(mesh);
          const s = box.getSize(new THREE.Vector3());
          normalizeObject(mesh);
          const group = new THREE.Group();
          group.add(mesh);
          setObject(group);
          const tris = geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
          if (!statsRef.current) {
            statsRef.current = true;
            onStats({ triangles: Math.round(tris), bbox: `${s.x.toFixed(0)} × ${s.y.toFixed(0)} × ${s.z.toFixed(0)} mm` });
          }
        });
      }
    });
    return () => { cancelled = true; setObject(null); };
  }, [model.id]);

  // Apply / remove wireframe whenever the prop changes
  useEffect(() => {
    if (!object) return;
    object.traverse(child => {
      if ((child as THREE.Mesh).isMesh) {
        const mat = (child as THREE.Mesh).material;
        (Array.isArray(mat) ? mat : [mat]).forEach(m => {
          (m as THREE.MeshStandardMaterial).wireframe = wireframe;
        });
      }
    });
  }, [wireframe, object]);

  if (!object) return null;
  return <primitive object={object} />;
}

function normalizeObject(obj: THREE.Object3D) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const scale = 3 / Math.max(size.x, size.y, size.z, 0.001);
  obj.position.sub(center);
  obj.scale.setScalar(scale);
}

/* ── Parametric mesh — real STL from OpenSCAD ── */
function ParametricMesh({
  code, wireframe, onStats, onRenderStart, onRenderEnd, onRenderError,
}: {
  code: string;
  wireframe: boolean;
  onStats: (s: MeshStats) => void;
  onRenderStart: () => void;
  onRenderEnd: () => void;
  onRenderError: (e: string) => void;
}) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    if (!code) return;
    let cancelled = false;
    onRenderStart();

    fetch('/api/render-scad', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    })
      .then(async res => {
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          throw new Error((d as any).error ?? `Render failed (${res.status})`);
        }
        return res.arrayBuffer();
      })
      .then(buf => {
        if (cancelled) return;
        return import('three/examples/jsm/loaders/STLLoader.js').then(({ STLLoader }) => {
          const geo = new STLLoader().parse(buf);
          geo.computeVertexNormals();
          // center + normalise to ~3 units
          geo.computeBoundingBox();
          const box = geo.boundingBox!;
          const size = new THREE.Vector3();
          const center = new THREE.Vector3();
          box.getSize(size);
          box.getCenter(center);
          geo.translate(-center.x, -center.y, -center.z);
          const scale = 3 / Math.max(size.x, size.y, size.z, 0.001);
          geo.scale(scale, scale, scale);
          if (!cancelled) {
            setGeometry(geo);
            const tris = geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
            onStats({ triangles: Math.round(tris), bbox: `${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} mm` });
            onRenderEnd();
          }
        });
      })
      .catch(e => {
        if (!cancelled) { onRenderError(e.message); onRenderEnd(); }
      });

    return () => { cancelled = true; };
  }, [code]);

  useEffect(() => {
    if (!geometry) return;
    (geometry as any).wireframe = undefined; // not a property, applied on material below
  }, [wireframe]);

  if (!geometry) return null;
  return (
    <mesh castShadow receiveShadow>
      <primitive object={geometry} attach="geometry" />
      <meshStandardMaterial color="#6366f1" roughness={0.4} metalness={0.6} wireframe={wireframe} />
    </mesh>
  );
}

interface MeshStats { triangles: number; bbox: string; }

interface Props {
  entry: EditorEntry;
  dark: boolean;
  onToggleDark: () => void;
  onHome: () => void;
  onModelSaved: (model: ModelRecord) => void;
}

export function EditorScreen({ entry, dark, onToggleDark, onHome, onModelSaved }: Props) {
  const [model, setModel] = useState<ModelRecord | null>(entry.model);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [promptText, setPromptText] = useState(entry.prompt ?? entry.model?.prompt ?? '');
  const [organicModelType, setOrganicModelType] = useState<'shap-e' | 'hunyuan3d'>('shap-e');
  const [scadCode, setScadCode] = useState(entry.model?.code ?? '');
  const [renderCode, setRenderCode] = useState(entry.model?.code ?? ''); // code actually sent to OpenSCAD
  const [params, setParams] = useState<Record<string, number>>({});
  const [paramsMeta, setParamsMeta] = useState<Record<string, ParamMeta>>({});
  const [paramsDirty, setParamsDirty] = useState(false);
  const [vt, setVt] = useState({ wireframe: false, grid: false, axes: false });
  const [scadOpen, setScadOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [meshStats, setMeshStats] = useState<MeshStats | null>(null);
  const [loadingStage, setLoadingStage] = useState('');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [openscadAvailable, setOpenscadAvailable] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(d => setOpenscadAvailable(d.openscadAvailable ?? false));
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setScadOpen(false); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  useEffect(() => {
    if (entry.model?.code) {
      const meta = parseScadParamsMeta(entry.model.code);
      setParams(Object.fromEntries(Object.entries(meta).map(([k, m]) => [k, m.value])));
      setParamsMeta(meta);
      setScadCode(entry.model.code);
      setRenderCode(entry.model.code);
    }
  }, [entry.model?.id]);

  const generate = useCallback(async (p: string, rt?: 'parametric' | 'organic') => {
    if (!p.trim()) return;
    setGenerating(true);
    setError(null);
    setMeshStats(null);
    const isGemini = entry.generationBackend === 'gemini';
    const stages = isGemini
      ? [{ p: 30, s: 'Calling Gemini API…' }, { p: 70, s: 'Parsing OpenSCAD…' }, { p: 95, s: 'Finalizing…' }]
      : [{ p: 15, s: 'Loading model weights…' }, { p: 35, s: 'Processing prompt…' }, { p: 60, s: 'Generating…' }, { p: 80, s: 'Repairing mesh…' }, { p: 95, s: 'Finalizing…' }];
    let idx = 0;
    const iv = setInterval(() => {
      if (idx < stages.length) { setLoadingProgress(stages[idx].p); setLoadingStage(stages[idx].s); idx++; }
    }, isGemini ? 600 : 1800);
    try {
      const res = await fetch('/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: p,
          modelType: rt === 'organic' || entry.route === 'organic' ? organicModelType : undefined,
          generationBackend: isGemini ? 'gemini' : undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? 'Generation failed');
      const savedModel: ModelRecord = data.model;
      setModel(savedModel);
      onModelSaved(savedModel);
      if (savedModel.code) {
        const meta = parseScadParamsMeta(savedModel.code);
        setScadCode(savedModel.code);
        setRenderCode(savedModel.code);
        setParams(Object.fromEntries(Object.entries(meta).map(([k, m]) => [k, m.value])));
        setParamsMeta(meta);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      clearInterval(iv);
      setLoadingProgress(100);
      setLoadingStage('Done');
      setTimeout(() => { setGenerating(false); setLoadingProgress(0); setLoadingStage(''); }, 400);
    }
  }, [organicModelType, entry.route, onModelSaved]);

  useEffect(() => {
    if (!entry.model && entry.prompt) generate(entry.prompt, entry.route);
  }, []);

  const saveThumbnail = useCallback((data: string) => {
    if (!model?.id) return;
    fetch(`/api/models/${model.id}/thumbnail`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thumbnail: data }),
    });
  }, [model?.id]);

  const onParamChange = (key: string, val: number) => {
    const next = { ...params, [key]: val };
    setParams(next);
    setScadCode(substituteScadParams(scadCode, next));
    setParamsDirty(true);
  };

  const downloadScad = () => {
    const blob = new Blob([scadCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${model?.name ?? 'model'}.scad`; a.click();
    URL.revokeObjectURL(url);
  };

  const downloadOrganicFile = (format: 'obj' | 'ply') => {
    if (!model) return;
    const file = format === 'obj' ? model.objFile : model.plyFile;
    if (!file) return;
    const a = document.createElement('a');
    a.href = `/api/organic/download/${encodeURIComponent(file)}`;
    a.download = file; a.click();
  };

  const route = model?.route ?? (entry.route ?? 'parametric');
  const isParametric = route === 'parametric';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', height: '100vh', background: 'var(--bg)' }}>

      {/* ── Stage ── */}
      <div style={{ position: 'relative', overflow: 'hidden' }}>
        {/* Stage bg */}
        <div style={{ position: 'absolute', inset: 0, background: dark ? 'radial-gradient(ellipse at center, #1A1814 0%, var(--bg) 70%)' : 'radial-gradient(ellipse at center, #FFFFFF 0%, var(--bg) 70%)' }} />
        {/* Grid overlay */}
        <div style={{ position: 'absolute', inset: 0, backgroundImage: `linear-gradient(to right, ${dark ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)'} 1px, transparent 1px), linear-gradient(to bottom, ${dark ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)'} 1px, transparent 1px)`, backgroundSize: '24px 24px', maskImage: 'radial-gradient(ellipse at center, black 30%, transparent 80%)', pointerEvents: 'none', zIndex: 1 }} />

        {/* Brand + back */}
        <div style={{ position: 'absolute', top: 16, left: 16, display: 'flex', alignItems: 'center', gap: 10, zIndex: 3 }}>
          <button onClick={onHome} style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '5px 10px', cursor: 'pointer', fontSize: 12, color: 'var(--text-2)', boxShadow: 'var(--shadow-1)' }}>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M10 12L6 8l4-4" strokeLinecap="round" strokeLinejoin="round"/></svg>
            Home
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <RouteBadge route={route} />
            <span style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'var(--font-mono)' }}>auto-classified</span>
          </div>
        </div>

        {/* Toolbar top-right */}
        <div style={{ position: 'absolute', top: 16, right: 16, display: 'flex', alignItems: 'center', gap: 4, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 999, padding: 4, boxShadow: 'var(--shadow-1)', zIndex: 3 }}>
          {(['wireframe', 'grid', 'axes'] as const).map(k => (
            <TbBtn key={k} active={vt[k]} onClick={() => setVt(v => ({ ...v, [k]: !v[k] }))}>
              {k.charAt(0).toUpperCase() + k.slice(1)}
            </TbBtn>
          ))}
          {isParametric && scadCode && (
            <>
              <span style={{ width: 1, height: 18, background: 'var(--border)', margin: '0 2px' }} />
              <TbBtn active={scadOpen} onClick={() => setScadOpen(o => !o)}>
                <CodeIcon /> Code
              </TbBtn>
            </>
          )}
          <span style={{ width: 1, height: 18, background: 'var(--border)', margin: '0 2px' }} />
          <TbBtn active={false} onClick={onToggleDark}>{dark ? '☀' : '☾'}</TbBtn>
        </div>

        {/* Canvas */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 2 }}>
          <Canvas
            shadows
            dpr={[1, 2]}
            gl={{ preserveDrawingBuffer: true }}
          >
            <PerspectiveCamera makeDefault position={[5, 5, 5]} fov={50} />
            <OrbitControls makeDefault />
            <Stage environment="city" intensity={dark ? 0.4 : 0.6}>
              {model?.route === 'organic' && <OrganicMesh model={model} onStats={setMeshStats} wireframe={vt.wireframe} />}
              {model?.route === 'parametric' && openscadAvailable && renderCode && (
                <ParametricMesh
                  key={renderCode}
                  code={renderCode}
                  wireframe={vt.wireframe}
                  onStats={setMeshStats}
                  onRenderStart={() => setIsRendering(true)}
                  onRenderEnd={() => setIsRendering(false)}
                  onRenderError={(e) => { setRenderError(e); setIsRendering(false); }}
                />
              )}
              {!model && !generating && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
                  <planeGeometry args={[10, 10]} />
                  <meshStandardMaterial color={dark ? '#111' : '#e8e4dc'} />
                </mesh>
              )}
            </Stage>
            {vt.grid && <Grid infiniteGrid fadeDistance={50} fadeStrength={5} cellSize={1} sectionSize={5}
              sectionColor={dark ? '#333' : '#ddd'} cellColor={dark ? '#222' : '#eee'} />}
            {vt.axes && <axesHelper args={[3]} />}
            <ambientLight intensity={dark ? 0.3 : 0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} castShadow />
            {model?.route === 'organic' && <pointLight position={[-8, 6, -8]} intensity={0.5} />}
            {model && <ThumbnailCapture onCapture={saveThumbnail} />}
          </Canvas>
        </div>

        {/* Generating overlay */}
        {generating && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
            <div style={{ background: dark ? 'rgba(26,25,22,.9)' : 'rgba(255,255,255,.9)', border: '1px solid var(--border)', borderRadius: 14, padding: '20px 28px', minWidth: 280, boxShadow: 'var(--shadow-3)', backdropFilter: 'blur(8px)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
                <span style={{ color: 'var(--indigo)', fontWeight: 600 }}>GENERATING</span>
                <span>{loadingProgress}%</span>
              </div>
              <div style={{ width: '100%', height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', background: 'var(--indigo)', borderRadius: 2, width: `${loadingProgress}%`, transition: 'width .5s ease-out' }} />
              </div>
              <div style={{ marginTop: 8, fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{loadingStage}</div>
            </div>
          </div>
        )}

        {/* OpenSCAD compiling overlay */}
        {isRendering && !generating && (
          <div style={{ position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '7px 14px', boxShadow: 'var(--shadow-1)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', pointerEvents: 'none' }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--indigo)', animation: 'pulse 1s infinite' }} />
            Compiling OpenSCAD…
          </div>
        )}

        {/* Render error */}
        {renderError && (
          <div style={{ position: 'absolute', bottom: 100, left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: 'var(--surface)', border: '1px solid #fca5a5', borderRadius: 10, padding: '12px 16px', maxWidth: 520, boxShadow: 'var(--shadow-2)', color: '#dc2626', fontSize: 12, fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap' }}>
            <b style={{ fontFamily: 'var(--font-sans)' }}>OpenSCAD error</b><br />{renderError}
            <button onClick={() => setRenderError(null)} style={{ display: 'block', marginTop: 8, fontSize: 11, color: 'var(--text-3)', background: 'none', border: 'none', cursor: 'pointer' }}>dismiss</button>
          </div>
        )}

        {/* Generation error overlay */}
        {error && (
          <div style={{ position: 'absolute', bottom: 100, left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: 'var(--surface)', border: '1px solid #fca5a5', borderRadius: 10, padding: '12px 16px', maxWidth: 480, boxShadow: 'var(--shadow-2)', color: '#dc2626', fontSize: 12.5 }}>
            {error}
            <button onClick={() => setError(null)} style={{ marginLeft: 12, fontSize: 11, color: 'var(--text-3)', background: 'none', border: 'none', cursor: 'pointer' }}>dismiss</button>
          </div>
        )}

        {/* SCAD code modal */}
        {scadOpen && isParametric && scadCode && (
          <div
            onClick={(e) => { if (e.target === e.currentTarget) setScadOpen(false); }}
            style={{ position: 'absolute', inset: 0, zIndex: 20, background: 'rgba(0,0,0,.45)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}
          >
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, boxShadow: 'var(--shadow-3)', width: '100%', maxWidth: 680, maxHeight: '80vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Modal header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 18px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CodeIcon />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600 }}>
                    {model?.name ? `${model.name}.scad` : 'model.scad'}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-4)', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 4, padding: '1px 7px' }}>
                    {Object.keys(params).length} params
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <div style={{ position: 'relative' }}>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(scadCode).then(() => {
                          setCopied(true);
                          setTimeout(() => setCopied(false), 2000);
                        });
                      }}
                      style={{ display: 'flex', alignItems: 'center', gap: 6, height: 28, padding: '0 12px', borderRadius: 6, border: '1px solid', borderColor: copied ? 'var(--green)' : 'var(--border)', background: copied ? 'rgba(22,163,74,.08)' : 'var(--surface-2)', color: copied ? 'var(--green)' : 'var(--text-2)', fontSize: 12, fontWeight: 500, cursor: 'pointer', transition: 'all .15s' }}
                    >
                      {copied ? <CheckIcon /> : <CopyIcon />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <button
                    onClick={downloadScad}
                    style={{ display: 'flex', alignItems: 'center', gap: 6, height: 28, padding: '0 12px', borderRadius: 6, border: 'none', background: 'var(--text)', color: 'var(--bg)', fontSize: 12, fontWeight: 500, cursor: 'pointer' }}
                  >
                    <DownloadIcon /> Download
                  </button>
                  <button
                    onClick={() => setScadOpen(false)}
                    style={{ width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-3)', cursor: 'pointer', display: 'grid', placeItems: 'center', fontSize: 16 }}
                  >
                    ×
                  </button>
                </div>
              </div>
              {/* Code body */}
              <pre style={{ margin: 0, padding: '16px 18px', fontFamily: 'var(--font-mono)', fontSize: 12.5, lineHeight: 1.6, color: 'var(--text-2)', overflowY: 'auto', overflowX: 'auto', whiteSpace: 'pre' }}>
                {highlightScad(scadCode)}
              </pre>
            </div>
          </div>
        )}

        {/* Prompt bar (bottom overlay) */}
        <div style={{ position: 'absolute', left: '50%', bottom: 20, transform: 'translateX(-50%)', width: 'min(600px, calc(100% - 48px))', zIndex: 4 }}>
          <form onSubmit={e => { e.preventDefault(); generate(promptText, entry.route); }}>
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, boxShadow: 'var(--shadow-2)' }}>
              <textarea
                value={promptText}
                onChange={e => setPromptText(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate(promptText, entry.route); } }}
                placeholder='Describe a 3D model… "a hex bolt with M8 thread"'
                rows={1}
                style={{ width: '100%', border: 'none', background: 'transparent', outline: 'none', resize: 'none', padding: '12px 14px 6px', fontFamily: 'inherit', fontSize: 13.5, color: 'var(--text)', lineHeight: 1.5 }}
              />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 8px 8px 12px' }}>
                <span style={{ fontSize: 11, color: 'var(--text-4)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  Route auto-detected
                  <span style={{ fontFamily: 'var(--font-mono)', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 4, padding: '1px 5px', fontSize: 10.5, color: 'var(--text-2)' }}>⌘ ↵</span>
                </span>
                <button
                  type="submit"
                  disabled={generating || !promptText.trim()}
                  style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--indigo)', color: 'white', border: 'none', cursor: 'pointer', display: 'grid', placeItems: 'center', opacity: (!promptText.trim() || generating) ? 0.35 : 1 }}
                >
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2"><path d="M8 3v10M4 9l4 4 4-4" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* ── Inspector ── */}
      <aside className="scrollbar-thin" style={{ borderLeft: '1px solid var(--border)', background: 'var(--bg-2)', overflowY: 'auto' }}>
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Inspector</h3>
        </div>

        {/* Model info */}
        {model ? (
          <>
            <InspSection>
              <div style={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.35, marginBottom: 4 }}>{model.name}</div>
              <div style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.4, marginBottom: 8 }}>"{model.prompt}"</div>
              <RouteBadge route={model.route} />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 10 }}>
                {meshStats && (
                  <>
                    <StatCell label="Triangles" value={meshStats.triangles.toLocaleString()} />
                    <StatCell label="Bounding box" value={meshStats.bbox} span />
                  </>
                )}
                {isParametric && model.paramCount !== undefined && (
                  <StatCell label="Parameters" value={`${model.paramCount}`} />
                )}
              </div>
              {/* Download */}
              <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                {isParametric ? (
                  <button onClick={downloadScad} style={btnPrimary}>
                    <DownloadIcon /> Download .scad
                  </button>
                ) : (
                  <>
                    {model.objFile && (
                      <button onClick={() => downloadOrganicFile('obj')} style={btnPrimary}>
                        <DownloadIcon /> Download .obj
                      </button>
                    )}
                    {model.plyFile && (
                      <button onClick={() => downloadOrganicFile('ply')} style={btnGhost}>
                        .ply
                      </button>
                    )}
                  </>
                )}
              </div>
            </InspSection>

            {/* Parameters */}
            {isParametric && Object.keys(params).length > 0 && (
              <InspSection>
                <SectionHeader title="Parameters" right={<span style={{ fontSize: 10.5, color: 'var(--text-4)', fontFamily: 'var(--font-mono)' }}>auto-extracted</span>} />
                {Object.entries(params).map(([key, val]) => {
                  const meta = paramsMeta[key];
                  const min  = meta?.min ?? Math.max(0.1, val * 0.1);
                  const max  = meta?.max ?? Math.max(val * 4, val + 20);
                  const step = meta?.step ?? (val >= 10 ? 0.5 : 0.1);
                  const unit = Number.isInteger(step) && (max - min) <= 30 ? '' : 'mm';
                  const pct  = Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));
                  return (
                    <div key={key} style={{ marginBottom: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 5 }}>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', fontWeight: 500 }}>{key}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <input
                            type="number"
                            value={val}
                            step={step}
                            min={min}
                            onChange={e => {
                              const v = parseFloat(e.target.value);
                              if (!isNaN(v) && v >= min) onParamChange(key, v);
                            }}
                            style={{ width: 54, height: 22, padding: '0 6px', borderRadius: 5, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)', fontFamily: 'var(--font-mono)', fontSize: 11, textAlign: 'right', outline: 'none' }}
                          />
                          {unit && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-4)', width: 20 }}>{unit}</span>}
                        </div>
                      </div>
                      <input
                        type="range" className="param-slider"
                        min={min} max={max} step={step}
                        value={Math.min(max, Math.max(min, val))}
                        style={{ '--p': `${pct}%` } as React.CSSProperties}
                        onChange={e => onParamChange(key, parseFloat(e.target.value))}
                      />
                    </div>
                  );
                })}
                <button
                  disabled={!paramsDirty}
                  onClick={() => { setRenderCode(scadCode); setParamsDirty(false); setRenderError(null); }}
                  style={{ ...btnPrimary, marginTop: 8, opacity: paramsDirty ? 1 : 0.35, cursor: paramsDirty ? 'pointer' : 'default' }}
                >
                  Re-render
                </button>
              </InspSection>
            )}

          </>
        ) : (
          <div style={{ padding: '20px 16px', color: 'var(--text-4)', fontSize: 12.5 }}>
            {generating ? 'Generating…' : 'Submit a prompt to get started.'}
          </div>
        )}
      </aside>
    </div>
  );
}

/* ── Shared sub-components ── */

function InspSection({ children }: { children: React.ReactNode }) {
  return <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)' }}>{children}</div>;
}

function SectionHeader({ title, right }: { title: string; right?: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
      <h4 style={{ margin: 0, fontSize: 11, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-3)' }}>{title}</h4>
      {right}
    </div>
  );
}

function StatCell({ label, value, span }: { label: string; value: string; span?: boolean }) {
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: '6px 8px', gridColumn: span ? 'span 2' : undefined }}>
      <div style={{ fontSize: 9.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500 }}>{value}</div>
    </div>
  );
}

function TbBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{ height: 28, padding: '0 12px', borderRadius: 999, border: 'none', background: active ? 'var(--text)' : 'transparent', color: active ? 'var(--bg)' : 'var(--text-2)', fontSize: 12, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, transition: 'all .12s' }}
    >
      {children}
    </button>
  );
}

function RouteBadge({ route }: { route: 'parametric' | 'organic' }) {
  const isP = route === 'parametric';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, height: 24, padding: '0 10px', borderRadius: 999, fontSize: 11.5, fontWeight: 600, border: '1px solid', background: isP ? 'var(--indigo-50)' : 'var(--terra-50)', color: isP ? 'var(--indigo)' : 'var(--terra)', borderColor: isP ? 'rgba(79,70,229,.18)' : 'rgba(194,65,12,.18)' }}>
      {isP ? <ParamIcon /> : <MeshIcon />}
      {isP ? 'Parametric' : 'Mesh'}
    </span>
  );
}

const btnPrimary: React.CSSProperties = { flex: 1, height: 32, borderRadius: 7, background: 'var(--text)', color: 'var(--bg)', border: 'none', cursor: 'pointer', fontSize: 12.5, fontWeight: 500, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7 };
const btnGhost: React.CSSProperties = { height: 32, padding: '0 12px', borderRadius: 7, background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', cursor: 'pointer', fontSize: 12.5, fontWeight: 500 };


const DownloadIcon = () => <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M8 3v8M5 8l3 3 3-3M3 13h10" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const CodeIcon = () => <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M5 4l-3 4 3 4M11 4l3 4-3 4M9 3l-2 10" strokeLinecap="round"/></svg>;
const CopyIcon = () => <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="6" y="6" width="8" height="8" rx="1.5"/><path d="M10 6V4a1.5 1.5 0 00-1.5-1.5H4A1.5 1.5 0 002.5 4v4.5A1.5 1.5 0 004 10h2" strokeLinecap="round"/></svg>;
const CheckIcon = () => <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 8l4 4 6-7" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const ParamIcon = () => <svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M3 6l5-3 5 3v4l-5 3-5-3z"/><path d="M3 6l5 3 5-3M8 9v4"/></svg>;
const MeshIcon = () => <svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="8" cy="8" r="5"/></svg>;
