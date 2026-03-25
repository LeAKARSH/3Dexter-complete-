import React, { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Stage } from '@react-three/drei';
import { Send, Box, Cpu, Download, Code, Layers, Trash2, ChevronRight, ChevronLeft } from 'lucide-react';
import { cn } from './lib/utils';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  type?: 'organic' | 'parametric';
  data?: any;
}

interface OrganicModelData {
  files: string[];
  objFile: string | null;
  plyFile: string | null;
  objUrl: string | null;
  plyUrl: string | null;
  message?: string;
}

const OrganicMesh = ({ model }: { model: OrganicModelData }) => {
  const [object, setObject] = useState<THREE.Group | null>(null);

  useEffect(() => {
    if (!model.objUrl) {
      setObject(null);
      return;
    }

    let cancelled = false;
    const loader = new OBJLoader();

    loader.load(
      model.objUrl,
      (loadedObject) => {
        if (cancelled) return;

        loadedObject.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.material = new THREE.MeshStandardMaterial({
              color: '#10b981',
              roughness: 0.45,
              metalness: 0.15,
            });
          }
        });

        setObject(loadedObject);
      },
      undefined,
      (error) => {
        console.error('Failed to load organic OBJ preview:', error);
        setObject(null);
      }
    );

    return () => {
      cancelled = true;
      setObject(null);
    };
  }, [model.objUrl]);

  if (!object) return null;

  return <primitive object={object} />;
};

const ParametricPreview = ({ code }: { code: string }) => {
  // Simple heuristic to show different shapes based on OpenSCAD keywords
  const isSphere = code.toLowerCase().includes('sphere');
  const isCylinder = code.toLowerCase().includes('cylinder');
  const isCube = code.toLowerCase().includes('cube') || code.toLowerCase().includes('box');

  return (
    <mesh castShadow receiveShadow>
      {isSphere ? (
        <sphereGeometry args={[1.5, 32, 32]} />
      ) : isCylinder ? (
        <cylinderGeometry args={[1, 1, 3, 32]} />
      ) : (
        <boxGeometry args={[2, 2, 2]} />
      )}
      <meshStandardMaterial color="#6366f1" roughness={0.4} metalness={0.6} />
    </mesh>
  );
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Welcome to ShapeFlow AI. Describe a 3D model you want to create.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentModel, setCurrentModel] = useState<{ type: 'organic' | 'parametric', data: any } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [modelType, setModelType] = useState<'shap-e' | 'hunyuan3d'>('shap-e');
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMsg, modelType }),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Pipeline request failed');
      }
      
      if (result.type === 'parametric') {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Parametric pipeline response.\n\nType: ${result.type}\n${result.message}`,
          type: 'parametric',
          data: result.code
        }]);
        if (result.code) {
          setCurrentModel({ type: 'parametric', data: result.code });
        }
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Organic pipeline response.\n\nGenerated files: ${(result.data.files || []).join(', ') || 'None'}\n${result.message}`,
          type: 'organic',
          data: result.data
        }]);
        setCurrentModel({ type: 'organic', data: result.data });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Sorry, I encountered an error in the pipeline.';
      setMessages(prev => [...prev, { role: 'assistant', content: errorMessage }]);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadFile = (format: 'obj' | 'ply' | 'scad') => {
    if (!currentModel) return;

    if (currentModel.type === 'organic') {
      const organicData = currentModel.data as OrganicModelData;
      const url = format === 'obj' ? organicData.objUrl : organicData.plyUrl;
      if (!url) return;

      const a = document.createElement('a');
      a.href = url;
      a.download = format === 'obj' ? organicData.objFile || 'model.obj' : organicData.plyFile || 'model.ply';
      a.click();
      return;
    }

    const blob = new Blob([currentModel.data], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `model.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-screen w-full bg-[#0a0a0a] text-white overflow-hidden font-sans">
      {/* Sidebar - Chat */}
      <div className={cn(
        "flex flex-col border-r border-white/10 transition-all duration-300 ease-in-out",
        sidebarOpen ? "w-[400px]" : "w-0 overflow-hidden border-none"
      )}>
        <div className="p-4 border-bottom border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
              <Cpu className="w-5 h-5 text-black" />
            </div>
            <h1 className="font-bold text-xl tracking-tight">ShapeFlow AI</h1>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="p-1 hover:bg-white/5 rounded">
            <ChevronLeft className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
          {messages.map((msg, i) => (
            <div key={i} className={cn(
              "flex flex-col max-w-[85%]",
              msg.role === 'user' ? "ml-auto items-end" : "items-start"
            )}>
              <div className={cn(
                "p-3 rounded-2xl text-sm leading-relaxed",
                msg.role === 'user' 
                  ? "bg-emerald-600 text-white rounded-tr-none" 
                  : "bg-white/5 text-gray-200 border border-white/10 rounded-tl-none"
              )}>
                {msg.content}
              </div>
              {msg.type && (
                <div className="mt-2 flex items-center gap-2 text-[10px] uppercase tracking-widest text-gray-500 font-mono">
                  {msg.type === 'organic' ? <Layers className="w-3 h-3" /> : <Code className="w-3 h-3" />}
                  {msg.type} pipeline
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex items-center gap-2 text-emerald-500 text-xs font-mono animate-pulse">
              <Cpu className="w-4 h-4 animate-spin" />
              Processing Pipeline...
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
          <div className="flex items-center gap-3 mb-3">
            <label className="text-xs text-gray-400 font-mono">Model:</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as 'shap-e' | 'hunyuan3d')}
              className="bg-white/5 border border-white/10 rounded-lg py-1 px-2 text-xs text-gray-200 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="shap-e">Shape-E (Fast)</option>
              <option value="hunyuan3d">Hunyuan3D-2GP (High Quality)</option>
            </select>
          </div>
          <div className="relative">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe a shape..."
              className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-4 pr-12 text-sm focus:outline-none focus:border-emerald-500/50 transition-colors"
            />
            <button 
              type="submit"
              disabled={isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-emerald-500 text-black rounded-lg hover:bg-emerald-400 transition-colors disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>

      {/* Main Viewport */}
      <div className="flex-1 relative flex flex-col">
        {!sidebarOpen && (
          <button 
            onClick={() => setSidebarOpen(true)}
            className="absolute top-4 left-4 z-10 p-2 bg-white/5 hover:bg-white/10 rounded-xl border border-white/10 backdrop-blur-md"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        )}

        {/* 3D Canvas */}
        <div className="flex-1 bg-gradient-to-b from-[#0a0a0a] to-[#151515]">
          <Canvas shadows dpr={[1, 2]}>
            <PerspectiveCamera makeDefault position={[5, 5, 5]} fov={50} />
            <OrbitControls makeDefault />
            
            <Stage environment="city" intensity={0.6}>
              {currentModel?.type === 'organic' && <OrganicMesh model={currentModel.data} />}
              {currentModel?.type === 'parametric' && <ParametricPreview code={currentModel.data} />}
              {!currentModel && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
                  <planeGeometry args={[10, 10]} />
                  <meshStandardMaterial color="#111" />
                </mesh>
              )}
            </Stage>

            <Grid 
              infiniteGrid 
              fadeDistance={50} 
              fadeStrength={5} 
              cellSize={1} 
              sectionSize={5} 
              sectionColor="#333" 
              cellColor="#222" 
            />
            
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} castShadow />
          </Canvas>
        </div>

        {/* Editor / Controls Overlay */}
        <div className="absolute bottom-6 left-6 right-6 flex items-end justify-between pointer-events-none">
          <div className="flex flex-col gap-4 pointer-events-auto">
            {currentModel?.type === 'parametric' && (
              <div className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-2xl p-4 w-[400px] shadow-2xl">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 text-xs font-mono text-emerald-500">
                    <Code className="w-3 h-3" />
                    OPENSCAD EDITOR
                  </div>
                  <button className="p-1 hover:bg-white/10 rounded text-gray-400">
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
                <textarea 
                  value={currentModel.data}
                  onChange={(e) => setCurrentModel({ ...currentModel, data: e.target.value })}
                  className="w-full h-40 bg-black/50 border border-white/5 rounded-lg p-3 text-[11px] font-mono text-gray-300 focus:outline-none resize-none"
                />
              </div>
            )}
          </div>

          <div className="flex gap-2 pointer-events-auto">
            <button 
              onClick={() => downloadFile(currentModel?.type === 'organic' ? 'ply' : 'scad')}
              disabled={!currentModel}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-xs font-medium backdrop-blur-md transition-all disabled:opacity-30"
            >
              <Download className="w-4 h-4" />
              {currentModel?.type === 'organic' ? 'DOWNLOAD .PLY' : 'DOWNLOAD .SCAD'}
            </button>
            <button 
              onClick={() => downloadFile('obj')}
              disabled={!currentModel}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-black hover:bg-emerald-400 rounded-xl text-xs font-bold shadow-lg shadow-emerald-500/20 transition-all disabled:opacity-30"
            >
              <Box className="w-4 h-4" />
              EXPORT .OBJ
            </button>
          </div>
        </div>

        {/* Status Bar */}
        <div className="absolute top-4 right-4 flex items-center gap-4 pointer-events-none">
          <div className="bg-black/50 backdrop-blur-md border border-white/10 rounded-full px-4 py-1.5 flex items-center gap-2 text-[10px] font-mono tracking-widest text-gray-400">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            PIPELINE STATUS: READY
          </div>
        </div>
      </div>
    </div>
  );
}
