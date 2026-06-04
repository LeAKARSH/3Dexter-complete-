import { useState, useEffect } from 'react';
import type { ModelRecord, ActivityRecord, EditorEntry } from './types';

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d === 1) return 'Yesterday';
  if (d < 7) return `${d}d ago`;
  return 'Last week';
}

const QUICK_PROMPTS = [
  'a hex bolt with M8 thread, 30mm shaft',
  'a phone stand with cable cutout',
  'enclosure for an 80×60mm PCB',
];

interface Props {
  dark: boolean;
  onToggleDark: () => void;
  onOpenEditor: (entry: EditorEntry) => void;
  parametricBackend: 'local' | 'gemini';
  onParametricBackendChange: (b: 'local' | 'gemini') => void;
}

export function HomeScreen({ onOpenEditor, parametricBackend, onParametricBackendChange }: Props) {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [activity, setActivity] = useState<ActivityRecord[]>([]);
  const [prompt, setPrompt] = useState('');
  const [geminiAvailable, setGeminiAvailable] = useState(false);
  const [filter, setFilter] = useState<'all' | 'parametric' | 'organic'>('all');

  const fetchData = () => {
    fetch('/api/models').then(r => r.json()).then(d => setModels(d.models ?? []));
    fetch('/api/activity').then(r => r.json()).then(d => setActivity(d.activity ?? []));
    fetch('/api/config').then(r => r.json()).then(d => setGeminiAvailable(d.geminiAvailable ?? false));
  };

  useEffect(() => { fetchData(); }, []);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!prompt.trim()) return;
    onOpenEditor({ model: null, prompt: prompt.trim() });
  };

  const filtered = filter === 'all' ? models : models.filter(m => m.route === filter);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', height: '100vh', background: 'var(--bg)', overflow: 'hidden' }}>

      {/* ── Sidebar ── */}
      <aside style={{ background: 'var(--sidebar-bg)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Brand */}
        <div style={{ height: 56, flexShrink: 0, display: 'flex', alignItems: 'center', padding: '0 16px', borderBottom: '1px solid var(--border)', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 7, background: 'var(--surface)', padding: 2, boxShadow: 'inset 0 0 0 1px var(--border)', display: 'grid', placeItems: 'center' }}>
            <img src="/assets/3dexter-logo.svg" alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: 14.5, letterSpacing: '-0.01em' }}>3Dexter</span>
        </div>

        {/* New model */}
        <button
          onClick={() => onOpenEditor({ model: null })}
          style={{ margin: '12px 12px 4px', height: 36, borderRadius: 9, background: 'var(--text)', color: 'var(--bg)', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500, display: 'flex', alignItems: 'center', gap: 8, padding: '0 12px' }}
        >
          <span style={{ width: 18, height: 18, borderRadius: 4, background: 'rgba(255,255,255,.12)', display: 'grid', placeItems: 'center', fontSize: 13 }}>+</span>
          New model
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,.10)', borderRadius: 4, padding: '0 6px', fontSize: 10.5 }}>⌘N</span>
        </button>

        {/* Search */}
        <div style={{ margin: '8px 12px 4px', display: 'flex', alignItems: 'center', gap: 8, height: 32, padding: '0 10px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12.5, color: 'var(--text-3)' }}>
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="7" cy="7" r="5"/><path d="M14 14l-3.5-3.5" strokeLinecap="round"/></svg>
          Search
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 4, padding: '0 5px', fontSize: 10.5 }}>⌘K</span>
        </div>

        {/* Nav */}
        <div className="scrollbar-thin" style={{ flex: 1, overflowY: 'auto', padding: '12px 8px' }}>
          <NavItem icon={<HomeIcon />} label="Home" active />
          <NavItem icon={<LibraryIcon />} label="Library" count={models.length} />
          <NavItem icon={<DraftsIcon />} label="Drafts" count={0} />

          {/* Recent */}
          <div style={{ padding: '16px 12px 4px', fontSize: 11, color: 'var(--text-4)', letterSpacing: '0.06em', fontWeight: 500 }}>Recent</div>
          {models.slice(0, 6).map(m => (
            <div
              key={m.id}
              onClick={() => onOpenEditor({ model: m })}
              style={{ padding: '5px 12px', fontSize: 12.5, color: 'var(--text-3)', cursor: 'pointer', margin: '0 4px', borderRadius: 6, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
              className="hover:bg-black/[.04] hover:text-[var(--text)]"
            >
              {m.name}
            </div>
          ))}
        </div>

        {/* Parametric backend toggle (only when Gemini key is set) */}
        {geminiAvailable && (
          <div style={{ borderTop: '1px solid var(--border)', padding: '10px 12px', flexShrink: 0 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', letterSpacing: '0.06em', fontWeight: 500, marginBottom: 6, textTransform: 'uppercase' }}>Parametric backend</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['local', 'gemini'] as const).map(b => (
                <button
                  key={b}
                  onClick={() => onParametricBackendChange(b)}
                  style={{ flex: 1, height: 26, borderRadius: 6, border: '1px solid', fontSize: 11.5, fontWeight: 500, cursor: 'pointer', background: parametricBackend === b ? 'var(--text)' : 'var(--surface)', color: parametricBackend === b ? 'var(--bg)' : 'var(--text-3)', borderColor: parametricBackend === b ? 'var(--text)' : 'var(--border)' }}
                >
                  {b === 'local' ? 'Local LoRA' : 'Gemini'}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* User block */}
        <div style={{ borderTop: '1px solid var(--border)', padding: 12, display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <div style={{ width: 30, height: 30, borderRadius: '50%', background: 'linear-gradient(135deg, #C9C2B0, #8A8779)', color: 'white', fontSize: 11.5, fontWeight: 600, display: 'grid', placeItems: 'center', flexShrink: 0 }}>
            A
          </div>
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 500, lineHeight: 1.2 }}>Anirudh</div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>PRO</div>
          </div>
          <button style={{ width: 28, height: 28, borderRadius: 6, border: 'none', background: 'transparent', color: 'var(--text-3)', cursor: 'pointer', display: 'grid', placeItems: 'center', marginLeft: 'auto' }}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="8" cy="8" r="2"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.5 3.5l1.4 1.4M11.1 11.1l1.4 1.4M3.5 12.5l1.4-1.4M11.1 4.9l1.4-1.4"/></svg>
          </button>
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' }}>
        {/* Topbar */}
        <div style={{ height: 56, flexShrink: 0, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', background: 'var(--bg-2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-3)' }}>
            <b style={{ color: 'var(--text)', fontWeight: 600 }}>Home</b>
          </div>
        </div>

        <div className="scrollbar-thin" style={{ flex: 1, overflowY: 'auto', padding: '32px 40px 64px' }}>
          <div style={{ maxWidth: 1100, margin: '0 auto' }}>

            {/* Greeting */}
            <h1 style={{ fontSize: 24, fontWeight: 600, letterSpacing: '-0.02em', margin: '0 0 4px' }}>Good to see you</h1>
            <p style={{ fontSize: 13, color: 'var(--text-3)', marginBottom: 24 }}>Pick up where you left off, or start something new.</p>

            {/* Composer */}
            <form onSubmit={handleSubmit}>
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 14, padding: 4, boxShadow: 'var(--shadow-2)', transition: 'border-color .15s, box-shadow .15s' }}
                onFocus={e => { e.currentTarget.style.borderColor = 'var(--indigo)'; e.currentTarget.style.boxShadow = '0 0 0 3px rgba(79,70,229,.10)'; }}
                onBlur={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.boxShadow = 'var(--shadow-2)'; }}
              >
                <div style={{ background: 'var(--surface)', borderRadius: 10, padding: '14px 16px 4px' }}>
                  <textarea
                    rows={2}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit(); }}
                    placeholder="Describe a part — dimensions, material, what it bolts onto…"
                    style={{ width: '100%', border: 'none', outline: 'none', background: 'transparent', resize: 'none', fontFamily: 'inherit', fontSize: 15, color: 'var(--text)', lineHeight: 1.5, minHeight: 48 }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', padding: '6px 8px 8px' }}>
                  <button
                    type="submit"
                    disabled={!prompt.trim()}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 8, height: 32, padding: '0 14px', borderRadius: 999, background: 'var(--text)', color: 'var(--bg)', border: 'none', cursor: 'pointer', fontSize: 12.5, fontWeight: 500, opacity: prompt.trim() ? 1 : 0.4 }}
                  >
                    Generate
                    <span style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,.12)', borderRadius: 4, padding: '0 5px', fontSize: 10 }}>⌘ ↵</span>
                  </button>
                </div>
              </div>
            </form>

            {/* Quick prompts */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 12, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 11.5, color: 'var(--text-4)', marginRight: 2 }}>Try:</span>
              {QUICK_PROMPTS.map(p => (
                <button
                  key={p}
                  onClick={() => { setPrompt(p); onOpenEditor({ model: null, prompt: p }); }}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 5, height: 26, padding: '0 11px', borderRadius: 999, background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-2)', fontSize: 12, cursor: 'pointer' }}
                  className="hover:bg-[var(--surface)] hover:border-[var(--border-strong)] hover:text-[var(--text)]"
                >
                  {p} <span style={{ color: 'var(--text-4)' }}>→</span>
                </button>
              ))}
            </div>

            {/* Two-col: models + activity */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, marginTop: 36 }}>
              {/* Recent models */}
              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
                  <h2 style={{ margin: 0, fontSize: 14.5, fontWeight: 600, letterSpacing: '-0.01em' }}>Recent models</h2>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    {(['all', 'parametric', 'organic'] as const).map(f => (
                      <button
                        key={f}
                        onClick={() => setFilter(f)}
                        style={{ padding: '4px 10px', borderRadius: 6, background: filter === f ? 'var(--surface)' : 'transparent', border: filter === f ? '1px solid var(--border)' : '1px solid transparent', color: filter === f ? 'var(--text)' : 'var(--text-3)', fontSize: 12, cursor: 'pointer' }}
                      >
                        {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
                        <span style={{ color: 'var(--text-4)', marginLeft: 4, fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>
                          {f === 'all' ? models.length : models.filter(m => m.route === f).length}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {filtered.length === 0 ? (
                  <div style={{ color: 'var(--text-4)', fontSize: 13, padding: '24px 0' }}>
                    No models yet — generate your first one above.
                  </div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                    {filtered.map(m => (
                      <ModelCard key={m.id} model={m} onClick={() => onOpenEditor({ model: m })} />
                    ))}
                  </div>
                )}
              </div>

              {/* Activity feed */}
              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
                  <h2 style={{ margin: 0, fontSize: 14.5, fontWeight: 600, letterSpacing: '-0.01em' }}>Activity</h2>
                </div>
                <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
                  {activity.length === 0 ? (
                    <div style={{ padding: '20px 14px', color: 'var(--text-4)', fontSize: 12.5 }}>No activity yet.</div>
                  ) : (
                    activity.slice(0, 8).map((a, i) => (
                      <ActivityItem key={a.id} item={a} last={i === Math.min(activity.length, 8) - 1} />
                    ))
                  )}
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}

/* ── Sub-components ── */

function NavItem({ icon, label, active, count }: { icon: React.ReactNode; label: string; active?: boolean; count?: number }) {
  return (
    <div
      style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', margin: '1px 4px', borderRadius: 7, fontSize: 13, color: active ? 'var(--text)' : 'var(--text-2)', cursor: 'pointer', fontWeight: active ? 500 : 400, background: active ? 'var(--surface)' : 'transparent', boxShadow: active ? 'inset 0 0 0 1px var(--border)' : 'none' }}
      className={active ? '' : 'hover:bg-black/[.04]'}
    >
      <span style={{ width: 14, height: 14, color: active ? 'var(--text)' : 'var(--text-3)', flexShrink: 0 }}>{icon}</span>
      {label}
      {count !== undefined && count > 0 && (
        <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-4)', background: 'rgba(0,0,0,.04)', padding: '1px 6px', borderRadius: 4 }}>
          {count}
        </span>
      )}
    </div>
  );
}


function ModelCard({ model, onClick }: { model: ModelRecord; onClick: () => void }) {
  const isParametric = model.route === 'parametric';
  return (
    <div
      onClick={onClick}
      style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden', cursor: 'pointer', transition: 'all .15s' }}
      className="hover:-translate-y-px hover:border-[var(--border-strong)] hover:shadow-md"
    >
      {/* Thumb */}
      <div style={{ aspectRatio: '4/3', background: 'var(--surface-2)', borderBottom: '1px solid var(--border)', position: 'relative', overflow: 'hidden' }}>
        {model.thumbnail ? (
          <img src={model.thumbnail} alt={model.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'grid', placeItems: 'center', background: isParametric ? 'var(--indigo-50)' : 'var(--terra-50)' }}>
            {isParametric ? <ParamIconLg /> : <MeshIconLg />}
          </div>
        )}
        <div style={{ position: 'absolute', top: 8, left: 8 }}>
          <RouteBadge route={model.route} />
        </div>
      </div>
      {/* Body */}
      <div style={{ padding: '10px 12px' }}>
        <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 2, letterSpacing: '-0.005em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{model.name}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', display: 'flex', gap: 6, alignItems: 'center' }}>
          <span>{timeAgo(model.createdAt)}</span>
          {model.paramCount !== undefined && model.paramCount > 0 && (
            <><span style={{ width: 3, height: 3, borderRadius: '50%', background: 'var(--text-4)', display: 'inline-block' }} /><span>{model.paramCount}p</span></>
          )}
          <span style={{ width: 3, height: 3, borderRadius: '50%', background: 'var(--text-4)', display: 'inline-block' }} />
          <span>v{model.version}</span>
        </div>
      </div>
    </div>
  );
}

function ActivityItem({ item, last }: { item: ActivityRecord; last: boolean }) {
  return (
    <div style={{ padding: '10px 14px', borderBottom: last ? 'none' : '1px solid var(--border)', display: 'flex', gap: 10, cursor: 'pointer' }}
      className="hover:bg-[var(--surface-2)]"
    >
      <div style={{ width: 24, height: 24, borderRadius: 5, background: 'var(--indigo-50)', color: 'var(--indigo)', display: 'grid', placeItems: 'center', flexShrink: 0, marginTop: 1 }}>
        <svg viewBox="0 0 14 14" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.8">
          {item.type === 'generated' ? <path d="M3 7l3 3 5-6" strokeLinecap="round" strokeLinejoin="round"/> : <path d="M7 9V3M4 6l3-3 3 3M3 11h8" strokeLinecap="round" strokeLinejoin="round"/>}
        </svg>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12.5, color: 'var(--text)', lineHeight: 1.4 }}>
          <b style={{ fontWeight: 600 }}>{item.modelName}</b> {item.type === 'generated' ? 'generated' : 'exported'}
        </div>
        {item.detail && <div style={{ fontSize: 11.5, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{item.detail}</div>}
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-4)', marginTop: 2 }}>{timeAgo(item.createdAt)}</div>
      </div>
    </div>
  );
}

function RouteBadge({ route }: { route: 'parametric' | 'organic' }) {
  const isP = route === 'parametric';
  return (
    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, padding: '2px 6px', background: 'rgba(255,255,255,.92)', border: '1px solid var(--border)', borderRadius: 4, color: isP ? 'var(--indigo)' : 'var(--terra)', backdropFilter: 'blur(4px)' }}>
      {isP ? 'parametric' : 'mesh'}
    </span>
  );
}

/* ── Icons ── */
const HomeIcon = () => <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 8l6-5 6 5v6H2z" strokeLinejoin="round"/></svg>;
const LibraryIcon = () => <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M2 7h12"/></svg>;
const DraftsIcon = () => <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 4h5l1.5 2H14v6H2z" strokeLinejoin="round"/></svg>;
const ParamIcon = () => <svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M3 6l5-3 5 3v4l-5 3-5-3z"/><path d="M3 6l5 3 5-3M8 9v4"/></svg>;
const MeshIcon = () => <svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="8" cy="8" r="5"/></svg>;
const ParamIconLg = () => <svg viewBox="0 0 32 32" width="32" height="32" fill="none" stroke="var(--indigo)" strokeWidth="1.6" opacity={0.4}><path d="M6 12l10-6 10 6v8l-10 6-10-6z"/><path d="M6 12l10 6 10-6M16 18v8"/></svg>;
const MeshIconLg = () => <svg viewBox="0 0 32 32" width="32" height="32" fill="none" stroke="var(--terra)" strokeWidth="1.6" opacity={0.4}><circle cx="16" cy="16" r="10"/></svg>;
