import { useState, useEffect } from 'react';
import { HomeScreen } from './HomeScreen';
import { EditorScreen } from './EditorScreen';
import type { AppScreen, EditorEntry, ModelRecord } from './types';

export default function App() {
  const [screen, setScreen] = useState<AppScreen>('home');
  const [editorEntry, setEditorEntry] = useState<EditorEntry>({ model: null });
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark');
  const [parametricBackend, setParametricBackend] = useState<'local' | 'gemini'>(
    () => (localStorage.getItem('parametricBackend') as 'local' | 'gemini') ?? 'local'
  );

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  }, [dark]);

  useEffect(() => {
    localStorage.setItem('parametricBackend', parametricBackend);
  }, [parametricBackend]);

  const openEditor = (entry: EditorEntry) => {
    setEditorEntry({ ...entry, generationBackend: parametricBackend === 'gemini' ? 'gemini' : undefined });
    setScreen('editor');
  };

  const handleModelSaved = (model: ModelRecord) => {
    setEditorEntry(prev => ({ ...prev, model }));
  };

  if (screen === 'editor') {
    return (
      <EditorScreen
        entry={editorEntry}
        dark={dark}
        onToggleDark={() => setDark(d => !d)}
        onHome={() => setScreen('home')}
        onModelSaved={handleModelSaved}
      />
    );
  }

  return (
    <HomeScreen
      dark={dark}
      onToggleDark={() => setDark(d => !d)}
      onOpenEditor={openEditor}
      parametricBackend={parametricBackend}
      onParametricBackendChange={setParametricBackend}
    />
  );
}
