# ShapeFlow AI

ShapeFlow AI routes prompts into two pipelines:
- Organic mesh generation (`Shape-E` or `HUNYUAN3D-2GP`)
- Parametric OpenSCAD code generation (local LoRA adapter)

## What You Need

- Node.js 18+
- Python 3.10+
- A local parametric model folder (LoRA adapter), set via `PARAMETRIC_MODEL_PATH`

## Quick Start

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install Node dependencies:

```bash
npm install
```

3. Create `.env` from `.env.example` and set:

```env
PARAMETRIC_MODEL_PATH=C:\path\to\openscad_lora_model_3b
```

4. Start the app:

```bash
npm run dev
```

5. Open `http://localhost:3000`

## Runtime Notes

- Organic models are loaded on demand and released after each request.
- Parametric generation uses your local model path from `.env`.
- `repair.py` is used by organic runners to normalize and repair meshes before export.
- The UI includes:
  - Organic model toggle (`Shape-E` / `HUNYUAN`)
  - Pipeline output panel (organic + parametric)
  - Generation progress bar

## API Endpoints

- `POST /api/route` - routes prompt to parametric or organic pipeline
- `GET /api/config` - reports config status (including local parametric model path detection)
- `GET /api/organic/view/:filename` - preview mesh files in the viewer
- `GET /api/organic/download/:filename` - download generated mesh files
