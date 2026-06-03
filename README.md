# 3Dexter

Text-to-3D model generator. Type a description, get a 3D model — parametric (OpenSCAD) or organic (mesh).

Prompts are auto-classified into one of two pipelines:

- **Parametric** — generates OpenSCAD code, compiles to a real 3D preview, exposes editable parameters with sliders
- **Organic** — runs Shape-E locally to produce a PLY/OBJ mesh for natural/freeform shapes

---

## Prerequisites

| | Mac | Windows |
|---|---|---|
| Node.js | 18+ | 18+ |
| Python | 3.10+ | 3.10+ |
| OpenSCAD | optional (for parametric preview) | optional (for parametric preview) |
| Gemini API key | optional (for cloud parametric) | optional (for cloud parametric) |

You need **at least one** parametric backend configured (Gemini API key or local LoRA adapter) and/or Shape-E installed for organic generation.

---

## Setup — macOS

### 1. Clone and install

```bash
git clone <repo>
cd 3Dexter-complete-
npm install --prefix shapeflow-ai
pip install -r shapeflow-ai/requirements.txt
```

### 2. Configure environment

```bash
cp shapeflow-ai/.env.example shapeflow-ai/.env
```

Open `shapeflow-ai/.env` and fill in what you need:

```env
# For Gemini parametric generation (recommended — no local model needed)
GEMINI_API_KEY=your_key_here

# For local LoRA parametric generation (optional)
PARAMETRIC_MODEL_PATH=/path/to/openscad_lora_model_3b

# Server port (default: 3000)
PORT=3000
```

Get a free Gemini API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 3. Install OpenSCAD (for parametric 3D preview)

```bash
brew install --cask openscad
```

Or download from [openscad.org/downloads](https://openscad.org/downloads.html). The server detects it automatically — no config needed.

### 4. Run

```bash
npm run dev
# or with a custom port:
PORT=8080 npm run dev
```

Open `http://localhost:3000` (or whichever port you set).

---

## Setup — Windows

### 1. Clone and install

```cmd
git clone <repo>
cd 3Dexter-complete-
npm install --prefix shapeflow-ai
pip install -r shapeflow-ai/requirements.txt
```

### 2. Configure environment

```cmd
copy shapeflow-ai\.env.example shapeflow-ai\.env
```

Open `shapeflow-ai\.env` and fill in:

```env
# For Gemini parametric generation (recommended — no local model needed)
GEMINI_API_KEY=your_key_here

# For local LoRA parametric generation (optional)
PARAMETRIC_MODEL_PATH=C:\path\to\openscad_lora_model_3b

# Server port (default: 3000)
PORT=3000
```

### 3. Install OpenSCAD (for parametric 3D preview)

Download the installer from [openscad.org/downloads](https://openscad.org/downloads.html) and run it. It installs to `C:\Program Files\OpenSCAD\` — the server detects it automatically.

### 4. Run

```cmd
set PORT=8080
npm run dev
```

Or in PowerShell:

```powershell
$env:PORT="8080"; npm run dev
```

Open `http://localhost:3000` (or your port).

---

## What runs where

| Feature | Requires |
|---|---|
| **Gemini parametric** | `GEMINI_API_KEY` in `.env`. API call — works on any machine, no GPU needed |
| **Local LoRA parametric** | `PARAMETRIC_MODEL_PATH` pointing to adapter files + GPU recommended |
| **Organic (Shape-E)** | `pip install -r requirements.txt` + GPU recommended (CPU works, slow) |
| **Parametric 3D preview** | OpenSCAD installed |
| **Parameter sliders** | Always available — edit live, click "Apply & re-render" to see the result |

If you only have a Gemini API key and OpenSCAD, everything except organic generation works fully with no GPU or local model.

---

## Parametric backend toggle

When `GEMINI_API_KEY` is set, a **"Parametric backend"** toggle appears in the sidebar:

- **Local LoRA** — uses the on-device model (requires `PARAMETRIC_MODEL_PATH`)
- **Gemini** — calls the Gemini API (faster, better code quality, requires internet)

The auto-classifier always decides whether a prompt goes to the parametric or organic pipeline. Gemini only affects *how* parametric models are generated, not whether they are.

---

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /api/route` | Generate a model from a prompt |
| `POST /api/render-scad` | Compile OpenSCAD code to STL |
| `GET /api/models` | List saved models |
| `GET /api/activity` | Generation activity log |
| `PATCH /api/models/:id/thumbnail` | Save a canvas thumbnail |
| `GET /api/config` | Server capability report |
| `GET /api/organic/view/:filename` | Stream a mesh file for preview |
| `GET /api/organic/download/:filename` | Download a mesh file |
