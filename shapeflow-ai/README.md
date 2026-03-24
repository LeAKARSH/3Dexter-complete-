# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/2b3da288-da58-489b-9b38-f97cb3b45cfe

## Run Locally

**Prerequisites:** Node.js, Python 3.8+

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Configure environment variables in [.env](.env) (already configured)

4. Run the app:
   ```bash
   npm run dev
   ```

That's it! Both models are loaded on-demand - no need to run separate services.

## Pipeline Notes

- **Organic mode** uses Shape-E for natural/organic shapes (animals, plants, etc.)
  - Models are loaded on-demand when generating organic shapes
  - Memory is released immediately after generation
- **Parametric mode** uses the local `openscad_lora_model_3b` LoRA fine-tuned model
  - Generates OpenSCAD code for parametric/geometric objects
  - Models are loaded on-demand when generating parametric shapes
  - Memory is released immediately after generation

- **Environment variables** configured in `.env`:
  **Parametric Model** (`openscad_lora_model_3b`):
- Base model: `unsloth/qwen2.5-coder-3b-bnb-4bit`
- Fine-tuning: LoRA adapter for OpenSCAD code generation
- Configuration: 4-bit quantization for efficient inference
- Task: Causal language modeling (code generation)

**Organic Model** (Shape-E):

- Pre-trained text-to-3D model from OpenAI
- Generates mesh files (.obj, .ply) from text prompts
- Optimized for natural and organic shapes

## Architecture

Both models use **lazy loading**:

- Models load only when a request is made
- Process the request
- Clear memory immediately after completion
- This allows both models to coexist without requiring excessive memory or background services
  The `openscad_lora_model_3b` model:

- Base model: `unsloth/qwen2.5-coder-3b-bnb-4bit`
- Fine-tuning: LoRA adapter for OpenSCAD code generation
- Configuration: 4-bit quantization for efficient inference
- Task: Causal language modeling (code generation)
