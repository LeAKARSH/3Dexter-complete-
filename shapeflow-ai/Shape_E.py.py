"""
Shap-E Text-to-3D API
Endpoint: POST http://localhost:5010/shap-e
Body:     { "text": "a shark" }
Returns:  { "files": ["example_mesh_0.ply", "example_mesh_0.obj", ...] }

Models are loaded fresh on each request and released immediately after,
so this API plays nicely alongside other GPU-heavy local services.
"""

import gc
import os
import tempfile
import traceback

import torch
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# Directory where generated mesh files will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "shap_e_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_shap_e(text: str, batch_size: int = 1):
    """
    Load models, generate meshes, unload models, return file paths.
    Everything is imported *inside* this function so that no shap-e /
    torch state leaks into module-level memory between calls.
    """
    # --- lazy imports so nothing loads at startup ---
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load models ---
    print(f"[shap-e] Loading models on {device} …")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    try:
        # --- sample latents ---
        print(f"[shap-e] Sampling for prompt: '{text}'")
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[text] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=16,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        # --- decode & save meshes ---
        saved_files = []

        import trimesh

        for i, latent in enumerate(latents):
            tri = decode_latent_mesh(xm, latent).tri_mesh()

            ply_name = f"mesh_{i}.ply"
            obj_name = f"mesh_{i}.obj"

            ply_path = os.path.join(OUTPUT_DIR, ply_name)
            obj_path = os.path.join(OUTPUT_DIR, obj_name)

            # Save original mesh
            with open(ply_path, "wb") as f:
                tri.write_ply(f)
            with open(obj_path, "w") as f:
                tri.write_obj(f)

            # --- Mesh repair using trimesh ---
            try:
                mesh = trimesh.load(ply_path, force='mesh')
                if not mesh.is_watertight or not mesh.is_winding_consistent:
                    mesh = mesh.fill_holes()
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()
                    mesh.remove_infinite_values()
                    mesh.remove_unreferenced_vertices()
                    mesh.process(validate=True)
                    mesh.export(ply_path)
                    mesh.export(obj_path)
                    print(f"[shap-e] Mesh repaired for {ply_name} and {obj_name}")
            except Exception as repair_exc:
                print(f"[shap-e] Mesh repair failed: {repair_exc}")

            saved_files.append(ply_name)
            saved_files.append(obj_name)
            print(f"[shap-e] Saved {ply_name} and {obj_name}")

        return saved_files

    finally:
        # --- unload models regardless of success/failure ---
        print("[shap-e] Unloading models …")
        del model, xm, diffusion, latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[shap-e] Models cleared.")


@app.route("/shap-e", methods=["POST"])
def shap_e_endpoint():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "'text' parameter is required and must not be empty."}), 400

    try:
        files = run_shap_e(text)
        # Find obj and ply files
        obj_file = next((f for f in files if f.endswith('.obj')), None)
        ply_file = next((f for f in files if f.endswith('.ply')), None)
        obj_url = request.host_url.rstrip("/") + f"/shap-e/download/{obj_file}" if obj_file else None
        ply_url = request.host_url.rstrip("/") + f"/shap-e/download/{ply_file}" if ply_file else None
        return jsonify({
            "prompt": text,
            "files": files,
            "objFile": obj_file,
            "plyFile": ply_file,
            "objUrl": obj_url,
            "plyUrl": ply_url,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/shap-e/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Convenience endpoint to download a generated mesh file."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    print("Starting Shap-E API on http://localhost:5010")
    print("POST /shap-e  — body: { \"text\": \"<your prompt>\" }")
    app.run(host="0.0.0.0", port=5010, debug=False)