"""
Shape-E runner script for on-demand 3D mesh generation.
Models are loaded fresh on each request and released immediately after,
so memory is only used when needed.

After mesh generation, the repair engine is applied to fix common issues
like non-watertight meshes, holes, and degenerate geometry.
"""

import argparse
import gc
import json
import os
import sys
import uuid
import torch

from repair import RepairEngine, RepairConfig, Track


def run_shap_e(text: str, output_dir: str, batch_size: int = 1):
    """
    Load models, generate meshes, unload models, return file paths.
    Everything is imported *inside* this function so that no shap-e /
    torch state leaks into module-level memory between calls.
    """
    # --- lazy imports so nothing loads at startup ---
    import trimesh
    
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load models ---
    print(f"[shap-e] Loading models on {device}", file=sys.stderr)
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    try:
        # --- sample latents ---
        print(f"[shap-e] Sampling for prompt: '{text}'", file=sys.stderr)
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
        repair_reports = []
        
        # Initialize repair engine for organic meshes (Shape-E generates neural diffusion meshes)
        repair_config = RepairConfig(
            track=Track.ORGANIC,
            assembly=False,
            normalize_scale=True,
            target_longest_edge_mm=100.0,
            up_axis="Z",
        )
        repair_engine = RepairEngine(repair_config)
        
        run_id = uuid.uuid4().hex[:8]

        for i, latent in enumerate(latents):
            tri = decode_latent_mesh(xm, latent).tri_mesh()

            # Apply repair engine to fix mesh issues
            print(f"[shap-e] Repairing mesh {i}...", file=sys.stderr)
            repair_result = repair_engine.repair(tri)
            
            # Ensure we always have a trimesh.Trimesh object for export
            if not repair_result.ok:
                print(f"[shap-e] Repair warning for mesh {i}: {repair_result.error}", file=sys.stderr)
                # Ensure original mesh is a trimesh object before using as fallback
                if not isinstance(tri, trimesh.Trimesh):
                    if hasattr(tri, 'vertices') and hasattr(tri, 'faces'):
                        tri = trimesh.Trimesh(vertices=tri.vertices, faces=tri.faces, process=False)
                    else:
                        print(f"[shap-e] Cannot use mesh {i} - not a valid trimesh object", file=sys.stderr)
                        continue
                repaired_mesh = tri
            else:
                repaired_mesh = repair_result.mesh
                repair_reports.append(repair_result.report)
                print(f"[shap-e] Mesh {i} repaired: watertight={repair_result.report['after']['watertight']}, "
                      f"holes={repair_result.report['after']['holes']}", file=sys.stderr)

            ply_name = f"shapee_{run_id}_{i}.ply"
            obj_name = f"shapee_{run_id}_{i}.obj"
            stl_name = f"shapee_{run_id}_{i}.stl"

            ply_path = os.path.join(output_dir, ply_name)
            obj_path = os.path.join(output_dir, obj_name)
            stl_path = os.path.join(output_dir, stl_name)

            with open(ply_path, "wb") as f:
                repaired_mesh.write_ply(f)
            with open(obj_path, "w") as f:
                repaired_mesh.write_obj(f)
            # Export repaired mesh as STL for 3D printing
            repaired_mesh.export(stl_path)

            saved_files.append(ply_name)
            saved_files.append(obj_name)
            saved_files.append(stl_name)
            print(f"[shap-e] Saved {ply_name}, {obj_name}, and {stl_name}", file=sys.stderr)

        return saved_files, repair_reports

    finally:
        # --- unload models regardless of success/failure ---
        print("[shap-e] Unloading models", file=sys.stderr)
        try:
            del model, xm, diffusion, latents
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[shap-e] Models cleared", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shape-E runner for on-demand mesh generation")
    parser.add_argument("--prompt", required=True, help="Text prompt for 3D mesh generation")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Generate meshes (includes repair)
        files, repair_reports = run_shap_e(args.prompt, args.output_dir)
        
        # Return formatted response with repair metrics
        response = {
            "files": files,
            "message": "Generated by Shape-E with mesh repair",
            "repair_reports": repair_reports,
            "mesh_count": len(files) // 3,  # 3 files per mesh (ply, obj, stl)
        }
        
        print(json.dumps(response))
        return 0
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "error": f"Generation failed: {str(e)}"
        }), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
