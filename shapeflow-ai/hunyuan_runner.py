"""
Hunyuan3D-2GP runner script for on-demand 3D mesh generation.
Models are loaded fresh on each request and released immediately after,
so memory is only used when needed.

After mesh generation, the repair engine is applied to fix common issues
like non-watertight meshes, holes, and degenerate geometry.

Hunyuan3D-2GP from: https://github.com/deepbeepmeep/Hunyuan3D-2GP
"""

import argparse
import gc
import json
import os
import sys
import torch

# Add parent directory to path for repair module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repair import RepairEngine, RepairConfig, Track


def run_hunyuan(text: str, output_dir: str, batch_size: int = 1):
    """
    Load models, generate meshes, unload models, return file paths.
    Everything is imported *inside* this function so that no model
    state leaks into module-level memory between calls.
    """
    # --- lazy imports ---
    from huggingface_hub import hf_hub_download
    from mglllm.utils.third_party.ollama import generate_3d
    from mglllm.utils.third_party.ply import save_mesh
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load models ---
    print(f"[hunyuan] Loading models on {device}", file=sys.stderr)
    
    # Download model files if not already present
    model_path = os.path.join(output_dir, "hunyuan3d-2gp")
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Download model from HuggingFace
        print(f"[hunyuan] Downloading model files...", file=sys.stderr)
        
        # The model consists of multiple files - download them
        model_files = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
        ]
        
        failed_downloads = []
        for fname in model_files:
            try:
                hf_hub_download(
                    repo_id="deepbeepmeep/Hunyuan3D-2GP",
                    filename=fname,
                    local_dir=model_path,
                )
            except Exception as e:
                failed_downloads.append((fname, str(e)))
        
        if failed_downloads:
            print(f"[hunyuan] Warning: {len(failed_downloads)} file(s) failed to download: {failed_downloads}", file=sys.stderr)
        
        # --- sample meshes ---
        print(f"[hunyuan] Generating for prompt: '{text}'", file=sys.stderr)
        
        # Generate mesh using Hunyuan3D-2GP
        # The generate_3d function handles the full pipeline
        result = generate_3d(
            prompt=text,
            model_path=model_path,
            device=device,
            guidance_scale=7.5,
            num_inference_steps=30,
        )
        
        # --- decode & save meshes ---
        saved_files = []
        repair_reports = []
        
        # Initialize repair engine for organic meshes
        repair_config = RepairConfig(
            track=Track.ORGANIC,
            assembly=False,
            normalize_scale=True,
            target_longest_edge_mm=100.0,
            up_axis="Z",
        )
        repair_engine = RepairEngine(repair_config)
        
        # Process generated mesh(es)
        # The result could be a single mesh or a list
        meshes = result if isinstance(result, list) else [result]
        
        if not meshes or (len(meshes) == 1 and meshes[0] is None):
            raise ValueError("Hunyuan generation returned no meshes")
        
        successful_meshes = 0
        for i, mesh_data in enumerate(meshes):
            # Extract mesh from result (format depends on generate_3d output)
            if hasattr(mesh_data, 'mesh'):
                mesh = mesh_data.mesh
            elif hasattr(mesh_data, 'vertices'):
                # Already a trimesh object
                import trimesh
                mesh = mesh_data
            else:
                print(f"[hunyuan] Unknown mesh format for mesh {i}", file=sys.stderr)
                continue
            
            # Apply repair engine to fix mesh issues
            print(f"[hunyuan] Repairing mesh {i}...", file=sys.stderr)
            repair_result = repair_engine.repair(mesh)
            
            if not repair_result.ok:
                print(f"[hunyuan] Repair warning for mesh {i}: {repair_result.error}", file=sys.stderr)
                repaired_mesh = mesh
            else:
                repaired_mesh = repair_result.mesh
                repair_reports.append(repair_result.report)
                print(f"[hunyuan] Mesh {i} repaired: watertight={repair_result.report['after']['watertight']}, "
                      f"holes={repair_result.report['after']['holes']}", file=sys.stderr)

            ply_name = f"mesh_{i}.ply"
            obj_name = f"mesh_{i}.obj"
            stl_name = f"mesh_{i}.stl"

            ply_path = os.path.join(output_dir, ply_name)
            obj_path = os.path.join(output_dir, obj_name)
            stl_path = os.path.join(output_dir, stl_name)

            # Save in multiple formats
            with open(ply_path, "wb") as f:
                repaired_mesh.write_ply(f)
            with open(obj_path, "w") as f:
                repaired_mesh.write_obj(f)
            repaired_mesh.export(stl_path)

            saved_files.append(ply_name)
            saved_files.append(obj_name)
            saved_files.append(stl_name)
            successful_meshes += 1
            print(f"[hunyuan] Saved {ply_name}, {obj_name}, and {stl_name}", file=sys.stderr)
        
        if successful_meshes == 0:
            raise ValueError("No meshes could be processed successfully")

        return saved_files, repair_reports

    finally:
        # --- unload models regardless of success/failure ---
        print("[hunyuan] Unloading models", file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[hunyuan] Models cleared", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hunyuan3D-2GP runner for on-demand mesh generation")
    parser.add_argument("--prompt", required=True, help="Text prompt for 3D mesh generation")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated files")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Generate meshes (includes repair)
        files, repair_reports = run_hunyuan(args.prompt, args.output_dir, args.batch_size)
        
        # Return formatted response with repair metrics
        response = {
            "files": files,
            "message": "Generated by Hunyuan3D-2GP with mesh repair",
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
