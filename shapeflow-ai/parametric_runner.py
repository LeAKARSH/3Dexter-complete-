import argparse
import json
import os
import sys
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def resolve_base_model_name(model_path: str) -> str:
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            if config.get("base_model_name_or_path"):
                return str(config["base_model_name_or_path"])
        except Exception as exc:
            print(f"[parametric] Warning: failed reading adapter_config.json: {exc}", file=sys.stderr)
    # Fallback for older adapters / missing config metadata
    return "unsloth/qwen2.5-coder-3b-bnb-4bit"


def generate_openscad_code(model_path: str, prompt: str) -> str:
    """
    Load model, generate OpenSCAD code, unload model, return code.
    Models are loaded fresh on each request and released immediately after,
    so memory is only used when needed.
    """
    try:
        base_model_name = resolve_base_model_name(model_path)
        
        print(f"[parametric] Loading base model: {base_model_name}", file=sys.stderr)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Ensure pad token is set properly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Create offload directory for model layers that don't fit in memory
        offload_folder = os.path.join(os.path.dirname(model_path), "model_offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        # Model is already quantized (bnb-4bit), load without additional quantization
        print("[parametric] Loading pre-quantized model", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapter
        print(f"[parametric] Loading LoRA adapter from: {model_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        
        print("[parametric] Model loaded successfully", file=sys.stderr)
        
        # Format the prompt using Qwen2.5 chat format (manual construction)
        # The model expects: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
        formatted_prompt = f"<|im_start|>system\nYou are an OpenSCAD code generator. Generate ONLY valid OpenSCAD code. Do not include explanations, comments about the code, or examples.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        print(f"[parametric] Generating code for prompt: '{prompt}'", file=sys.stderr)
        
        # Get special token IDs to stop generation early
        stop_token_ids = [tokenizer.eos_token_id]
        # Add code-specific stop tokens if they exist
        for stop_token in ["<|repo_name|>", "<|file_sep|>", "<|fim_prefix|>"]:
            if stop_token in tokenizer.get_vocab():
                stop_token_ids.append(tokenizer.convert_tokens_to_ids(stop_token))
        
        # Generate with optimized parameters for code generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Reduced to prevent including training examples
                temperature=0.2,  # Even lower for more deterministic code
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.2,  # Increased to prevent repetitive patterns
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        
        # Decode the generated code
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)  # Keep special tokens to identify them
        
        # Extract only the assistant's response (the OpenSCAD code)
        if "<|im_start|>assistant" in generated_text:
            code = generated_text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in code:
                code = code.split("<|im_end|>")[0].strip()
        else:
            # Fallback: remove the prompt part
            code = generated_text[len(formatted_prompt):].strip()
        
        # Clean up special tokens from code training (fim = fill-in-middle, repo_name, etc.)
        special_token_markers = [
            "<|repo_name|>", "<|file_sep|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"
        ]
        for marker in special_token_markers:
            if marker in code:
                code = code.split(marker)[0].strip()
        
        # Clean up training artifacts - stop at common training conversation markers
        training_markers = ["Human:", "Assistant:", "User:", "Question:", "import os", "def generate_", "print("]
        for marker in training_markers:
            if marker in code:
                code = code.split(marker)[0].strip()
        
        # Clean up any markdown code blocks if present
        if code.startswith("```openscad") or code.startswith("```scad") or code.startswith("```"):
            lines = code.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = '\n'.join(lines).strip()
        
        # Remove any leading incomplete variable assignments (like "ount = 3;")
        lines = code.split('\n')
        # Find first line that looks like a complete statement
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if line starts with a valid OpenSCAD keyword or complete variable name
            if stripped and (
                stripped.startswith('module ') or 
                stripped.startswith('function ') or
                stripped.startswith('$') or
                stripped.startswith('//') or
                (not stripped.startswith('=') and '=' in stripped and stripped[0].isalpha())
            ):
                code = '\n'.join(lines[i:])
                break
        
        print(f"[parametric] Generated {len(code)} characters of OpenSCAD code", file=sys.stderr)
        
        return code
        
    finally:
        # --- unload models regardless of success/failure ---
        print("[parametric] Unloading models", file=sys.stderr)
        try:
            del model, tokenizer, inputs, outputs
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[parametric] Models cleared", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local parametric model runner")
    parser.add_argument("--model", required=True, help="Path to the LoRA adapter model")
    parser.add_argument("--prompt", required=True, help="Parametric modeling prompt")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)

    if not os.path.exists(model_path):
        print(json.dumps({
            "error": f"Model path does not exist: {model_path}"
        }), file=sys.stderr)
        return 1

    try:
        # Generate code (model is loaded and unloaded inside this function)
        generated_code = generate_openscad_code(model_path, args.prompt)
        
        # Return formatted response
        response = {
            "code": generated_code,
            "message": "Generated by openscad_lora_model_3b",
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
