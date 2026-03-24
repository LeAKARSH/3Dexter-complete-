import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Local parametric model runner")
    parser.add_argument("--model", required=True, help="Path to the local fine-tuned model file or directory")
    parser.add_argument("--prompt", required=True, help="Parametric modeling prompt")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)

    if not os.path.exists(model_path):
        print(json.dumps({
            "error": f"Model path does not exist: {model_path}"
        }), file=sys.stderr)
        return 1

    # Replace this stub with your actual model loading and inference code.
    # The app expects JSON on stdout shaped like:
    # { "code": "...", "message": "..." }
    response = {
        "code": (
            "// Local parametric runner placeholder\n"
            f"// Model path: {model_path}\n"
            f"// Prompt: {args.prompt}\n"
            "size = 20;\n"
            "cube([size, size, size], center=true);\n"
        ),
        "message": "Placeholder local parametric runner executed. Replace parametric_runner.py with your model logic.",
    }

    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
