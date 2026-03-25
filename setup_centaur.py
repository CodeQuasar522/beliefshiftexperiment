"""
Setup script for Centaur-70B in ollama.
Downloads the LoRA adapter from HuggingFace, converts to GGUF, and registers in ollama.

Requirements: ollama must be running, ~45GB disk space for the 70B base model.
"""
import subprocess
import os
import sys

ADAPTER_REPO = "marcelbinz/Llama-3.1-Centaur-70B-adapter"
ADAPTER_DIR = "./centaur-adapter"
GGUF_FILE = "centaur-adapter.gguf"
OLLAMA_BASE = "llama3.1:70b"
OLLAMA_MODEL = "centaur"
MODELFILE = "Modelfile.centaur"

def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)
    return result

def main():
    print("=" * 60)
    print("Centaur-70B Setup for ollama")
    print("=" * 60)

    # Step 1: Install Python dependencies
    print("\n[1/6] Installing Python dependencies...")
    run("pip3 install huggingface_hub gguf safetensors")

    # Step 2: Download adapter from HuggingFace
    print(f"\n[2/6] Downloading Centaur adapter from {ADAPTER_REPO}...")
    if os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        print("  Already downloaded, skipping.")
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(ADAPTER_REPO, local_dir=ADAPTER_DIR)
        print(f"  Downloaded to {ADAPTER_DIR}")

    # Step 3: Get llama.cpp conversion script
    print("\n[3/6] Getting llama.cpp conversion tools...")
    if not os.path.exists("llama.cpp"):
        run("git clone --depth 1 https://github.com/ggml-org/llama.cpp.git")
    else:
        print("  llama.cpp already cloned, skipping.")

    # Step 4: Convert adapter to GGUF
    print(f"\n[4/6] Converting adapter to GGUF format...")
    if os.path.exists(GGUF_FILE):
        print(f"  {GGUF_FILE} already exists, skipping.")
    else:
        # The conversion script is at the root of llama.cpp
        convert_script = "llama.cpp/convert_lora_to_gguf.py"
        if not os.path.exists(convert_script):
            # Try alternative locations
            for alt in ["llama.cpp/scripts/convert_lora_to_gguf.py",
                        "llama.cpp/gguf-py/scripts/convert_lora_to_gguf.py"]:
                if os.path.exists(alt):
                    convert_script = alt
                    break
        run(f"python3 {convert_script} {ADAPTER_DIR} --outfile {GGUF_FILE}")

    # Step 5: Pull base model in ollama
    print(f"\n[5/6] Pulling base model ({OLLAMA_BASE})...")
    print("  This may take a while for the 70B model (~40GB).")
    run(f"ollama pull {OLLAMA_BASE}")

    # Step 6: Create centaur model in ollama
    print(f"\n[6/6] Creating '{OLLAMA_MODEL}' model in ollama...")
    gguf_abs = os.path.abspath(GGUF_FILE)
    modelfile_content = f"""FROM {OLLAMA_BASE}
ADAPTER {gguf_abs}
"""
    with open(MODELFILE, "w") as f:
        f.write(modelfile_content)
    print(f"  Modelfile written to {MODELFILE}")
    run(f"ollama create {OLLAMA_MODEL} -f {MODELFILE}")

    # Verify
    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Centaur model registered as '{OLLAMA_MODEL}' in ollama.")
    print("Run:  python shift3_centaur.py -test")
    print("=" * 60)

if __name__ == "__main__":
    main()
