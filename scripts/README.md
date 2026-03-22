Helper scripts to download Llama-2-13b and start a minimal HTTP server.

Usage:

1) Download model (requires Hugging Face token in `hf_token.txt`):

```bash
python scripts/download_model.py
```

2) Start the local HTTP server (serves POST /v1/generate):

```bash
python scripts/start_llm_server.py
```

The server loads the model with `torch_dtype=float16` and `device_map='auto'`.
Ensure GPUs and CUDA drivers are available.
