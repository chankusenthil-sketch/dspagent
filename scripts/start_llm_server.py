"""Start a FastAPI HTTP server that serves a local HF model for text generation.

Supports both base models (raw prompt) and instruct models (chat template).
Set MODEL_DIR env var to point to the model directory.
Default: Mistral-7B-Instruct-v0.3.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
import torch


import os
import sys

# Model directory can be set via the MODEL_DIR env var, otherwise a default path is used.
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(os.path.dirname(__file__), "..", "models", "Mistral-7B-Instruct-v0.3")


def find_model_dir(root: str, max_depth: int = 3) -> str | None:
    """Search recursively under `root` for a directory that contains model files (config.json/tokenizer_config.json).

    Returns the first matching directory or None.
    """
    for depth in range(max_depth + 1):
        # breadth-first-ish: check root, then its immediate subdirs, etc.
        if os.path.exists(os.path.join(root, "config.json")) or os.path.exists(os.path.join(root, "tokenizer_config.json")) or os.path.exists(os.path.join(root, "pytorch_model.bin.index.json")):
            return root
        # look for snapshots subdir
        snapshots = os.path.join(root, "snapshots")
        if os.path.isdir(snapshots):
            # pick first snapshot folder
            subs = sorted(os.listdir(snapshots))
            for s in subs:
                candidate = os.path.join(snapshots, s)
                if os.path.isdir(candidate) and (os.path.exists(os.path.join(candidate, "config.json")) or os.path.exists(os.path.join(candidate, "tokenizer_config.json"))):
                    return candidate
        # otherwise dive one level
        subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if not subdirs:
            break
        # pick first and continue
        root = subdirs[0]
    return None


resolved = find_model_dir(MODEL_DIR)
if resolved:
    MODEL_DIR = resolved
else:
    # leave as-is; tokenizer will raise clearer error
    pass


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.3
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2


app = FastAPI()

tokenizer = None
is_instruct_model = False


@app.on_event("startup")
def load_model():
    global generator, tokenizer, is_instruct_model
    print("Loading tokenizer and model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("CUDA GPU not available — this server requires a GPU to run.")

    dtype = torch.float16
    generator = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Detect if this is an instruct/chat model
    model_name = MODEL_DIR.lower()
    is_instruct_model = any(kw in model_name for kw in ["instruct", "chat", "it-"])
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        is_instruct_model = True

    print(f"Model loaded. Instruct mode: {is_instruct_model}. Ready to serve.")


@app.post("/v1/generate")
def generate(req: GenRequest):
    # For instruct models, wrap the raw prompt in the chat template
    if is_instruct_model and tokenizer is not None:
        messages = [{"role": "user", "content": req.prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = req.prompt

    gen = generator(
        formatted,
        max_new_tokens=req.max_tokens,
        do_sample=(req.temperature > 0.0),
        temperature=req.temperature if req.temperature > 0.0 else None,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        repetition_penalty=req.repetition_penalty,
    )
    full_text = gen[0].get("generated_text", "")
    # Strip the prompt portion to return only the assistant's response
    if full_text.startswith(formatted):
        generated = full_text[len(formatted):].strip()
    else:
        generated = full_text
    return {"generated_text": generated}


@app.post("/v1/chat")
def chat(req: ChatRequest):
    """Chat endpoint using the model's chat template."""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if is_instruct_model and tokenizer is not None:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: concatenate messages
        formatted = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    gen = generator(
        formatted,
        max_new_tokens=req.max_tokens,
        do_sample=(req.temperature > 0.0),
        temperature=req.temperature if req.temperature > 0.0 else None,
        no_repeat_ngram_size=req.no_repeat_ngram_size,
        repetition_penalty=req.repetition_penalty,
    )
    full_text = gen[0].get("generated_text", "")
    if full_text.startswith(formatted):
        generated = full_text[len(formatted):].strip()
    else:
        generated = full_text
    return {"role": "assistant", "content": generated}


@app.get("/health")
def health():
    """Simple health check endpoint used by automated readiness checks."""
    return {"status": "ok", "model_dir": MODEL_DIR, "instruct_mode": is_instruct_model}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
