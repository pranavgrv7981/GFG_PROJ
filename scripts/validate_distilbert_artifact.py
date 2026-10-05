import time
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def validate():
    model_dir = Path("models/distilbert/gfg_distilbert_sentiment")
    if not model_dir.exists():
        model_dir = Path("models/distilbert")

    print(f"Inspecting directory: {model_dir}")
    files = list(model_dir.glob("*"))
    for f in files:
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    # 1. Tokenizer loading
    try:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        t_tok = time.time() - t0
        print(f"Tokenizer loaded successfully in {t_tok:.3f}s: {type(tokenizer).__name__}")
        tok_ok = True
    except Exception as e:
        print(f"Tokenizer loading FAILED: {e}")
        tok_ok = False

    # 2. Model loading
    try:
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        t_model = time.time() - t0
        print(f"Model loaded successfully in {t_model:.3f}s")
        model_ok = True
    except Exception as e:
        print(f"Model loading FAILED: {e}")
        model_ok = False

    if not (tok_ok and model_ok):
        print("CANNOT PROCEED TO INFERENCE.")
        return

    # 3. Label mapping validation
    id2label = model.config.id2label
    print(f"Config id2label mapping: {id2label}")
    label_ok = (id2label.get(0) == "negative" or id2label.get("0") == "negative") and \
               (id2label.get(1) == "neutral" or id2label.get("1") == "neutral") and \
               (id2label.get(2) == "positive" or id2label.get("2") == "positive")
    print(f"Label mapping check: {'PASS' if label_ok else 'FAIL'}")

    # Standardize integer keys
    normalized_id2label = {int(k): v for k, v in id2label.items()}

    # 4. Device detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"CUDA available: {cuda_available} | Using device: {device}")
    model.to(device)
    model.eval()

    # 5. Single sentence CPU/device inference
    sample_texts = [
        "I absolutely love this product! Super fast delivery and amazing quality.",
        "Tracking says order is delayed by 5 days. Terrible service and rude staff.",
        "What is the return window for this order?"
    ]

    print("\n--- Single Sentence Inference ---")
    cpu_ok = True
    for text in sample_texts:
        t0 = time.time()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            top_idx = torch.argmax(probs).item()
            pred_label = normalized_id2label[top_idx]
            conf = probs[top_idx].item()
        dur = (time.time() - t0) * 1000
        print(f"Input: \"{text}\"")
        print(f"  -> Prediction: {pred_label} (Confidence: {conf:.4f}, Latency: {dur:.1f}ms)")
        if conf <= 0.0 or conf > 1.0:
            cpu_ok = False

    # 6. Batch inference
    print("\n--- Batch Inference ---")
    t0 = time.time()
    batch_inputs = tokenizer(sample_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        batch_outputs = model(**batch_inputs)
        batch_probs = F.softmax(batch_outputs.logits, dim=-1)
        batch_preds = torch.argmax(batch_probs, dim=-1)
    batch_dur = (time.time() - t0) * 1000

    batch_ok = len(batch_preds) == len(sample_texts)
    for i, text in enumerate(sample_texts):
        p_idx = batch_preds[i].item()
        p_conf = batch_probs[i][p_idx].item()
        print(f"Batch item [{i}]: {normalized_id2label[p_idx]} ({p_conf:.4f})")
    print(f"Batch throughput: {len(sample_texts)} sentences processed in {batch_dur:.1f}ms")

    # 7. Model file size
    safetensors_file = model_dir / "model.safetensors"
    model_size_mb = safetensors_file.stat().st_size / (1024 * 1024) if safetensors_file.exists() else 0
    print(f"\nModel weight file size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    validate()
