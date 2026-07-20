#!/bin/bash
# cf#29 D1 non-interactive driver: build isolated aitoolkit env, stage Wan base, run the REAL merged
# train_lora+wan path through the worker's run_job, harvest two experts to R2, verify. Logs stream to
# /var/log/d1.log, served over the RunPod HTTP proxy on :19123 so progress is watchable without SSH.
exec > >(stdbuf -oL tee -a /var/log/d1.log) 2>&1
export PATH=/opt/conda/bin:$PATH
cd /var/log && stdbuf -oL python3 -m http.server 19123 >/dev/null 2>&1 &
cd /root
echo "=================== D1 START ==================="
AITK_REF=6e158dd1f1552b73b7aca6d7ddaa46a783538052

echo "--- cred presence (names only) ---"
for v in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ENDPOINT R2_BUCKET HF_TOKEN; do eval "echo PRESENCE $v=\${$v:+SET}"; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== PHASE1: clone branch + ai-toolkit ==="
rm -rf /root/vjb && git clone -q -b feat/wan-lora-train-image-cf29 https://github.com/skyphusion-labs/vivijure-backend /root/vjb && echo "CLONE_BRANCH_OK $(git -C /root/vjb rev-parse --short HEAD)" || { echo "CLONE_BRANCH_FAIL"; }
rm -rf /opt/ai-toolkit && git clone -q https://github.com/ostris/ai-toolkit /opt/ai-toolkit && git -C /opt/ai-toolkit checkout -q $AITK_REF && echo "CLONE_AITK_OK $(git -C /opt/ai-toolkit rev-parse --short HEAD)" || echo "CLONE_AITK_FAIL"

echo "=== PHASE2: isolated aitoolkit conda env ==="
conda create -y -n aitoolkit -c conda-forge --override-channels python=3.11 pip >/dev/null 2>&1 && echo "AITK_ENV_CREATED" || echo "AITK_ENV_FAIL"
printf 'torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\n' > /root/tc.txt
echo "--- torch cu128 into aitoolkit env ---"
conda run --no-capture-output -n aitoolkit python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && echo "AITK_TORCH_OK" || echo "AITK_TORCH_FAIL"
echo "--- ai-toolkit requirements (constrained to hold torch) ---"
conda run --no-capture-output -n aitoolkit python -m pip install -c /root/tc.txt -r /opt/ai-toolkit/requirements.txt && echo "AITK_REQS_OK" || echo "AITK_REQS_FAIL"
echo "--- aitoolkit import smoke ---"
conda run --no-capture-output -n aitoolkit python -c "import torch,diffusers,transformers,safetensors;print('AITK_IMPORT_OK torch',torch.__version__,'diffusers',diffusers.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())" || echo "AITK_IMPORT_FAIL"
echo "--- frozen aitoolkit pin set (for the record) ---"
conda run --no-capture-output -n aitoolkit python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|torchcodec|torchao|diffusers|transformers|accelerate|peft|safetensors|av|bitsandbytes|optimum|optimum-quanto|huggingface-hub|numpy)==' || true

echo "=== PHASE3: stage Wan base WITHOUT HF token (hard gate) ==="
conda run --no-capture-output -n aitoolkit env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_TOKEN= HUGGING_FACE_HUB_TOKEN= python - <<'PYWAN'
import sys
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16")
    print("WAN_BASE_STAGED_NOTOKEN_OK", p)
except Exception as e:
    print("WAN_BASE_NOTOKEN_FAIL", type(e).__name__, str(e)[:300])
    sys.exit(3)
PYWAN
WAN_RC=$?
du -sh /opt/models/hf-cache/hub/models--ai-toolkit--Wan2.2-T2V-A14B-Diffusers-bf16 2>/dev/null || true
if [ $WAN_RC -ne 0 ]; then echo "D1_HALT: Wan base needs an HF token -> HARD GATE, stopping for lead"; echo "=================== D1 END (halt) ==================="; sleep infinity; fi

echo "=== PHASE4: build bundle + run REAL merged train_lora+wan via run_job ==="
export PYTHONPATH=/root/vjb/src
export VIVIJURE_AITOOLKIT_DIR=/opt/ai-toolkit
export VIVIJURE_AITOOLKIT_PYTHON=/opt/conda/envs/aitoolkit/bin/python
cat > /root/driver.py <<'DRIVEREOF'
"""cf#29 D1: exercise the REAL merged train_lora+wan path end to end through the worker's run_job,
then verify both Wan experts landed in R2 with the right keys/shape. Runs in the vivijure env with
PYTHONPATH pointed at the cf#29 branch."""
import io, json, sys, struct, tarfile, time
from pathlib import Path

PROJECT = "cf29-d1"
SLOT = "A"
TRIGGER = "cf29d1hero"
BUNDLE_KEY = "bundles/cf29-d1.tar.gz"

def log(*a): print("[driver]", *a, flush=True)

# --- 12 distinct stills so training is not fully degenerate (plumbing proof, not a bind) ---
from PIL import Image, ImageDraw
def make_stills(dst: Path, n=12):
    dst.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = Image.new("RGB", (768, 768))
        px = img.load()
        h = (i * 21) % 256
        for y in range(768):
            for x in range(0, 768, 8):
                px[x, y] = ((h + x // 3) % 256, (y // 3) % 256, (h + y // 5) % 256)
        d = ImageDraw.Draw(img)
        d.ellipse([180 + i*8, 180, 560 - i*4, 560], outline=(255, 255, 255), width=6)
        d.rectangle([120, 120, 240, 240], fill=((i*40) % 256, 80, (i*17) % 256))
        d.text((60, 60), f"{TRIGGER} pose {i:02d}", fill=(255, 255, 0))
        p = dst / f"ref_{i:02d}.png"
        img.save(p); paths.append(p)
    return paths

def build_bundle(root: Path) -> Path:
    refs = root / "characters" / "refs" / SLOT
    make_stills(refs, 12)
    sb = {"title": "cf29 d1", "use_characters": [SLOT],
          "scenes": [{"id": "s1", "prompt": f"{TRIGGER} standing", "character_slots": [SLOT]}]}
    (root / "storyboard.yaml").write_text(json.dumps(sb))  # yaml loader reads json fine
    (root / "characters").mkdir(exist_ok=True)
    (root / "characters" / "registry.json").write_text(json.dumps(
        {"characters": {SLOT: {"name": TRIGGER, "prompt": "a stylized figure"}}}))
    tar_path = root.parent / "cf29-d1.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(root / "storyboard.yaml", arcname="storyboard.yaml")
        tf.add(root / "characters" / "registry.json", arcname="characters/registry.json")
        for p in sorted(refs.glob("*.png")):
            tf.add(p, arcname=f"characters/refs/{SLOT}/{p.name}")
    return tar_path

def parse_safetensors_header(b: bytes) -> dict:
    n = struct.unpack("<Q", b[:8])[0]
    return json.loads(b[8:8+n].decode("utf-8"))

def main():
    from vivijure_backend.contract import RenderRequest
    from vivijure_backend.pipeline import GpuPipeline
    from vivijure_backend.harness.handler import run_job
    from vivijure_backend.harness.r2 import R2, R2Config
    from vivijure_backend.harness import keys
    import vivijure_backend.wan_lora_train as W
    log("seam VIVIJURE_AITOOLKIT_PYTHON ->", W.aitoolkit_python())
    assert W.aitoolkit_python().endswith("/aitoolkit/bin/python"), W.aitoolkit_python()

    work = Path("/root/vjwork"); work.mkdir(parents=True, exist_ok=True)
    proj_root = work / "bundle_src" / "project"
    proj_root.mkdir(parents=True, exist_ok=True)
    tar_path = build_bundle(proj_root)
    log("bundle built", tar_path, tar_path.stat().st_size, "bytes")

    store = R2(R2Config.from_env())
    store.put_file(tar_path, BUNDLE_KEY, content_type="application/gzip")
    log("bundle uploaded to R2", BUNDLE_KEY)

    job = {"action": "train_lora", "project": PROJECT, "bundle_key": BUNDLE_KEY, "model_family": "wan"}
    req = RenderRequest.from_dict(job)
    pipeline = GpuPipeline(config=req.config, pretrained_loras=req.pretrained_loras, server=None)
    log("=== RUN_JOB START (real orchestrator->pipeline->wan_lora_train->handler) ===", time.strftime("%H:%M:%S"))
    t0 = time.time()
    res = run_job(job, pipeline=pipeline, store=store, workdir=work / "run", job_id="cf29-d1")
    dt = time.time() - t0
    log("=== RUN_JOB DONE in %.1f min ===" % (dt / 60))
    log("result:", json.dumps(res)[:800])

    # --- verify both experts in R2 with the right keys + shape ---
    kh = keys.wan_lora_key(PROJECT, SLOT, "high")
    kl = keys.wan_lora_key(PROJECT, SLOT, "low")
    report = {"project": PROJECT, "slot": SLOT, "run_minutes": round(dt/60, 1),
              "key_high": kh, "key_low": kl}
    report["exists_high"] = store.exists(kh)
    report["exists_low"] = store.exists(kl)
    for tag, k in (("high", kh), ("low", kl)):
        try:
            b = store.get_bytes(k)
            hdr = parse_safetensors_header(b)
            meta = hdr.get("__metadata__", {})
            tensors = {kk: vv for kk, vv in hdr.items() if kk != "__metadata__"}
            dtypes = sorted({vv.get("dtype") for vv in tensors.values()})
            sample = list(tensors.keys())[:4]
            report[f"{tag}_bytes"] = len(b)
            report[f"{tag}_tensor_count"] = len(tensors)
            report[f"{tag}_dtypes"] = dtypes
            report[f"{tag}_sample_keys"] = sample
            report[f"{tag}_has_diffusion_model_keys"] = any(s.startswith("diffusion_model") for s in tensors)
            report[f"{tag}_metadata"] = meta
        except Exception as e:
            report[f"{tag}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print("D1_REPORT_JSON " + json.dumps(report), flush=True)
    ok = (report["exists_high"] and report["exists_low"]
          and report.get("high_tensor_count", 0) > 0 and report.get("low_tensor_count", 0) > 0)
    print("D1_VERDICT " + ("PASS" if ok else "FAIL"), flush=True)
    # also stash the report in R2 so it survives independent of the log
    store.put_bytes(json.dumps(report, indent=2).encode(), "bundles/cf29-d1-report.json",
                    content_type="application/json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("D1_VERDICT FAIL (exception)", flush=True)

DRIVEREOF
conda run --no-capture-output -n vivijure python /root/driver.py
echo "DRIVER_RC=$?"
echo "=================== D1 END ==================="
sleep infinity
