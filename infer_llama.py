import argparse, sys, json, os, re, random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# ---- prompt (match training) ----
PROMPT_PREFIX = "### Instruction:\nCreate a valid minimal SVG for: "
PROMPT_SUFFIX = "\n### Response:\n"

def build_prompt(desc: str) -> str:
    return f"{PROMPT_PREFIX}{desc.strip()}{PROMPT_SUFFIX}"

# ---- stop when we see "</svg>" ----
class StopOnTokenSeq(StoppingCriteria):
    def __init__(self, seq_ids: torch.LongTensor):
        super().__init__()
        self.seq = seq_ids  # 1D tensor on the same device as input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # only checks the first sequence (greedy / single-sample generation)
        L = self.seq.numel()
        if input_ids.shape[1] < L:
            return False
        return torch.equal(input_ids[0, -L:], self.seq)

def extract_svg(text: str) -> str:
    """
    Return the first <svg ...>...</svg> block if present; otherwise the whole string.
    """
    m_open = re.search(r"<svg\b[^>]*>", text, flags=re.I | re.S)
    m_close = re.search(r"</svg\s*>", text, flags=re.I)
    if m_open and m_close:
        start = m_open.start()
        end = m_close.end()
        return text[start:end].strip()
    return text.strip()

def load_model_and_tokenizer(model_path: str, adapter_path: str | None, bf16: bool, fp16: bool):
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    model_dir = Path(model_path)
    adapter_dir = Path(adapter_path) if adapter_path else None
    looks_like_adapter = (model_dir / "adapter_config.json").exists() or (model_dir / "adapter_model.bin").exists()

    try:
        if adapter_dir or looks_like_adapter:
            # Base + LoRA
            if looks_like_adapter and adapter_dir is None:
                adapter_dir = model_dir
                raise FileNotFoundError("Adapter was passed as model; you must also pass --base_model.")


        if adapter_dir:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
            tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
        else:
            # Fully saved model (no separate adapter)
            tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)

    except Exception as e:
        print(f"[ERROR] Failed to load model/tokenizer: {e}", file=sys.stderr)
        raise

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tok, device

def generate_svg(model, tok, device, desc: str, max_new_tokens: int, temperature: float, top_p: float,
                 repetition_penalty: float, stop_on_svg: bool, eos_token_id: int | None):
    prompt = build_prompt(desc)
    ipt = tok(prompt, return_tensors="pt").to(device)

    do_sample = temperature is not None and temperature > 0.0
    stopping = None
    if stop_on_svg:
        # Build token pattern for "</svg>"
        svg_end_ids = tok("</svg>", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        stopping = StoppingCriteriaList([StopOnTokenSeq(svg_end_ids)])

    with torch.no_grad():
        out = model.generate(
            **ipt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=(temperature if do_sample else None),
            top_p=(top_p if do_sample else None),
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id if eos_token_id is not None else tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=stopping,
        )

    # Decode and trim to the last portion (model returns prompt+response)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Keep only the part after PROMPT_SUFFIX to avoid echo
    tail = text.split(PROMPT_SUFFIX, 1)[-1]
    return extract_svg(tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or HF id of the base or merged model.")
    ap.add_argument("--adapter", default=None, help="(Optional) LoRA adapter dir. If provided, --model is the base.")
    ap.add_argument("--prompt", default=None, help="Single prompt (description).")
    ap.add_argument("--infile", default=None, help="Text file with one prompt per line.")
    ap.add_argument("--outfile", default=None, help="Where to write output. If --infile is set, writes JSONL.")
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--stop_on_svg", action="store_true", help="Stop when '</svg>' is generated.")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok, device = load_model_and_tokenizer(args.model, args.adapter, args.bf16, args.fp16)

    def _one(desc: str) -> str:
        return generate_svg(
            model, tok, device, desc=desc,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_on_svg=args.stop_on_svg,
            eos_token_id=tok.eos_token_id,
        )

    if args.prompt and args.infile:
        print("[WARN] Both --prompt and --infile provided; using --prompt only.", file=sys.stderr)

    if args.prompt:
        svg = _one(args.prompt)
        if args.outfile:
            Path(args.outfile).write_text(svg, encoding="utf-8")
        else:
            print(svg)
        return

    if args.infile:
        outpath = Path(args.outfile) if args.outfile else None
        if outpath:
            f = outpath.open("w", encoding="utf-8")
        else:
            f = sys.stdout

        with open(args.infile, "r", encoding="utf-8") as fin:
            for line in fin:
                desc = line.strip()
                if not desc:
                    continue
                svg = _one(desc)
                rec = {"prompt": desc, "svg": svg, "ok": svg.lower().startswith("<svg")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if outpath:
            f.close()
        return

    print("Provide either --prompt or --infile.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()