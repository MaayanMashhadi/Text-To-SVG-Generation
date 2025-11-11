import os, re, json, random, argparse, signal, sys, time
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback

os.environ.setdefault("HF_HOME", "")
os.environ.setdefault("HF_HUB_CACHE", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model

from svg_dsl import svg_to_dsl, dsl_to_svg, make_chunks, stitch


RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process() -> bool:
    return RANK == 0

def log0(*a, **k):
    if is_main_process():
        print(*a, **k)

def barrier_file(flag_path: str):
    """Simple file-based barrier usable before torch.distributed init."""
    if WORLD_SIZE == 1 or not flag_path:
        return
    if is_main_process():
        return
    while not os.path.exists(flag_path):
        time.sleep(1.0)


_NUM_RE = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+)")
def svg_minify(svg: str, decimals: int = 1) -> str:
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S | re.I)
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s{2,}", " ", svg).strip()
    def _round(m):
        s = m.group(1)
        try:
            v = float(s)
            if v.is_integer(): return str(int(v))
            return f"{v:.{decimals}f}".rstrip("0").rstrip(".")
        except Exception:
            return s
    svg = _NUM_RE.sub(_round, svg)
    return svg.replace("'", '"')

def looks_like_svg(s: str) -> bool:
    return "<svg" in s[:512].lower() and s.strip().endswith(">")


def _iter_manifest_paths(paths: List[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p
        elif p.is_dir():
            for m in sorted(p.glob("manifest_rank*.csv")):
                yield m

def _caption_from_row(row: pd.Series) -> Optional[str]:
    for key in ("caption_raw", "simplified", "caption", "prompt"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _svg_path_from_row(man_path: Path, row: pd.Series) -> Optional[Path]:
    if "svg_path" in row and isinstance(row["svg_path"], str):
        p = Path(row["svg_path"])
        if not p.is_absolute(): p = (man_path.parent / p).resolve()
        return p if p.exists() else None
    if "png_path" in row and isinstance(row["png_path"], str):
        png = Path(row["png_path"])
        if not png.is_absolute(): png = (man_path.parent / png).resolve()
        for sub in ("svg", "svg_rejects"):
            cand = png.parent.parent / sub / f"{png.stem}.svg"
            if cand.exists(): return cand
    return None

def build_imgid_to_caption(simp_dirs: List[Path]) -> Dict[int, str]:
    id2cap: Dict[int, str] = {}
    for d in simp_dirs:
        files = [d] if d.is_file() else sorted(d.glob("simplified_rank*.json"))
        for jl in files:
            try:
                with jl.open("r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        cap = (rec.get("simplified") or rec.get("caption") or "").strip()
                        iid = rec.get("image_id")
                        if cap and isinstance(iid, int):
                            id2cap.setdefault(iid, cap)
            except Exception:
                continue
    return id2cap

_IMGID_RE = re.compile(r"img(?P<id>\d+)", re.I)
def _image_id_from_stem(stem: str) -> Optional[int]:
    m = _IMGID_RE.search(stem)
    return int(m.group("id")) if m else None

def load_pairs_from_manifests(
    manifest_inputs: List[Path],
    simplified_dirs: List[Path],
    limit: Optional[int]=None,
    progress_every: int = 5000,
    chunksize: int = 200_000,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    id2cap = build_imgid_to_caption(simplified_dirs) if simplified_dirs else {}
    total_rows = 0

    read_csv_kwargs = dict(low_memory=False)
    try:
        read_csv_kwargs["engine"] = "pyarrow"  # if available
    except Exception:
        pass

    for man in _iter_manifest_paths(manifest_inputs):
        try:
            for chunk in pd.read_csv(man, chunksize=chunksize, **read_csv_kwargs):
                for row in chunk.to_dict("records"):
                    total_rows += 1
                    if is_main_process() and total_rows % progress_every == 0:
                        log0(f"[pairs] {man.name}: rows={total_rows:,} | pairs={len(pairs):,}")

                    svg_path = _svg_path_from_row(man, row)
                    if not svg_path: 
                        continue

                    iid = _image_id_from_stem(Path(svg_path).stem)
                    caption = id2cap.get(iid) if iid in id2cap else None
                    if not caption:
                        caption = _caption_from_row(row)
                    if not caption:
                        continue

                    try:
                        txt = Path(svg_path).read_text(encoding="utf-8")
                        if not looks_like_svg(txt):
                            continue
                        txt = svg_minify(txt, decimals=1)
                        pairs.append({"input": caption, "svg": txt, "svg_path": str(svg_path)})
                        if limit and len(pairs) >= limit:
                            log0(f"[pairs] finished (limit hit). total rows={total_rows:,} | pairs={len(pairs):,}")
                            return pairs
                    except Exception:
                        continue
        except Exception as e:
            log0(f"[pairs] failed reading {man}: {e}")
            continue

    log0(f"[pairs] finished. total rows={total_rows:,} | pairs={len(pairs):,}")
    return pairs


CTX_TOK     = "<|ctx|>"
OBJS_TOK    = "<|objs|>"
END_SVG_TOK = "<|end_svg|>"

def pack_io(ctx_json: str, tgt_json: str) -> Tuple[str, str]:
    prompt  = f"{CTX_TOK}\n{ctx_json}\n{OBJS_TOK}\n"
    target  = f"{tgt_json}\n{END_SVG_TOK}"
    return prompt, target


class DSLDataset(Dataset):
    """
    Packed samples: multiple (CTX,new_objects) segments from the SAME SVG
    concatenated up to `max_len`. Guarantees:
      - Every SVG contributes ≥1 sample (fallback chunk),
      - Oversize objects are split (path/polyline/polygon),
      - Prompts are masked (-100); only targets supervised.
    """
    def __init__(self, items, tokenizer, max_len=4096,
                 k_tail=16, m_new=32, seed=0, pack=True):
        self.tok = tokenizer
        self.max_len = max_len
        self.pack = pack
        self.samples: List[Tuple[List[int], List[int]]] = []  # (input_ids, label_ids)
        rng = random.Random(seed)

        self.stats = {
            "svgs": len(items),
            "packed_samples": 0,
            "segments_emitted": 0,
            "svg_to_dsl_errors": 0,
            "svgs_zero_chunks_before_fix": 0,
            "chunks_kept": 0,
            "chunks_shrunk": 0,
            "oversize_single_object_splits": 0,
            "max_tokens_observed": 0,
        }

        # tiny tokenization cache to avoid repeated probes during split/shrink
        self._tok_cache: Dict[str, List[int]] = {}
        total = len(items)
        for idx, rec in enumerate(items):
            if is_main_process() and (idx % 1000 == 0):
                log0(f"[build] {idx}/{total} svgs processed...")
            # ... existing body ...

            try:
                dsl = svg_to_dsl(rec["svg"])
            except Exception:
                self.stats["svg_to_dsl_errors"] += 1
                continue

            chunks = list(make_chunks(dsl, k_tail=k_tail, m_new=m_new))
            if not chunks:
                self.stats["svgs_zero_chunks_before_fix"] += 1
                fallback = self._minimal_chunk_from_dsl(dsl, k_tail)
                if fallback is not None:
                    chunks = [fallback]
                else:
                    continue

            if self.pack:
                for ids, labs in self._pack_chunks(chunks):
                    self._append_example(ids, labs)
            else:
                for ctx, tgt in chunks:
                    ids, labs = self._encode_segment_as_standalone(ctx, tgt)
                    self._append_example(ids, labs)

        rng.shuffle(self.samples)
        for ids, _ in self.samples:
            assert len(ids) <= self.max_len, f"Sample exceeds max_len={self.max_len} (got {len(ids)})"
        self.lengths = [len(ids) for ids, _ in self.samples]

    def _append_example(self, ids: List[int], labs: List[int]):
        self.samples.append((torch.tensor(ids, dtype=torch.long),
                             torch.tensor(labs, dtype=torch.long)))
        self.stats["packed_samples"] += 1
        self.stats["max_tokens_observed"] = max(self.stats["max_tokens_observed"], len(ids))

    def _tok_ids(self, text: str) -> List[int]:
        hit = self._tok_cache.get(text)
        if hit is None:
            hit = self.tok(text, add_special_tokens=False)["input_ids"]
            self._tok_cache[text] = hit
        return hit

    def _end_id(self) -> int:
        return self.tok.convert_tokens_to_ids(END_SVG_TOK)

    def _pack_chunks(self, chunks: List[Tuple[dict, dict]]) -> Iterable[Tuple[List[int], List[int]]]:
        maxL = self.max_len
        cur_ids: List[int] = []
        cur_labs: List[int] = []
        end_tok = [self._end_id()]

        def flush():
            if not cur_ids:
                return None
            ids = cur_ids + end_tok
            labs = cur_labs + end_tok
            self.stats["segments_emitted"] += 1
            return ids, labs

        for (ctx, tgt) in chunks:
            p_ids, t_ids = self._encode_segment_pair(ctx, tgt, allow_split=True)
            seg_ids = p_ids + t_ids
            seg_labs = [-100] * len(p_ids) + t_ids

            if len(seg_ids) + 1 > maxL:
                room = maxL - 1 - len(p_ids)
                if room <= 0:
                    continue
                seg_ids = p_ids + t_ids[:room]
                seg_labs = [-100] * len(p_ids) + t_ids[:room]
                self.stats["chunks_shrunk"] += 1

            if cur_ids and (len(cur_ids) + len(seg_ids) + 1 > maxL):
                out = flush()
                if out:
                    yield out
                cur_ids, cur_labs = [], []

            cur_ids.extend(seg_ids)
            cur_labs.extend(seg_labs)
            self.stats["chunks_kept"] += 1

        out = flush()
        if out:
            yield out

    def _encode_segment_as_standalone(self, ctx: dict, tgt: dict) -> Tuple[List[int], List[int]]:
        p_ids, t_ids = self._encode_segment_pair(ctx, tgt, allow_split=True)
        ids = p_ids + t_ids + [self._end_id()]
        labs = [-100]*len(p_ids) + t_ids + [self._end_id()]
        return ids, labs

    def _encode_segment_pair(self, ctx: dict, tgt: dict, allow_split: bool) -> Tuple[List[int], List[int]]:
        ctx_json = json.dumps(ctx, separators=(",",":"))
        prompt = f"{CTX_TOK}\n{ctx_json}\n{OBJS_TOK}\n"
        p_ids = self._tok_ids(prompt)

        tgt_json = json.dumps(tgt, separators=(",",":"))
        t_ids = self._tok_ids(tgt_json)

        max_allowed = self.max_len - 1
        if len(p_ids) + len(t_ids) > max_allowed and allow_split:
            objs = list(tgt.get("new_objects", []))
            if objs:
                left, right, ok_m = 1, len(objs), 0
                while left <= right:
                    m = (left + right)//2
                    t_try = {"new_objects": objs[:m]}
                    t_try_ids = self._tok_ids(json.dumps(t_try, separators=(",",":")))
                    if len(p_ids) + len(t_try_ids) <= max_allowed:
                        ok_m = m; left = m + 1
                    else:
                        right = m - 1
                if ok_m == 0:
                    splitted = self._split_oversize_object(objs[0], max_tokens=max_allowed-len(p_ids))
                    if len(splitted) > 1:
                        self.stats["oversize_single_object_splits"] += 1
                        t_ids = self._tok_ids(json.dumps({"new_objects": [splitted[0]]}, separators=(",",":")))
                    else:
                        t_ids = t_ids[:max_allowed - len(p_ids)]
                        self.stats["chunks_shrunk"] += 1
                else:
                    t_ids = self._tok_ids(json.dumps({"new_objects": objs[:ok_m]}, separators=(",",":")))
                    if ok_m < len(objs):
                        self.stats["chunks_shrunk"] += 1
        return p_ids, t_ids

    def _minimal_chunk_from_dsl(self, dsl: dict, k_tail: int) -> Optional[Tuple[dict, dict]]:
        objs = None
        for key in ("objects","objs","shapes","elements"):
            if key in dsl and isinstance(dsl[key], list):
                objs = dsl[key]; break
        if not objs: return None
        n = len(objs)
        t = min(k_tail, max(0, n-1))
        ctx = {"header": dsl.get("header", {}), "objects": objs[max(0, n - t - 1): n - 1]}
        tgt = {"new_objects": [objs[-1]]}
        return ctx, tgt

    def _split_oversize_object(self, obj: dict, max_tokens: int) -> List[dict]:
        try:
            if isinstance(obj, dict) and obj.get("type") == "path" and isinstance(obj.get("d"), str):
                d = obj["d"]
                parts = re.split(r'(?=[MmLlHhVvCcSsQqTtAaZz])', d)
                parts = [p for p in parts if p.strip()]
                out, cur = [], ""
                for p in parts:
                    cand = cur + p
                    tmp = dict(obj); tmp["d"] = cand
                    tok_len = len(self._tok_ids(json.dumps({"new_objects":[tmp]}, separators=(",",":"))))
                    if tok_len <= max_tokens or not cur:
                        cur = cand
                    else:
                        tmp2 = dict(obj); tmp2["d"] = cur
                        out.append(tmp2); cur = p
                if cur:
                    tmp3 = dict(obj); tmp3["d"] = cur
                    out.append(tmp3)
                return out if len(out) > 1 else [obj]
            if isinstance(obj, dict) and obj.get("type") in ("polyline","polygon"):
                pts = obj.get("points")
                if isinstance(pts, str):
                    pts_list = pts.strip().split()
                elif isinstance(pts, list):
                    pts_list = [str(x) for x in pts]
                else:
                    return [obj]
                out, cur = [], []
                for token in pts_list:
                    cand = cur + [token]
                    tmp = dict(obj); tmp["points"] = " ".join(cand)
                    tok_len = len(self._tok_ids(json.dumps({"new_objects":[tmp]}, separators=(",",":"))))
                    if tok_len <= max_tokens or not cur:
                        cur = cand
                    else:
                        tmp2 = dict(obj); tmp2["points"] = " ".join(cur)
                        out.append(tmp2); cur = [token]
                if cur:
                    tmp3 = dict(obj); tmp3["points"] = " ".join(cur)
                    out.append(tmp3)
                return out if len(out) > 1 else [obj]
        except Exception:
            pass
        return [obj]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        ids, labs = self.samples[i]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn, "labels": labs}

class PrebuiltDataset(Dataset):
    def __init__(self, samples, stats):
        self.samples = [
            (torch.tensor(i, dtype=torch.long), torch.tensor(l, dtype=torch.long))
            for (i, l) in samples
        ]
        self.stats = stats
    def __len__(self): return len(self.samples)
    def __getitem__(self, i: int):
        ids, labs = self.samples[i]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn, "labels": labs}


def pad_batch_collator(tokenizer):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    def _collate(features):
        ids  = [f["input_ids"].tolist()      for f in features]
        attn = [f["attention_mask"].tolist() for f in features]
        labs = [f["labels"].tolist()         for f in features]
        maxlen = max(len(x) for x in ids)
        def _pad(seq, val): return seq + [val] * (maxlen - len(seq))
        ids  = [_pad(x, pad_id) for x in ids]
        attn = [_pad(x, 0)      for x in attn]
        labs = [_pad(x, -100)   for x in labs]
        for i in range(len(labs)):
            if all(v == -100 for v in labs[i]):
                j = max(0, sum(1 for v in ids[i] if v != pad_id) - 1)
                labs[i][j] = ids[i][j]
        for row in labs:
            assert any(v != -100 for v in row), "fully-masked labels!"
        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels":         torch.tensor(labs, dtype=torch.long),
        }
    return _collate



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="llama_svg_dsl_out")

    ap.add_argument("--manifests_from", type=str, nargs="+", required=True)
    ap.add_argument("--simplified_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--limit_pairs", type=int, default=None)

    # sequence budget & chunking
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--k_tail", type=int, default=16)
    ap.add_argument("--m_new",  type=int, default=32)

    # dataset cache
    ap.add_argument("--dataset_cache", type=str, default=None,
                    help="Path to .pt cache with prebuilt tokenized datasets")
    ap.add_argument("--rebuild_cache", action="store_true",
                    help="Force rebuilding the dataset cache")

    # training hparams
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA / full finetune
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    manifest_inputs = [Path(p) for p in args.manifests_from]
    simplified_dirs = [Path(p) for p in args.simplified_dirs]
    

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    extra_tokens = [CTX_TOK, OBJS_TOK, END_SVG_TOK]
    tok.add_special_tokens({"additional_special_tokens": extra_tokens})
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # model (flash-attn v2 with safe fallback to SDPA)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    #try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    # except Exception:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_dir, torch_dtype=dtype,
    #         attn_implementation="sdpa", low_cpu_mem_usage=True
    #     )
    model.resize_token_embeddings(len(tok))
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if not args.no_lora:
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )
        model = get_peft_model(model, lcfg)
        if is_main_process():
            model.print_trainable_parameters()


    def save_dataset_cache(path, ds_train, ds_val):
        obj = {
            "train": ([ (t.tolist(), l.tolist()) for (t,l) in ds_train.samples ], ds_train.stats),
            "val":   ([ (t.tolist(), l.tolist()) for (t,l) in ds_val.samples ],   ds_val.stats),
            "tok_len": len(tok),
            "specials": [CTX_TOK, OBJS_TOK, END_SVG_TOK],
        }
        torch.save(obj, path)

    def load_dataset_cache(path):
        obj = torch.load(path, map_location="cpu")
        (train_samples, train_stats) = obj["train"]
        (val_samples,   val_stats)   = obj["val"]
        ds_train = PrebuiltDataset(train_samples, train_stats)
        ds_val   = PrebuiltDataset(val_samples,   val_stats)
        return ds_train, ds_val

    cache_path = args.dataset_cache
    done_flag = (cache_path + ".done") if cache_path else None

    build_needed = True
    if cache_path and os.path.exists(cache_path) and not args.rebuild_cache:
        try:
            ds_train, ds_val = load_dataset_cache(cache_path)
            build_needed = False
            log0(f"[cache] Loaded dataset cache: {cache_path}")
        except Exception as e:
            log0(f"[cache] Failed to load cache ({e}), rebuilding...")

    if build_needed:
        if is_main_process():
            t0 = time.time()
            log0("[build] rank-0 building tokenized datasets...")
            pairs = load_pairs_from_manifests(manifest_inputs, simplified_dirs, limit=args.limit_pairs)
            if not pairs:
                raise RuntimeError("No (caption, SVG) pairs found.")

            log0(f"[data] pairs (SVG files): {len(pairs)}")
            # split pairs
            random.shuffle(pairs)
            n_val = max(100, int(0.02 * len(pairs)))
            val_items, train_items = pairs[:n_val], pairs[n_val:]
            log0(f"[data] train pairs: {len(train_items)} | val pairs: {len(val_items)}")

            ds_train = DSLDataset(train_items, tok, max_len=args.max_length,
                                  k_tail=args.k_tail, m_new=args.m_new, seed=args.seed, pack=True)
            ds_val   = DSLDataset(val_items, tok, max_len=args.max_length,
                                  k_tail=args.k_tail, m_new=args.m_new, seed=args.seed, pack=True)
            log0(f"[build] done in {time.time()-t0:.1f}s | train={len(ds_train)} val={len(ds_val)}")
            if cache_path:
                log0(f"[cache] saving to {cache_path}")
                save_dataset_cache(cache_path, ds_train, ds_val)
                with open(done_flag, "w") as f: f.write("ok")
        else:
            barrier_file(done_flag)
            if cache_path and os.path.exists(cache_path):
                ds_train, ds_val = load_dataset_cache(cache_path)
            else:
                # last-resort fallback (shouldn't happen if you pass --dataset_cache)
                ds_train = DSLDataset(train_items, tok, max_len=args.max_length,
                                      k_tail=args.k_tail, m_new=args.m_new, seed=args.seed, pack=True)
                ds_val   = DSLDataset(val_items, tok, max_len=args.max_length,
                                      k_tail=args.k_tail, m_new=args.m_new, seed=args.seed, pack=True)

    # stats
    def _report(tag, ds):
        s = ds.stats
        log0(
            f"[{tag}] svgs={s['svgs']} chunks_kept={s['chunks_kept']} "
            f"chunks_shrunk={s['chunks_shrunk']} "
            f"oversize_single_object_splits={s['oversize_single_object_splits']} "
            f"max_tokens_seen={s['max_tokens_observed']}"
        )
        log0(f"[{tag}] dataset_size={len(ds)}")
    _report("train", ds_train)
    _report("val", ds_val)

    collator = pad_batch_collator(tok)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,                
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,                     
        max_grad_norm=0.3,                     
        optim="adamw_torch",                    
        log_level="info",
        logging_steps=25,
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        disable_tqdm=False,
        save_total_limit=3,
        save_on_each_node=False,
        save_safetensors=True,
        bf16=args.bf16,                        
        fp16=args.fp16 and not args.bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        group_by_length=True,
        report_to=[],
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=collator, processing_class=tok
    )


    last_ckpt = get_last_checkpoint(args.out_dir) if os.path.isdir(args.out_dir) else None
    if last_ckpt:
        log0(f"[resume] Resuming from {last_ckpt}")

    def _graceful_terminate(signum, frame):
        try:
            if trainer.is_world_process_zero():
                tag = f"checkpoint-{trainer.state.global_step}-SIGTERM"
                path = os.path.join(args.out_dir, tag)
                print(f"[signal] SIGTERM received, saving {path} ...", flush=True)
                trainer.save_state()
                trainer.save_model(path)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_terminate)

    

    class AbortOnNaNCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            logs = kwargs.get("logs", {})
            loss = logs.get("loss", None)
            if loss is not None and (loss != loss):  # NaN check
                print("[safety] NaN loss detected, aborting.")
                control.should_training_stop = True

    trainer.add_callback(AbortOnNaNCallback())

    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

    if is_main_process() and 'val_items' in locals() and val_items:
        trainer.save_model(args.out_dir)
        tok.save_pretrained(args.out_dir)

        device = model.device
        model.eval()
        ex = random.choice(val_items)
        dsl = svg_to_dsl(ex["svg"])
        chunks = list(make_chunks(dsl, k_tail=args.k_tail, m_new=args.m_new))
        if chunks:
            ctx0, _tgt0 = chunks[0]
            ctx_json = json.dumps(ctx0, separators=(",",":"))
            prompt = f"{CTX_TOK}\n{ctx_json}\n{OBJS_TOK}\n"
            inp = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7,
                    eos_token_id=tok.convert_tokens_to_ids(END_SVG_TOK)
                )
            gen = tok.decode(out[0], skip_special_tokens=False)
            seg = gen.split(OBJS_TOK,1)[-1].split(END_SVG_TOK,1)[0].strip()
            try:
                decoder = json.JSONDecoder()
                start = seg.find("{")
                if start == -1: raise ValueError("No JSON object start found.")
                obj, _ = decoder.raw_decode(seg[start:])
                tgt = obj
                stitched = {"header": ctx0["header"], "objects": tgt.get("new_objects", [])}
                svg_out = dsl_to_svg(stitched)
                print("\n--- CHUNKED CONTINUATION SVG (trimmed) ---\n", svg_out[:1000])
            except Exception as e:
                print("Decode error (CHUNKED):", e)

if __name__ == "__main__":
    main()
