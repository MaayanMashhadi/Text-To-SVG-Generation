import os, io, csv, json, re, math, random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import torch
from diffusers import FluxPipeline
from tracer import *
import pd

OUTDIR = Path("out_flux_svg"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Flux
FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"   # or "black-forest-labs/FLUX.1-dev"
WIDTH, HEIGHT = 768, 768
NUM_INFERENCE_STEPS = 4
GUIDANCE = 0.0
TORCH_DTYPE = torch.float16

# # Generation
# PROMPTS = [
#     "flat vector illustration of a playful cat with bold clean outlines and 6 solid colors, high contrast, minimal gradients, poster style",
#     #"minimal bee icon with clean outlines, flat fills, high contrast, scalable vector-style graphic",
#     #"simple bicycle illustration, crisp contours, flat color palette, high contrast, posterized vector aesthetic",
# ]

excel_path = "prompts.xlsx"  # Path to your Excel file
df = pd.read_excel(excel_path)
PROMPTS = df.iloc[:, 1].dropna().tolist()  # Take second column, drop empty cells

print(f"Loaded {len(PROMPTS)} prompts from Excel.")

NEGATIVE = "photorealistic, noise, texture, watercolor, gradients, film grain, shading"

TOTAL_IMAGES = 20       
SEED_BASE = 2025         

K_COLORS = 8
TRANSPARENT_BG = True   
MIN_AREA = 1             

# --- contour clean-up knobs (tune if grainy) ---
SPECK_MIN_AREA_PX  = 1500    
SPECK_MIN_AREA_REL = 0.001   
DROP_BORDER_SPECKS = True
OPEN_KERNEL = 5             
OPEN_ITERS  = 2              
CLOSE_ITERS = 1            
EPS_REL     = 0.0025        

# SVG debug (normally keep False)
DEBUG_STROKE    = False
DEBUG_COLORIZE  = False
ROUND_COORDS_TO_INT = True  

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _enable_memory_savers(pipe):
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

def init_flux() -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=TORCH_DTYPE)
    _enable_memory_savers(pipe)
    return pipe

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

def generate_flux_image(pipe: FluxPipeline, prompt: str, seed: int) -> Image.Image:
    set_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE,
        width=WIDTH,
        height=HEIGHT
    ).images[0]

def quantize_colors_paletted(im: Image.Image, k: int):
    """Return (P-mode image, full palette list (len 256*3), idx array HxW, usage dict)."""
    qP = im.convert("RGB").quantize(colors=k, method=Image.MEDIANCUT, dither=Image.NONE)
    pal_full = qP.getpalette()  
    idx = np.array(qP, dtype=np.uint8)
    uniq, counts = np.unique(idx, return_counts=True)
    usage = dict(zip(uniq.tolist(), counts.tolist()))
    return qP, pal_full, idx, usage

def rgb_for_index(pal_full: list[int], ci: int) -> Tuple[int,int,int]:
    base = ci * 3
    if base + 2 >= len(pal_full): return (0, 0, 0)
    return tuple(pal_full[base:base+3])

def guess_background_index(idx_arr: np.ndarray) -> int:
    top = idx_arr[0, :]; bottom = idx_arr[-1, :]
    left = idx_arr[:, 0]; right = idx_arr[:, -1]
    border = np.concatenate([top, bottom, left, right])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _preprocess_mask(mask01: np.ndarray) -> np.ndarray:
    """Morphologically smooth the 0/1 mask and remove tiny components."""
    m = (mask01.astype(np.uint8) * 255)

    if OPEN_KERNEL and (OPEN_ITERS or CLOSE_ITERS):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
        if OPEN_ITERS:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
        if CLOSE_ITERS:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)

    bw = (m > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1: return bw

    on_pixels = int(bw.sum())
    area_abs = max(1, int(SPECK_MIN_AREA_PX))
    area_rel = max(1, int(on_pixels * SPECK_MIN_AREA_REL))

    H, W = bw.shape[:2]
    keep = np.zeros(n, dtype=bool); keep[0] = False
    for i in range(1, n):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                           stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], \
                           stats[i, cv2.CC_STAT_AREA]
        drop = (area < area_abs) or (area < area_rel)
        if DROP_BORDER_SPECKS and not drop:
            if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
                drop = True
        keep[i] = not drop

    out = (labels > 0) & keep[labels]
    return out.astype(np.uint8)

def _round_if_needed(x: float) -> str:
    if ROUND_COORDS_TO_INT:
        return str(int(round(x)))
    return str(x)

def trace_contours_paths(mask01: np.ndarray, eps_rel: float = EPS_REL) -> List[str]:
    """Trace binary mask (0/1) into SVG subpaths using OpenCV contours (CCOMP + RDP)."""
    m = _preprocess_mask(mask01)
    if m.max() == 0: return []
    m255 = (m * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(m255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0: return []

    d_list: List[str] = []
    for cnt in contours:
        if len(cnt) < 3: continue
        perim = max(1.0, cv2.arcLength(cnt, True))
        eps = perim * float(eps_rel)
        cnt_simpl = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        pts = cnt_simpl.reshape(-1, 2).astype(float)

        d = [f"M {_round_if_needed(pts[0,0])} {_round_if_needed(pts[0,1])}"]
        d.extend(f"L {_round_if_needed(x)} {_round_if_needed(y)}" for x, y in pts[1:])
        d.append("Z")
        d_list.append(" ".join(d))
    return d_list

# --- SVG compose & tokens ---
def compose_colored_svg(width, height, color_paths: List[Tuple[Tuple[int,int,int], List[str]]]) -> str:
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'shape-rendering="geometricPrecision">\n'
    )
    body = []
    for i, (rgb, d_list) in enumerate(color_paths):
        if not d_list: continue
        if DEBUG_COLORIZE:
            hue = (i * 137) % 360
            fill = f"hsl({hue}, 70%, 55%)"
        else:
            fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        d_all = " ".join(d_list)  # combine subpaths → evenodd handles holes
        if DEBUG_STROKE:
            body.append(
                f'<path d="{d_all}" fill="{fill}" fill-rule="evenodd" clip-rule="evenodd" '
                f'stroke="black" stroke-width="1" vector-effect="non-scaling-stroke" />'
            )
        else:
            body.append(
                f'<path d="{d_all}" fill="{fill}" fill-rule="evenodd" clip-rule="evenodd" />'
            )
    return header + "\n".join(body) + "\n</svg>\n"

_PATH_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)")

def tokenize_path_d(d: str, quantize: int | None = 0):
    toks = []
    for cmd, num in _PATH_TOKEN_RE.findall(d):
        if cmd:
            toks.append(cmd.upper())
        else:
            if quantize is None:
                toks.append(num)
            else:
                val = round(float(num), quantize)
                toks.append(str(int(val)) if quantize == 0 else str(val))
    return toks

def tokenize_svg(svg_text: str, number_quantize_decimals: int | None = 0):
    tag_tokens, attr_tokens, path_tokens = [], [], []
    for m in re.finditer(r"<(/?)(svg|g|path)\b([^>]*)>", svg_text, flags=re.I):
        closing, tag, attrs = m.groups()
        tag_tokens.append(f"</{tag.lower()}>" if closing else f"<{tag.lower()}>")
        for name, value in re.findall(r'([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)"', attrs):
            nl = name.lower(); attr_tokens.append(nl)
            if nl == "d":
                path_tokens += tokenize_path_d(value, number_quantize_decimals)
            elif nl in ("fill", "stroke"):
                attr_tokens.append("rgb()" if value.startswith("rgb(") else ("#hex" if value.startswith("#") else value.lower()))
            else:
                attr_tokens.append("<val>")
    all_tokens = tag_tokens + attr_tokens + path_tokens
    return {
        "counts": {
            "tags": len(tag_tokens),
            "attrs_plus_vals": len(attr_tokens),
            "path_cmds_nums": len(path_tokens),
        },
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
    }


def per_prompt_counts(total: int, num_prompts: int) -> List[int]:
    base = total // num_prompts
    rem = total % num_prompts
    counts = [base] * num_prompts
    for i in range(rem):
        counts[i] += 1
    return counts

def run_batch():
    print(f"Generating {TOTAL_IMAGES} images across {len(PROMPTS)} prompts...")
    pipe = init_flux()

    counts = per_prompt_counts(TOTAL_IMAGES, len(PROMPTS))
    manifest_rows = []
    manifest_path = OUTDIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "global_idx","prompt_idx","prompt","seed",
            "png_path","quantized_png","svg_path",
            "total_tokens","tags","attrs_plus_vals","path_cmds_nums"
        ])

        global_idx = 0
        for p_idx, (prompt, n_images) in enumerate(zip(PROMPTS, counts)):
            sub = OUTDIR / f"prompt_{p_idx:02d}"
            (sub / "png").mkdir(parents=True, exist_ok=True)
            (sub / "svg").mkdir(parents=True, exist_ok=True)
            (sub / "debug").mkdir(parents=True, exist_ok=True)

            for i in range(n_images):
                seed = SEED_BASE + global_idx
                print(f"\n[{global_idx+1}/{TOTAL_IMAGES}] prompt#{p_idx} seed={seed}")
                img = generate_flux_image(pipe, prompt, seed)
                png_path = sub / "png" / f"img_{global_idx:05d}.png"
                img.save(png_path)

                arr = np.array(img.convert("RGB"))
                smooth = cv2.bilateralFilter(arr, d=7, sigmaColor=75, sigmaSpace=75)
                clean = Image.fromarray(smooth)

                qP, pal_full, idx_arr, usage = quantize_colors_paletted(clean, K_COLORS)
                if len(usage) > 0:
                    bg = guess_background_index(idx_arr)
                    bg_area = usage.get(bg, 0)
                    bg_frac = bg_area / (WIDTH*HEIGHT)
                    if bg_frac < 0.6 and K_COLORS > 6:
                        qP, pal_full, idx_arr, usage = quantize_colors_paletted(clean, K_COLORS-2)

                qpng = sub / "png" / f"img_{global_idx:05d}_q{k_colors_str(K_COLORS)}.png"
                qP.convert("RGB").save(qpng)

                bg_idx = guess_background_index(idx_arr)
                items = sorted(usage.items(), key=lambda kv: kv[1], reverse=True)

                color_paths: List[Tuple[Tuple[int,int,int], List[str]]] = []
                bg_rgb = np.array(rgb_for_index(pal_full, bg_idx))
                def near_bg(ci, thr=18.0):
                    rgb = np.array(rgb_for_index(pal_full, ci), float)
                    return np.linalg.norm(rgb - bg_rgb) < thr
                for ci, area in items:
                    if area < MIN_AREA: continue
                    if TRANSPARENT_BG and (ci == bg_idx or near_bg(ci, thr=18.0)):
                        continue
                    mask01 = (idx_arr == ci).astype(np.uint8)

                    d_list = trace_contours_paths_autotuned(mask01, WIDTH, HEIGHT)
                    if not d_list: continue
                    rgb = rgb_for_index(pal_full, ci)
                    color_paths.append((rgb, d_list))

                svg_text = compose_colored_svg(WIDTH, HEIGHT, color_paths)
                svg_path = sub / "svg" / f"img_{global_idx:05d}.svg"
                svg_path.write_text(svg_text, encoding="utf-8")

                stats = tokenize_svg(svg_text, number_quantize_decimals=0)
                writer.writerow([
                    global_idx, p_idx, prompt, seed,
                    str(png_path), str(qpng), str(svg_path),
                    stats["total_tokens"],
                    stats["counts"]["tags"],
                    stats["counts"]["attrs_plus_vals"],
                    stats["counts"]["path_cmds_nums"],
                ])
                mf.flush()
                global_idx += 1

    print(f"\nDone. Manifest: {manifest_path}")

def k_colors_str(k: int) -> str:
    return str(k)


if __name__ == "__main__":
    run_batch()
