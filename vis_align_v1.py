#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
import numpy as np

import os



# 可选：HF tokenizer
def maybe_load_tokenizer(tok_name_or_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_name_or_path, use_fast=True)
        return tok
    except Exception as e:
        print(f"[warn] cannot load tokenizer: {e}")
        return None

# 你的特殊 token 列表（可按需补充/修改）
SPECIAL_TOKENS = {
    151643: "<|endoftext|>",
    151644: "<|im_start|>",
    151645: "<|im_end|>",
    151646: "<|object_ref_start|>",
    151647: "<|object_ref_end|>",
    151648: "<|box_start|>",
    151649: "<|box_end|>",
    151650: "<|quad_start|>",
    151651: "<|quad_end|>",
    151652: "<|vision_start|>",
    151653: "<|vision_end|>",
    151654: "<|vision_pad|>",
    151655: "<|image_pad|>",
    151656: "<|video_pad|>",
    151657: "<tool_call>",
    151658: "</tool_call>",
    151659: "<|fim_prefix|>",
    151660: "<|fim_middle|>",
    151661: "<|fim_suffix|>",
    151662: "<|fim_pad|>",
    151663: "<|repo_name|>",
    151664: "<|file_sep|>",
}

HTML_HEAD = """<!doctype html>
<html lang="zh">
<meta charset="utf-8">
<title>Tensor Mask Viewer</title>
<style>
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }
h1 { margin-bottom: 8px; }
.section { margin: 18px 0 28px; }
.rowtitle { font-weight: 700; margin: 8px 0 4px; }
.tokenline { display: flex; flex-wrap: wrap; gap: 6px; line-height: 1.8; }
.token {
  padding: 2px 6px;
  border-radius: 6px;
  border: 1px solid rgba(0,0,0,0.08);
  background: #fff;
  user-select: text;
  white-space: pre;
}
.token.special {
  background: #fff7cc;      /* 浅黄底 */
  border-color: #7c3aed;    /* 紫色边框 */
  box-shadow: 0 0 0 1px rgba(124,58,237,0.15) inset;
}
.legend { display: flex; gap: 18px; margin: 8px 0 16px;}
.legend .chip { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle;}
.small { color:#666; font-size: 12px;}
hr { border: none; border-top: 1px solid #eee; margin: 18px 0; }
.header { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
.badge { font-size:12px; padding:2px 6px; border-radius:12px; background:#f1f1f1;}
.idx { color:#888; }
</style>
"""

HTML_TAIL = "</html>"

def color_for_mask(mask_value, mode):
    """
    mode: 'loss' or 'attn'
    return css color string
    """
    if mode == 'loss':
        return "#d00" if mask_value == 1 else "#000"
    elif mode == 'attn':
        return "#0a0" if mask_value == 1 else "#000"
    return "#000"

def token_label(token_id, tokenizer=None):
    # 优先：特殊 token 可读名；其次：HF tokenizer；最后：数字 ID
    if token_id in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[token_id]
    if tokenizer is not None:
        try:
            return tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:
            pass
    return str(int(token_id))

def make_line_html(ids_row, mask_row, mode, tokenizer=None):
    spans = []
    for j, tid in enumerate(ids_row):
        mv = int(mask_row[j]) if np.ndim(mask_row) > 0 else int(mask_row)
        color = color_for_mask(mv, mode)
        label = token_label(int(tid), tokenizer)
        special_cls = " special" if int(tid) in SPECIAL_TOKENS else ""
        spans.append(f'<span class="token{special_cls}" style="color:{color}">{label}</span>')
    return '<div class="tokenline">' + "".join(spans) + "</div>"

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr[None, :]
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="./debug/input_ids.npy")
    ap.add_argument("--loss", default="./debug/loss_mask.npy")
    ap.add_argument("--attn", default="./debug/attention_mask.npy")
    ap.add_argument("--out", default="./debug/inspect.html")
    ap.add_argument("--tokenizer", default=None,
                    help="可选：HF tokenizer 名称或本地路径，用于把ID转成token字符串")
    args = ap.parse_args()

    ids = np.load(args.ids, allow_pickle=False)
    loss = np.load(args.loss, allow_pickle=False)
    attn = np.load(args.attn, allow_pickle=False)

    ids = ensure_2d(ids)
    loss = ensure_2d(loss)
    attn = ensure_2d(attn)

    B, T = ids.shape
    print(f"[info] loaded ids={ids.shape}, loss={loss.shape}, attn={attn.shape}")

    tok = maybe_load_tokenizer(args.tokenizer) if args.tokenizer else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(HTML_HEAD)
        f.write("<body>")
        f.write("<h1>Tensor Mask Viewer</h1>")
        f.write('<div class="header">')
        f.write(f'<div class="badge">Batch size: {B}</div>')
        f.write(f'<div class="badge">Seq len: {T}</div>')
        if args.tokenizer:
            f.write(f'<div class="badge">Tokenizer: {args.tokenizer}</div>')
        f.write("</div>")

        f.write('<div class="legend">')
        f.write('<div><span class="chip" style="background:#d00"></span>Loss=1 (红)</div>')
        f.write('<div><span class="chip" style="background:#000"></span>Loss=0 (黑)</div>')
        f.write('<div><span class="chip" style="background:#0a0"></span>Attn=1 (绿)</div>')
        f.write('<div><span class="chip" style="background:#fff;border:1px solid #7c3aed"></span>特殊Token（黄底紫框）</div>')
        f.write('</div>')

        for i in range(B):
            f.write('<div class="section">')
            f.write(f'<div class="rowtitle">序列 <span class="idx">#{i}</span> : IDs</div>')
            # 中性一行（只做特殊 token 的高亮，文本颜色均为黑）
            f.write(make_line_html(ids[i], np.zeros_like(ids[i]), mode="neutral", tokenizer=tok).replace('style="color:#000"', ''))

            f.write('<div class="rowtitle">Loss Mask 视图（红/黑）</div>')
            f.write(make_line_html(ids[i], loss[i], mode="loss", tokenizer=tok))

            f.write('<div class="rowtitle">Attention Mask 视图（绿/黑）</div>')
            f.write(make_line_html(ids[i], attn[i], mode="attn", tokenizer=tok))
            f.write('</div><hr>')
        f.write("</body>")
        f.write(HTML_TAIL)

    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
