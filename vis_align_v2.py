# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import os
# import numpy as np

# # =============== 可选：HF tokenizer ===============
# def maybe_load_tokenizer(tok_name_or_path):
#     try:
#         from transformers import AutoTokenizer
#         tok = AutoTokenizer.from_pretrained(tok_name_or_path, use_fast=True)
#         return tok
#     except Exception as e:
#         print(f"[warn] cannot load tokenizer: {e}")
#         return None

# # =============== 特殊 token 定义（可按需扩展/修改） ===============
# SPECIAL_TOKENS = {
#     151643: "<|endoftext|>",
#     151644: "<|im_start|>",
#     151645: "<|im_end|>",
#     151646: "<|object_ref_start|>",
#     151647: "<|object_ref_end|>",
#     151648: "<|box_start|>",
#     151649: "<|box_end|>",
#     151650: "<|quad_start|>",
#     151651: "<|quad_end|>",
#     151652: "<|vision_start|>",
#     151653: "<|vision_end|>",
#     151654: "<|vision_pad|>",
#     151655: "<|image_pad|>",
#     151656: "<|video_pad|>",
#     151657: "<tool_call>",
#     151658: "</tool_call>",
#     151659: "<|fim_prefix|>",
#     151660: "<|fim_middle|>",
#     151661: "<|fim_suffix|>",
#     151662: "<|fim_pad|>",
#     151663: "<|repo_name|>",
#     151664: "<|file_sep|>",
# }

# # =============== HTML 模板 ===============
# HTML_HEAD = """<!doctype html>
# <html lang="zh">
# <meta charset="utf-8">
# <title>Tensor Mask Viewer</title>
# <style>
# body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }
# h1 { margin-bottom: 8px; }
# h2 { margin: 12px 0 6px; }
# .section { margin: 18px 0 28px; }
# .rowtitle { font-weight: 700; margin: 8px 0 4px; }
# .tokenline { display: flex; flex-wrap: wrap; gap: 6px; line-height: 1.8; }
# .token {
#   padding: 2px 6px;
#   border-radius: 6px;
#   border: 1px solid rgba(0,0,0,0.08);
#   background: #fff;
#   user-select: text;
#   white-space: pre;
# }
# .token.special {
#   background: #fff7cc;      /* 浅黄底 */
#   border-color: #7c3aed;    /* 紫色边框 */
#   box-shadow: 0 0 0 1px rgba(124,58,237,0.15) inset;
# }
# .small { color:#666; font-size: 12px;}
# hr { border: none; border-top: 1px solid #eee; margin: 18px 0; }
# .header { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
# .badge { font-size:12px; padding:2px 6px; border-radius:12px; background:#f1f1f1;}
# .idx { color:#888; }
# .table { border-collapse: collapse; font-size: 12px; }
# .table td, .table th { border: 1px solid #eee; padding: 6px 8px; }
# .ok { color: #0a0; }
# .bad { color: #d00; }
# .code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; border:1px solid #eee; padding:8px; border-radius:8px; }
# .legend { display: flex; gap: 18px; margin: 8px 0 16px;}
# .legend .chip { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle;}
# </style>
# """

# HTML_TAIL = "</html>"

# # =============== 工具函数 ===============
# def ensure_2d(arr):
#     if arr.ndim == 1:
#         return arr[None, :]
#     return arr

# def color_for_mask(mask_value, mode):
#     """
#     mode: 'loss' or 'attn' or 'neutral'
#     """
#     if mode == 'loss':
#         return "#d00" if mask_value == 1 else "#000"  # 红/黑
#     if mode == 'attn':
#         return "#0a0" if mask_value == 1 else "#000"  # 绿/黑
#     return "#000"

# def token_label(token_id, tokenizer=None):
#     # 优先：特殊 token 可读名；其次：HF tokenizer；最后：数字 ID
#     if token_id in SPECIAL_TOKENS:
#         return SPECIAL_TOKENS[token_id]
#     if tokenizer is not None:
#         try:
#             return tokenizer.convert_ids_to_tokens(int(token_id))
#         except Exception:
#             pass
#     return str(int(token_id))

# def make_line_html(ids_row, mask_row, mode, tokenizer=None):
#     spans = []
#     for j, tid in enumerate(ids_row):
#         mv = int(mask_row[j]) if np.ndim(mask_row) > 0 else int(mask_row)
#         color = color_for_mask(mv, mode)
#         label = token_label(int(tid), tokenizer)
#         special_cls = " special" if int(tid) in SPECIAL_TOKENS else ""
#         spans.append(f'<span class="token{special_cls}" style="color:{color}">{label}</span>')
#     return '<div class="tokenline">' + "".join(spans) + "</div>"

# def compare_ids(a, b, max_show=50):
#     """
#     a, b: list[int] 或 1D np.array
#     返回长度是否一致、匹配比例、前若干处差异 [(idx, a_i, b_i), ...]
#     """
#     la, lb = len(a), len(b)
#     L = min(la, lb)
#     matches = 0
#     diffs = []
#     for i in range(L):
#         if int(a[i]) == int(b[i]):
#             matches += 1
#         elif len(diffs) < max_show:
#             diffs.append((i, int(a[i]), int(b[i])))
#     ratio = (matches / L) if L > 0 else 0.0
#     same_len = (la == lb)
#     return same_len, ratio, diffs, la, lb

# # =============== 主流程 ===============
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ids", default="./debug/input_ids.npy")
#     ap.add_argument("--loss", default="./debug/loss_mask.npy")
#     ap.add_argument("--attn", default="./debug/attention_mask.npy")
#     ap.add_argument("--msg", default="./debug/message.txt")    # 解码后的字符串
#     ap.add_argument("--out", default="./debug/inspect.html")
#     ap.add_argument("--tokenizer", default=None,
#                     help="可选：HF tokenizer 名称或本地路径，用于把 message.encode 再对齐 input_ids")
#     ap.add_argument("--max_diffs", type=int, default=50, help="顶部差异表最多显示多少条")
#     args = ap.parse_args()

#     # 载入 numpy
#     ids = np.load(args.ids, allow_pickle=False)
#     loss = np.load(args.loss, allow_pickle=False)
#     attn = np.load(args.attn, allow_pickle=False)

#     ids = ensure_2d(ids)
#     loss = ensure_2d(loss)
#     attn = ensure_2d(attn)

#     B, T = ids.shape
#     print(f"[info] loaded ids={ids.shape}, loss={loss.shape}, attn={attn.shape}")

#     # 载入 tokenizer（可选）
#     tok = maybe_load_tokenizer(args.tokenizer) if args.tokenizer else None

#     # 读取 message，并尝试 encode
#     msg_str = None
#     enc_from_msg = None
#     if os.path.exists(args.msg):
#         with open(args.msg, "r", encoding="utf-8") as f:
#             msg_str = f.read()
#         if tok is not None:
#             try:
#                 # 关键：不要自动插入 special tokens
#                 enc_from_msg = tok.encode(msg_str, add_special_tokens=False)
#                 print(f"[check] message re-encoded length={len(enc_from_msg)}, input_ids length={T}")
#             except Exception as e:
#                 print(f"[warn] cannot encode message: {e}")
#         else:
#             print("[info] tokenizer not provided; skipping message encode check")
#     else:
#         print(f"[info] message file not found at {args.msg}")

#     os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
#     with open(args.out, "w", encoding="utf-8") as f:
#         f.write(HTML_HEAD)
#         f.write("<body>")
#         f.write("<h1>Tensor Mask Viewer</h1>")

#         # ===== 顶部：Message ↔ Input IDs 对比 =====
#         f.write('<div class="section">')
#         f.write("<h2>Message ↔ Input IDs 对比</h2>")

#         if msg_str is None:
#             f.write("<p class='small'>未找到 message.txt，跳过对比。</p>")
#         elif tok is None:
#             f.write("<p class='small'>未提供 --tokenizer，跳过对比。</p>")
#         else:
#             same_len, ratio, diffs, la, lb = compare_ids(ids[0].tolist(), enc_from_msg, max_show=args.max_diffs)
#             # f.write(f"<p>长度比较：message.encode={lb}，input_ids[0]={la} "
#             #         f"{'<span class=\"ok\">✅ 一致</span>' if same_len else '<span class=\"bad\">❌ 不一致</span>'}</p>")
#             status_html = '<span class="ok">✅ 一致</span>' if same_len else '<span class="bad">❌ 不一致</span>'
#             f.write(f"<p>长度比较：message.encode={lb}，input_ids[0]={la} {status_html}</p>")
#             f.write(f"<p>前 {min(la, lb)} 个位置的匹配率：<b>{ratio*100:.2f}%</b></p>")

#             # 展示前若干处差异
#             if diffs:
#                 f.write("<div class='rowtitle'>前若干处差异（index, input_ids[i], msg_encode[i]）</div>")
#                 f.write("<table class='table'><tr><th>idx</th><th>input_ids</th><th>msg_encode</th></tr>")
#                 for idx, ai, bi in diffs:
#                     f.write(f"<tr><td>{idx}</td><td>{ai}</td><td>{bi}</td></tr>")
#                 f.write("</table>")
#             else:
#                 f.write("<p class='ok'>未发现差异（在比较范围内）。</p>")

#             # 可视化头 50 个 token 文本（帮助定位特殊字符）
#             if tok is not None:
#                 def ids_to_readable(arr, limit=50):
#                     toks = []
#                     for t in arr[:limit]:
#                         t = int(t)
#                         toks.append(SPECIAL_TOKENS.get(t, tok.convert_ids_to_tokens(t)))
#                     return toks

#                 f.write("<div class='rowtitle'>input_ids[0] → tokens（前50）</div>")
#                 f.write("<div class='code'>" + " ".join(ids_to_readable(ids[0].tolist(), 50)) + "</div>")

#                 f.write("<div class='rowtitle'>message.encode → tokens（前50）</div>")
#                 f.write("<div class='code'>" + " ".join(ids_to_readable(enc_from_msg, 50)) + "</div>")

#         f.write("</div><hr>")

#         # ===== 统计徽标 =====
#         f.write('<div class="header">')
#         f.write(f'<div class="badge">Batch size: {B}</div>')
#         f.write(f'<div class="badge">Seq len: {T}</div>')
#         if args.tokenizer:
#             f.write(f'<div class="badge">Tokenizer: {args.tokenizer}</div>')
#         f.write("</div>")

#         f.write('<div class="legend">')
#         f.write('<div><span class="chip" style="background:#d00"></span>Loss=1 (红)</div>')
#         f.write('<div><span class="chip" style="background:#000"></span>Loss=0 (黑)</div>')
#         f.write('<div><span class="chip" style="background:#0a0"></span>Attn=1 (绿)</div>')
#         f.write('<div><span class="chip" style="background:#fff;border:1px solid #7c3aed"></span>特殊Token（黄底紫框）</div>')
#         f.write('</div>')

#         # ===== 主体可视化 =====
#         for i in range(B):
#             f.write('<div class="section">')
#             f.write(f'<div class="rowtitle">序列 <span class="idx">#{i}</span> : IDs</div>')
#             # 中性一行（仅呈现特殊 token 的背景，不改文字颜色）
#             zeros = np.zeros_like(ids[i])
#             f.write(make_line_html(ids[i], zeros, mode="neutral", tokenizer=tok))

#             f.write('<div class="rowtitle">Loss Mask 视图（红/黑）</div>')
#             f.write(make_line_html(ids[i], loss[i], mode="loss", tokenizer=tok))

#             f.write('<div class="rowtitle">Attention Mask 视图（绿/黑）</div>')
#             f.write(make_line_html(ids[i], attn[i], mode="attn", tokenizer=tok))
#             f.write('</div><hr>')

#         f.write("</body>")
#         f.write(HTML_TAIL)

#     print(f"[ok] wrote {args.out}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import html  # for HTML escape

# =================== 可选：HF tokenizer ===================
def maybe_load_tokenizer(tok_name_or_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_name_or_path, use_fast=True)
        return tok
    except Exception as e:
        print(f"[warn] cannot load tokenizer: {e}")
        return None

# =================== 特殊 token 定义 ===================
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

# =================== HTML 模板 ===================
HTML_HEAD = """<!doctype html>
<html lang="zh">
<meta charset="utf-8">
<title>Tensor Mask Viewer</title>
<style>
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }
h1 { margin-bottom: 8px; }
h2 { margin: 12px 0 6px; }
.section { margin: 18px 0 28px; }
.rowtitle { font-weight: 700; margin: 8px 0 4px; }
.tokenline { display: flex; flex-wrap: wrap; gap: 6px; line-height: 1.8; margin-bottom: 6px; }
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
.small { color:#666; font-size: 12px;}
hr { border: none; border-top: 1px solid #eee; margin: 18px 0; }
.header { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
.badge { font-size:12px; padding:2px 6px; border-radius:12px; background:#f1f1f1;}
.idx { color:#888; }
.table { border-collapse: collapse; font-size: 12px; }
.table td, .table th { border: 1px solid #eee; padding: 6px 8px; }
.ok { color: #0a0; }
.bad { color: #d00; }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#fafafa; border:1px solid #eee; padding:8px; border-radius:8px; white-space: pre-wrap; }
.legend { display: flex; gap: 18px; margin: 8px 0 16px;}
.legend .chip { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle;}
</style>
"""

HTML_TAIL = "</html>"

# =================== 工具函数 ===================
def html_safe(s: str) -> str:
    return html.escape(s, quote=False)

def pretty_bpe_token(t: str) -> str:
    """仅用于显示的美化（不影响对齐逻辑）"""
    if t == "Ċ":
        return "\\n"            # 按需：显示为字面量 \n
    if t and t.startswith("Ġ"):
        return " " + t[1:]      # 显示一个真实空格
    return t

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr[None, :]
    return arr

def color_for_mask(mask_value, mode):
    """
    mode: 'loss' or 'attn' or 'neutral'
    """
    if mode == 'loss':
        return "#d00" if mask_value == 1 else "#000"  # 红/黑
    if mode == 'attn':
        return "#0a0" if mask_value == 1 else "#000"  # 绿/黑
    return "#000"

def token_label(token_id: int, tokenizer=None):
    """返回展示用的 token 文本（已转义、已美化）"""
    if token_id in SPECIAL_TOKENS:
        return html_safe(SPECIAL_TOKENS[token_id])

    if tokenizer is not None:
        try:
            raw = tokenizer.convert_ids_to_tokens(int(token_id))
            return html_safe(pretty_bpe_token(raw))
        except Exception:
            pass
    return str(int(token_id))

def make_lines_html(ids_row, mask_row, mode, tokenizer=None):
    """
    将一个序列渲染为若干行：遇到 Ċ 就换到新行（在 flex 容器中 <br> 不可靠）
    """
    rows_html = []
    current = []

    for j, tid in enumerate(ids_row):
        mv = int(mask_row[j]) if np.ndim(mask_row) > 0 else int(mask_row)
        color = color_for_mask(mv, mode)

        # 原始 token 文本（用于判断是否 Ċ）
        raw_tok = None
        if tokenizer is not None:
            try:
                raw_tok = tokenizer.convert_ids_to_tokens(int(tid))
            except Exception:
                raw_tok = None

        label = token_label(int(tid), tokenizer)
        special_cls = " special" if int(tid) in SPECIAL_TOKENS else ""
        current.append(f'<span class="token{special_cls}" style="color:{color}">{label}</span>')

        # 真·换行：遇到 Ċ 收一行
        if raw_tok == "Ċ":
            rows_html.append('<div class="tokenline">' + "".join(current) + "</div>")
            current = []

    if current:
        rows_html.append('<div class="tokenline">' + "".join(current) + "</div>")

    return "".join(rows_html)

def compare_ids(a, b, max_show=50):
    """
    a, b: list[int] 或 1D np.array
    返回长度是否一致、匹配比例、前若干处差异 [(idx, a_i, b_i), ...]
    """
    la, lb = len(a), len(b)
    L = min(la, lb)
    matches = 0
    diffs = []
    for i in range(L):
        if int(a[i]) == int(b[i]):
            matches += 1
        elif len(diffs) < max_show:
            diffs.append((i, int(a[i]), int(b[i])))
    ratio = (matches / L) if L > 0 else 0.0
    same_len = (la == lb)
    return same_len, ratio, diffs, la, lb

# =================== 主流程 ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="./debug/input_ids.npy")
    ap.add_argument("--loss", default="./debug/loss_mask.npy")
    ap.add_argument("--attn", default="./debug/attention_mask.npy")
    ap.add_argument("--msg", default="./debug/message.txt")    # 解码后的字符串
    ap.add_argument("--out", default="./debug/inspect_v3.html")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer 名称或本地路径（如 Qwen/Qwen2.5-VL-7B-Instruct）")
    ap.add_argument("--max_diffs", type=int, default=50, help="差异表最多显示多少条")
    args = ap.parse_args()

    # 载入 numpy
    ids = np.load(args.ids, allow_pickle=False)
    loss = np.load(args.loss, allow_pickle=False)
    attn = np.load(args.attn, allow_pickle=False)

    ids = ensure_2d(ids)
    loss = ensure_2d(loss)
    attn = ensure_2d(attn)

    B, T = ids.shape
    print(f"[info] loaded ids={ids.shape}, loss={loss.shape}, attn={attn.shape}")

    # 载入 tokenizer（可选）
    tok = maybe_load_tokenizer(args.tokenizer) if args.tokenizer else None

    # 读取 message，并尝试 encode
    msg_str = None
    enc_from_msg = None
    if os.path.exists(args.msg):
        with open(args.msg, "r", encoding="utf-8") as f:
            msg_str = f.read()
        if tok is not None:
            try:
                enc_from_msg = tok.encode(msg_str, add_special_tokens=False)
                print(f"[check] message re-encoded length={len(enc_from_msg)}, input_ids length={T}")
            except Exception as e:
                print(f"[warn] cannot encode message: {e}")
        else:
            print("[info] tokenizer not provided; skipping message encode check")
    else:
        print(f"[info] message file not found at {args.msg}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(HTML_HEAD)
        f.write("<body>")
        f.write("<h1>Tensor Mask Viewer</h1>")

        # ===== 顶部：原文/直读预览（可快速确认不是乱码） =====
        if msg_str is not None:
            f.write('<div class="section">')
            f.write("<h2>Message 原文</h2>")
            f.write("<pre class='code'>" + html_safe(msg_str[:8000]) + "</pre>")

            if tok is not None:
                try:
                    decoded_ids0 = tok.decode(ids[0].tolist(), skip_special_tokens=False)
                    f.write("<h2>decode(input_ids[0]) 直读</h2>")
                    f.write("<pre class='code'>" + html_safe(decoded_ids0[:8000]) + "</pre>")
                except Exception as e:
                    f.write(f"<p class='small'>decode 失败：{html_safe(str(e))}</p>")
            f.write("</div><hr>")

        # ===== Message ↔ Input IDs 对比 =====
        f.write('<div class="section">')
        f.write("<h2>Message ↔ Input IDs 对比</h2>")

        if msg_str is None:
            f.write("<p class='small'>未找到 message.txt，跳过对比。</p>")
        elif tok is None:
            f.write("<p class='small'>未提供 --tokenizer，跳过对比。</p>")
        else:
            same_len, ratio, diffs, la, lb = compare_ids(ids[0].tolist(), enc_from_msg, max_show=args.max_diffs)
            status_html = '<span class="ok">✅ 一致</span>' if same_len else '<span class="bad">❌ 不一致</span>'
            f.write(f"<p>长度比较：message.encode={lb}，input_ids[0]={la} {status_html}</p>")
            f.write(f"<p>前 {min(la, lb)} 个位置的匹配率：<b>{ratio*100:.2f}%</b></p>")

            if diffs:
                f.write("<div class='rowtitle'>前若干处差异（index, input_ids[i], msg_encode[i]）</div>")
                f.write("<table class='table'><tr><th>idx</th><th>input_ids</th><th>msg_encode</th></tr>")
                for idx, ai, bi in diffs:
                    f.write(f"<tr><td>{idx}</td><td>{ai}</td><td>{bi}</td></tr>")
                f.write("</table>")
            else:
                f.write("<p class='ok'>未发现差异（在比较范围内）。</p>")

            # 可视化头 50 个 token 文本（帮助定位特殊字符）
            def ids_to_readable(arr, limit=50):
                toks = []
                for t in arr[:limit]:
                    t = int(t)
                    toks.append(SPECIAL_TOKENS.get(t, tok.convert_ids_to_tokens(t)))
                return toks

            f.write("<div class='rowtitle'>input_ids[0] → tokens（前50）</div>")
            f.write("<div class='code'>" + " ".join(map(html_safe, ids_to_readable(ids[0].tolist(), 50))) + "</div>")

            f.write("<div class='rowtitle'>message.encode → tokens（前50）</div>")
            f.write("<div class='code'>" + " ".join(map(html_safe, ids_to_readable(enc_from_msg, 50))) + "</div>")

        f.write("</div><hr>")

        # ===== 统计徽标 =====
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

        # ===== 主体可视化（按 Ċ 分多行） =====
        for i in range(B):
            f.write('<div class="section">')
            f.write(f'<div class="rowtitle">序列 <span class="idx">#{i}</span> : IDs</div>')
            zeros = np.zeros_like(ids[i])
            f.write(make_lines_html(ids[i], zeros, mode="neutral", tokenizer=tok))

            f.write('<div class="rowtitle">Loss Mask 视图（红/黑）</div>')
            f.write(make_lines_html(ids[i], loss[i], mode="loss", tokenizer=tok))

            f.write('<div class="rowtitle">Attention Mask 视图（绿/黑）</div>')
            f.write(make_lines_html(ids[i], attn[i], mode="attn", tokenizer=tok))
            f.write('</div><hr>')

        f.write("</body>")
        f.write(HTML_TAIL)

    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
