#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insight.py - 以已处理的 CSV/图像为证据调用 LLM 生成自动化见解，并生成团队友好的查看产物（txt/json/md/html）。
特点：
- 零或最少第三方依赖：优先使用已安装的 langchain_openai（如果有），否则用标准库通过 OpenAI-compatible HTTP 接口调用。
- 如果无 API key 或调用失败，脚本生成确定性模拟输出，方便团队在没有密钥的情况下查看输出格式。
- 输出文件（insights_output/）包含：insights.txt, insights.json, prompt_used.txt, insights.md, insights.html
  - insights.md/insights.html 便于在 Jupyter Notebook / VSCode / 浏览器 中直接查看（HTML 会嵌入仓库中的图片引用/CSV 链接，未上传图片内容）。
- 安全：不要把 API key 写入代码或提交到仓库。请在运行前设置 OPENAI_API_KEY（或让团队自行设置）。
使用示例：
    export OPENAI_API_KEY="sk-..."
    export OPENAI_API_BASE="https://api.openai.com/v1"
    python insight.py --data-dir For_AI/FD/csv --graph-dir For_AI/FD/graph --model gpt-4 --run-once
    # 在 Jupyter 中查看 HTML：
    from IPython.display import IFrame
    IFrame("insights_output/insights.html", width="100%", height=600)
"""

from __future__ import annotations
import os
import sys
import argparse
import json
from pathlib import Path
import csv
from datetime import datetime
import urllib.request
import urllib.error
import html
import shutil

# Try to import langchain usage as in user's snippet (non-fatal)
USE_LANGCHAIN = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    USE_LANGCHAIN = True
except Exception:
    USE_LANGCHAIN = False

# -------------------- 辅助函数：读取 CSV、列出图像 --------------------
def read_csv_sample(path, max_rows=20):
    """读取 CSV 的表头与前若干行，返回列名和样本行（list of dict）以及总行数估计。"""
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            sample = []
            total = 0
            for i, r in enumerate(reader):
                total += 1
                if i < max_rows:
                    row = {k: (v if v is None or len(str(v)) <= 200 else str(v)[:197] + '...') for k, v in r.items()}
                    sample.append(row)
        # Count rows accurately
        count = 0
        with open(path, 'rb') as f:
            for _ in f:
                count += 1
        if count > 0:
            count = count - 1
        return cols, sample, count
    except Exception:
        try:
            with open(path, newline='', encoding='latin-1') as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                sample = []
                total = 0
                for i, r in enumerate(reader):
                    total += 1
                    if i < max_rows:
                        row = {k: (v if v is None or len(str(v)) <= 200 else str(v)[:197] + '...') for k, v in r.items()}
                        sample.append(row)
            return cols, sample, total
        except Exception:
            return [], [], 0

def list_csvs(dirpath):
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        return {}
    out = {}
    for f in sorted(p.glob("*.csv")):
        cols, sample, count = read_csv_sample(str(f))
        out[f.name] = {"path": str(f), "columns": cols, "sample_rows": sample, "rows_count": count}
    return out

def list_graphs(dirpath):
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        return []
    exts = {'.png', '.jpg', '.jpeg', '.svg'}
    return [f.name for f in sorted(p.iterdir()) if f.suffix.lower() in exts]

# -------------------- 构造 Prompt（中文） --------------------
PROMPT_SYSTEM = (
    "你是一个高级数据分析与科研写作助手，精通从已处理好的 CSV 表格与图表文件名中生成结构化、可操作的见解与报告。"
    "请严格按照用户要求输出：先给出中文可读报告（包含 TL;DR、8-12 条关键见解，每条附证据与置信度、3 条可执行建议、3 个幻灯片结构），"
    "随后以 JSON 格式返回结构化结果（字段必须包含：tldr, insights, recommendations, slides）。"
    "每条见解必须明确列出证据来源（CSV 文件名或图像文件名）和关键数值（若 CSV 有样例行请据样例给出数值），并对不确定性做简短说明（high/medium/low）。"
    "不要在输出中包含任何 API key 或本地敏感信息。"
)

PROMPT_TEMPLATE_INTRO = """
下面是当前仓库中已生成的“证据”，包括 CSV 表格的表头与若干样例行（仅为提示，完整数据在仓库中），以及图像文件名（图像本身不可见）。
请基于这些证据生成报告与结构化 JSON，严格满足 System 的输出要求。
"""

def build_prompt(csvs_meta, graphs_list):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [PROMPT_TEMPLATE_INTRO, f"当前时间(UTC): {now}\n", "CSV 文件清单与样例："]
    if not csvs_meta:
        lines.append("- 无 CSV 文件（请确认路径）")
    for fname, meta in csvs_meta.items():
        lines.append(f"- 文件: {fname}")
        lines.append(f"    - 路径: {meta.get('path')}")
        lines.append(f"    - 行数估计: {meta.get('rows_count')}")
        cols = meta.get('columns') or []
        lines.append(f"    - 列名: {cols}")
        sample_rows = meta.get('sample_rows') or []
        if sample_rows:
            lines.append(f"    - 前 {len(sample_rows)} 行样例：")
            for r in sample_rows:
                try:
                    lines.append("      - " + json.dumps(r, ensure_ascii=False))
                except Exception:
                    lines.append("      - (sample row)")
        else:
            lines.append("    - 无样例行或文件为空")
    lines.append("\n图像文件清单（模型无法直接查看图片；请根据文件名理解其含义）：")
    if not graphs_list:
        lines.append("- 无图像文件")
    else:
        for g in graphs_list:
            lines.append(f"- {g}")
    lines.append(
        "\n任务（请严格遵守格式）:\n"
        "1) 给出 TL;DR（1-2 行）。\n"
        "2) 给出 8-12 条关键见解（要点形式），每条后面标注证据来源（CSV 文件名或图像名）和关键数值/样例引用，并给出置信度（high/medium/low）与简短不确定性说明。\n"
        "3) 给出 3 条可执行建议（按优先级排序）。\n"
        "4) 给出 3 个幻灯片结构（每页 1 行标题 + 2 行左右演讲稿）。\n"
        "5) 最后以 JSON 格式输出结构化结果（tldr, insights, recommendations, slides）。\n"
        "注意：当引用样例数值时请精确到 2-4 位小数或如样例原样输出。\n"
    )
    return "\n".join(lines)

# -------------------- 调用 LLM（langchain 或 HTTP） --------------------
def call_with_langchain(model_name, system_message, user_prompt):
    try:
        llm = ChatOpenAI(model_name=model_name)
        messages = [SystemMessage(content=system_message), HumanMessage(content=user_prompt)]
        if hasattr(llm, "invoke"):
            out = llm.invoke(messages)
            content = getattr(out, "content", None)
            if content is None:
                try:
                    content = str(out)
                except Exception:
                    content = ""
            return content
        else:
            try:
                resp = llm(messages)
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "content"):
                    return resp.content
                if hasattr(resp, "generations"):
                    gens = resp.generations
                    if gens and len(gens) > 0 and len(gens[0]) > 0:
                        return gens[0][0].text
                return str(resp)
            except Exception:
                return None
    except Exception as e:
        print("使用 langchain 调用模型失败：", e, file=sys.stderr)
        return None

def call_via_http_openai(api_key, api_base, model_name, system_message, user_prompt, timeout=180):
            import urllib.request, urllib.error, time
            if not api_key:
                return None

            # If prompt string is extremely long, truncate conservatively (avoid token overflow)
            MAX_PROMPT_CHARS = 30000
            if isinstance(user_prompt, str) and len(user_prompt) > MAX_PROMPT_CHARS:
                user_prompt = user_prompt[-MAX_PROMPT_CHARS:]  # keep the tail (recent info)

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

            def extract_text_from_json(j):
                try:
                    # common chat/completions places
                    choices = j.get("choices", [])
                    if isinstance(choices, list) and choices:
                        c0 = choices[0]
                        msg = c0.get("message") or {}
                        if isinstance(msg, dict):
                            cont = msg.get("content")
                            if isinstance(cont, str) and cont.strip():
                                return cont
                            if isinstance(cont, (dict, list)):
                                parts = cont.get("parts") if isinstance(cont, dict) else cont
                                if isinstance(parts, list) and parts:
                                    return "".join(p for p in parts if isinstance(p, str))
                        t0 = c0.get("text")
                        if isinstance(t0, str) and t0.strip():
                            return t0
                        delta = c0.get("delta") or {}
                        if isinstance(delta, dict):
                            dt = delta.get("content") or delta.get("text")
                            if isinstance(dt, str) and dt.strip():
                                return dt
                except Exception:
                    pass

                # responses-like structure
                try:
                    out = j.get("output") or j.get("results") or j.get("data") or []

                    def find_text(obj):
                        if isinstance(obj, str) and obj.strip():
                            return obj
                        if isinstance(obj, dict):
                            for k in ("text", "content", "message", "output", "parts"):
                                if k in obj:
                                    res = find_text(obj[k])
                                    if res:
                                        return res
                        if isinstance(obj, list):
                            for it in obj:
                                res = find_text(it)
                                if res:
                                    return res
                        return None

                    txt = find_text(out)
                    if txt:
                        return txt
                except Exception:
                    pass
                return None

            # 1) Try chat/completions (max_tokens increased)
            url1 = api_base.rstrip("/") + "/chat/completions"
            payload1 = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 3000,  # request larger output (provider may cap)
                "temperature": 0.0
            }
            try:
                data1 = json.dumps(payload1).encode("utf-8")
                req1 = urllib.request.Request(url1, data=data1, headers=headers, method="POST")
                with urllib.request.urlopen(req1, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    j = json.loads(body)
                    text = extract_text_from_json(j)
                    if text:
                        return text
                    raw_j = j
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    body = ""
                raw_j = None
            except Exception:
                raw_j = None

            # 2) If chat returned no text, try /responses (some providers return actual text here)
            url2 = api_base.rstrip("/") + "/responses"
            payload2 = {
                "model": model_name,
                "input": system_message + "\n\n" + (user_prompt if len(user_prompt) < 20000 else user_prompt[-20000:]),
                "max_output_tokens": 3000,
                "temperature": 0.0
            }
            try:
                data2 = json.dumps(payload2).encode("utf-8")
                req2 = urllib.request.Request(url2, data=data2, headers=headers, method="POST")
                with urllib.request.urlopen(req2, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    j = json.loads(body)
                    text = extract_text_from_json(j)
                    if text:
                        return text
                    # try direct paths
                    out = j.get("output") or j.get("results") or []
                    if isinstance(out, list) and out:
                        first = out[0]
                        if isinstance(first, dict):
                            conts = first.get("content") or first.get("data") or []
                            if isinstance(conts, list):
                                for c in conts:
                                    if isinstance(c, dict):
                                        for key in ("text", "title", "caption"):
                                            if key in c and isinstance(c[key], str) and c[key].strip():
                                                return c[key]
                    return json.dumps(j, ensure_ascii=False)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    body = ""
                return json.dumps(raw_j, ensure_ascii=False) if raw_j is not None else body
            except Exception:
                return json.dumps(raw_j, ensure_ascii=False) if raw_j is not None else None


        # -------------------- 输出为 Markdown & HTML（便于团队查看） --------------------
def save_markdown_and_html(response_text, csvs_meta, graphs, out_dir):
    out_dir = Path(out_dir)
    md_path = out_dir / "insights.md"
    html_path = out_dir / "insights.html"

    # Save markdown: include raw response as preformatted block, then CSV & images list with links
    md_lines = []
    md_lines.append("# 自动化见解（LLM 输出）")
    md_lines.append("")
    md_lines.append("## 原始模型输出")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(response_text)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## 可用证据（CSV 与图片）")
    md_lines.append("")
    if csvs_meta:
        md_lines.append("### CSV 文件")
        for name, meta in csvs_meta.items():
            rel = make_relative_link(out_dir, meta.get("path"))
            md_lines.append(f"- **{name}** — 行数估计: {meta.get('rows_count')} — [文件]({rel})")
    else:
        md_lines.append("- 无 CSV 文件")
    md_lines.append("")
    if graphs:
        md_lines.append("### 图像文件（在支持的查看器中将显示）")
        for g in graphs:
            gpath = Path(g)
            # Try to generate relative path from output dir; assume graphs located in graph_dir (not full path here)
            # If graph filename is only name, try to find it in parent folders; otherwise link by name
            md_lines.append(f"- {g}  ![img]({g})")
    else:
        md_lines.append("- 无图像文件")
    md_text = "\n".join(md_lines)
    md_path.write_text(md_text, encoding="utf-8")

    # Save simple HTML that includes raw output and links / inline images (relative paths)
    safe_text = html.escape(response_text).replace("\n", "<br/>\n")
    html_lines = []
    html_lines.append("<!doctype html>")
    html_lines.append("<html><head><meta charset='utf-8'><title>Insights</title></head><body>")
    html_lines.append("<h1>自动化见解（LLM 输出）</h1>")
    html_lines.append("<h2>原始模型输出</h2>")
    html_lines.append(f"<div style='white-space:pre-wrap; font-family: monospace; background:#f8f8f8; padding:10px; border-radius:6px;'>{safe_text}</div>")
    html_lines.append("<h2>可用证据（CSV 与图片）</h2>")
    if csvs_meta:
        html_lines.append("<h3>CSV 文件</h3><ul>")
        for name, meta in csvs_meta.items():
            rel = make_relative_link(out_dir, meta.get("path"))
            html_lines.append(f"<li><strong>{html.escape(name)}</strong> — 行数估计: {meta.get('rows_count')} — <a href='{html.escape(rel)}' target='_blank'>打开文件</a></li>")
        html_lines.append("</ul>")
    else:
        html_lines.append("<p>- 无 CSV 文件</p>")
    if graphs:
        html_lines.append("<h3>图像文件（若路径正确，浏览器可直接显示）</h3>")
        for g in graphs:
            # try to locate graph file on disk relative to out_dir
            graph_path = find_graph_file(out_dir, g)
            if graph_path:
                # use relative path from HTML file to image
                rel = make_relative_link(out_dir, graph_path)
                html_lines.append(f"<div style='margin:8px 0;'><p>{html.escape(g)}</p><img src='{html.escape(rel)}' style='max-width:90%;height:auto;border:1px solid #ddd;padding:4px;background:#fff;'></div>")
            else:
                html_lines.append(f"<p>{html.escape(g)} (文件未找到于常见位置)</p>")
    else:
        html_lines.append("<p>- 无图像文件</p>")
    html_lines.append("</body></html>")
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

def make_relative_link(out_dir: Path, target_path: str) -> str:
    """返回从 out_dir 到 target_path 的相对链接（若无法 relativize，则返回绝对 path）。"""
    try:
        if not target_path:
            return ""
        t = Path(target_path)
        # If target is absolute and exists, try to compute relative to out_dir
        if t.exists():
            return str(t.relative_to(out_dir)) if t.is_relative_to(out_dir) else os.path.relpath(str(t), start=str(out_dir))
        else:
            # If not exists, return provided path as-is (likely relative already)
            return target_path
    except Exception:
        try:
            return os.path.relpath(str(target_path), start=str(out_dir))
        except Exception:
            return target_path

def find_graph_file(out_dir: Path, filename: str):
    """尝试在常见位置查找图像：同目录（out_dir）、相对工作目录、父级 For_AI/FD/graph 等。返回可用路径字符串或 None."""
    # 1) If filename already an absolute or relative path, check it
    p = Path(filename)
    if p.exists():
        return str(p)
    # 2) check common places relative to current working directory
    cand = Path.cwd() / filename
    if cand.exists():
        return str(cand)
    # 3) search in repository tree (limited depth)
    for root, dirs, files in os.walk(Path.cwd()):
        if filename in files:
            return str(Path(root) / filename)
    return None

# -------------------- 主流程 --------------------
def run_analysis(data_dir, graph_dir, model_name, output_dir):
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print("扫描 CSV 与图像目录...")
    csvs = list_csvs(data_dir) if data_dir else {}
    graphs = list_graphs(graph_dir) if graph_dir else []

    prompt = build_prompt(csvs, graphs)
    with open(outp / "prompt_used.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    system_message = PROMPT_SYSTEM
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    print(f"准备调用模型：{model_name}")
    response_text = None

    if USE_LANGCHAIN:
        print("尝试使用 langchain_openai.ChatOpenAI 调用（环境中已安装）。")
        response_text = call_with_langchain(model_name, system_message, prompt)
        if response_text:
            print("langchain 返回结果。")
    if response_text is None and api_key:
        print("使用 HTTP POST 调用 OpenAI-compatible 接口...")
        response_text = call_via_http_openai(api_key, api_base, model_name, system_message, prompt)
        if response_text:
            print("HTTP 调用返回结果。")
    if response_text is None:
        print("未提供 API KEY 或调用失败，进入模拟模式（确定性示例输出）。")
        response_text = simulated_output_example(csvs, graphs)

    # Save plain text
    with open(outp / "insights.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    # Try to parse JSON block at end
    parsed_json = None
    try:
        idx = response_text.rfind("{")
        if idx != -1:
            maybe = response_text[idx:]
            parsed_json = json.loads(maybe)
    except Exception:
        parsed_json = None

    if parsed_json:
        with open(outp / "insights.json", "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)
    else:
        # Save wrapper JSON with raw
        with open(outp / "insights.json", "w", encoding="utf-8") as f:
            json.dump({"raw": response_text}, f, ensure_ascii=False, indent=2)

    # Save Markdown and HTML for easy team viewing (Jupyter/Browser)
    save_markdown_and_html(response_text, csvs, graphs, outp)

    print("已保存输出到：", outp.resolve())
    print("建议在 Jupyter 中查看 HTML：from IPython.display import IFrame; IFrame('insights_output/insights.html', width='100%', height=600)")

def simulated_output_example(csvs_meta, graphs):
    tldr = "TL;DR: 示例 — 基于仓库中已生成的 CSV 与图像，全球温度呈长期上升趋势，近几十年加速明显。"
    insights = [
        {"id": 1, "text": "长期升温趋势（示例）：1880-2025 年度均值长期上升。", "evidence": ["temperature_timeline.png", "decade_summary.csv"], "confidence": "high"},
        {"id": 2, "text": "近 50 年加速（示例）：1975 起斜率高于整体。", "evidence": ["recent_trends.csv"], "confidence": "medium"},
    ]
    recommendations = [
        "将见解整理为 3 张幻灯片：概览、证据与建议（示例）。",
        "在报告中附上原始 CSV 链接以便可审计每项发现（示例）。",
        "对 1975+ 数据做更严格的回归并提供置信区间（示例）。"
    ]
    slides = [
        {"title": "TL;DR & 关键结论", "notes": "展示长期趋势图与近年加速的关键数值。"},
        {"title": "详细证据", "notes": "列出 decade_summary、warmest_years、recent_trends 的关键数值与图示。"},
        {"title": "建议与不确定性", "notes": "给出三条行动建议与主要不确定性来源。"}
    ]
    out = {
        "tldr": tldr,
        "insights": insights,
        "recommendations": recommendations,
        "slides": slides
    }
    pretty = "示例输出（模拟模式）：\n\n" + tldr + "\n\n关键见解示例：\n"
    for ins in insights:
        pretty += f"- ({ins['confidence']}) {ins['text']} 证据: {ins['evidence']}\n"
    pretty += "\nJSON 模拟:\n" + json.dumps(out, ensure_ascii=False, indent=2)
    return pretty

# -------------------- CLI / REPL --------------------
def main():
    parser = argparse.ArgumentParser(description="insight.py - 以已处理的 CSV/图像为证据，调用 LLM 生成自动化见解")
    parser.add_argument("--data-dir", "-d", required=True, help="CSV 目录，例如 For_AI/FD/csv")
    parser.add_argument("--graph-dir", "-g", required=False, help="图像目录，例如 For_AI/FD/graph")
    parser.add_argument("--model", "-m", default=os.environ.get("MODEL_NAME", "gpt-4"), help="模型名称，例如 gpt-4 或 gpt-5")
    parser.add_argument("--output", "-o", default="insights_output", help="输出目录")
    parser.add_argument("--run-once", action="store_true", help="运行一次后退出（非交互式）")
    args = parser.parse_args()

    data_dir = args.data_dir
    graph_dir = args.graph_dir or ""
    model = args.model
    outdir = args.output

    if args.run_once:
        run_analysis(data_dir, graph_dir, model, outdir)
        return

    print("启动交互模式。输入 'analyze' 触发一次分析并保存结果；输入 'exit' 退出。")
    while True:
        try:
            cmd = input("命令 ('analyze'/'exit'): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if cmd in ("exit", "quit"):
            print("退出。")
            break
        if cmd == "analyze":
            run_analysis(data_dir, graph_dir, model, outdir)
            print("分析完成。")
        else:
            print("未知命令。请输入 'analyze' 或 'exit'。")

if __name__ == "__main__":
    main()