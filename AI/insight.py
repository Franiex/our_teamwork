#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, csv
from pathlib import Path
from datetime import datetime

# Try import langchain (optional)
USE_LANGCHAIN = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    USE_LANGCHAIN = True
except Exception:
    USE_LANGCHAIN = False

# ---------- utils ----------
def read_csv_sample(path, max_rows=3):
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            sample = []
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                sample.append({k: (v if v is not None else "") for k, v in r.items()})
        return cols, sample
    except Exception:
        return [], []

def list_csvs(dirpath):
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        return {}
    out = {}
    for f in sorted(p.glob("*.csv")):
        cols, sample = read_csv_sample(str(f), max_rows=3)
        out[f.name] = {"path": str(f), "columns": cols, "sample_rows": sample}
    return out

def list_graphs(dirpath):
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        return []
    exts = {'.png','.jpg','.jpeg','.svg'}
    return [f.name for f in sorted(p.iterdir()) if f.suffix.lower() in exts]

# ---------- robust HTTP call (chat.completions + responses polling) ----------
def extract_text_from_json(j):
    # try common fields
    try:
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
                        return "".join([p for p in parts if isinstance(p, str)])
            t0 = c0.get("text")
            if isinstance(t0, str) and t0.strip():
                return t0
    except Exception:
        pass
    # try responses-like output
    try:
        out = j.get("output") or j.get("results") or j.get("data") or []
        def find_text(o):
            if isinstance(o, str) and o.strip():
                return o
            if isinstance(o, dict):
                for k in ("text","content","message","output","parts","results"):
                    if k in o:
                        r = find_text(o[k])
                        if r:
                            return r
                if "content" in o and isinstance(o["content"], list):
                    for c in o["content"]:
                        if isinstance(c, dict):
                            for key in ("text","title","caption"):
                                if key in c and isinstance(c[key], str) and c[key].strip():
                                    return c[key]
            if isinstance(o, list):
                for it in o:
                    r = find_text(it)
                    if r:
                        return r
            return None
        return find_text(out)
    except Exception:
        pass
    return None

def call_via_http(api_key, api_base, model_name, system_message, user_prompt, timeout=120):
    import urllib.request, urllib.error
    if not api_key:
        return None
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {api_key}"}
    # 1) try chat/completions
    url1 = api_base.rstrip("/") + "/chat/completions"
    payload1 = {"model":model_name, "messages":[{"role":"system","content":system_message},{"role":"user","content":user_prompt}], "max_tokens":1500, "temperature":0.0}
    try:
        data = json.dumps(payload1).encode("utf-8")
        req = urllib.request.Request(url1, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            j = json.loads(body)
            text = extract_text_from_json(j)
            if text:
                return text
            # else continue
            raw_j = j
    except urllib.error.HTTPError as e:
        try:
            raw_j = json.loads(e.read().decode("utf-8"))
        except Exception:
            raw_j = None
    except Exception:
        raw_j = None

    # 2) try /responses (and poll if incomplete)
    url2 = api_base.rstrip("/") + "/responses"
    payload2 = {"model":model_name, "input": system_message + "\n\n" + user_prompt, "max_output_tokens":1500, "temperature":0.0}
    try:
        data = json.dumps(payload2).encode("utf-8")
        req = urllib.request.Request(url2, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            j = json.loads(body)
            text = extract_text_from_json(j)
            if text:
                return text
            # if incomplete, poll using id
            resp_id = j.get("id") or j.get("response_id")
            status = j.get("status")
            if resp_id and status and status not in ("succeeded","completed","complete"):
                poll_url = api_base.rstrip("/") + "/responses/" + str(resp_id)
                waited = 0
                interval = 1.0
                max_wait = min(timeout, 120)
                while waited < max_wait:
                    time.sleep(interval)
                    waited += interval
                    interval = min(interval*1.5, 5.0)
                    try:
                        reqp = urllib.request.Request(poll_url, headers=headers, method="GET")
                        with urllib.request.urlopen(reqp, timeout=10) as pr:
                            bodyp = pr.read().decode("utf-8")
                            jp = json.loads(bodyp)
                            txt = extract_text_from_json(jp)
                            if txt:
                                return txt
                            st = jp.get("status")
                            if st in ("succeeded","completed","complete"):
                                txt = extract_text_from_json(jp)
                                if txt:
                                    return txt
                                return json.dumps(jp, ensure_ascii=False)
                    except Exception:
                        pass
                return json.dumps(j, ensure_ascii=False)
            return json.dumps(j, ensure_ascii=False)
    except urllib.error.HTTPError as e:
        try:
            return json.dumps(raw_j, ensure_ascii=False) if raw_j is not None else e.read().decode("utf-8")
        except Exception:
            return None
    except Exception:
        return json.dumps(raw_j, ensure_ascii=False) if raw_j is not None else None

def call_with_langchain(model_name, system_message, user_prompt):
    try:
        llm = ChatOpenAI(model_name=model_name)
        messages = [SystemMessage(content=system_message), HumanMessage(content=user_prompt)]
        if hasattr(llm, "invoke"):
            out = llm.invoke(messages)
            return getattr(out, "content", str(out))
        else:
            resp = llm(messages)
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "content"):
                return resp.content
            return str(resp)
    except Exception:
        return None

# ---------- two-step workflow ----------
def run_two_step(data_dir, graph_dir, model_name, output_dir):
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    csvs = list_csvs(data_dir)
    graphs = list_graphs(graph_dir)
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    system_msg = "你是一个精确的中文数据分析助理。回答简洁、要点化。"

    # Step 1: per-file short summaries
    summaries = {}
    for name, meta in csvs.items():
        sample = meta.get("sample_rows") or []
        sample_txt = ""
        for r in sample:
            # join first up to 5 columns for brevity
            items = []
            for k,v in list(r.items())[:5]:
                items.append(f"{k}:{v}")
            sample_txt += "; ".join(items) + " | "
        prompt = f"请根据以下 CSV 文件名与样例行，生成1-2句中文摘要（强调关键数值或趋势），并在末尾用括号注明证据文件名。例如：'（证据: {name}）'.\n文件名: {name}\n样例: {sample_txt}"
        # call model (prefer langchain if available)
        resp = None
        if USE_LANGCHAIN:
            resp = call_with_langchain(model_name, system_msg, prompt)
        if resp is None and api_key:
            resp = call_via_http(api_key, api_base, model_name, system_msg, prompt)
        if resp is None:
            resp = f"（模拟）{name} 的简短摘要不可用。"
        summaries[name] = resp.strip()

    # Step 2: aggregate summaries -> final insights
    aggregate_lines = ["为生成最终见解，下面是每个文件的 1-2 句摘要："]
    for n,s in summaries.items():
        aggregate_lines.append(f"- {n}: {s}")
    aggregate_text = "\n".join(aggregate_lines)
    final_prompt = (
        "基于下面的每文件简短摘要，生成：\n"
        "1) TL;DR（1-2 行）。\n"
        "2) 8-12 条关键见解（每条标注证据来源及置信度 high/medium/low，简要说明不确定性）。\n"
        "3) 3 条可执行建议（按优先级）。\n"
        "4) 3 个幻灯片结构（每页 1 行标题 + 2 行演讲稿）。\n"
        "最后以严格的 JSON 格式输出：{\"tldr\":..., \"insights\":[{\"id\":..,\"text\":..,\"evidence\":[],\"confidence\":\"\"},...], \"recommendations\":[], \"slides\":[]}\n\n"
        + aggregate_text
    )
    final_resp = None
    if USE_LANGCHAIN:
        final_resp = call_with_langchain(model_name, system_msg, final_prompt)
    if final_resp is None and api_key:
        final_resp = call_via_http(api_key, api_base, model_name, system_msg, final_prompt)
    if final_resp is None:
        final_resp = "（模拟）未能从远程模型获取最终见解。"

    # Save outputs
    (outp / "prompt_used.txt").write_text(final_prompt, encoding="utf-8")
    (outp / "insights.txt").write_text(final_resp, encoding="utf-8")
    # attempt parse JSON at end
    parsed = None
    try:
        idx = final_resp.rfind("{")
        if idx != -1:
            maybe = final_resp[idx:]
            parsed = json.loads(maybe)
    except Exception:
        parsed = None
    if parsed:
        (outp / "insights.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (outp / "insights.json").write_text(json.dumps({"raw": final_resp}, ensure_ascii=False, indent=2), encoding="utf-8")
    # create simple md/html
    md = "# 自动化见解（LLM 输出）\n\n```\n" + final_resp + "\n```\n\n## 每文件摘要\n"
    for n,s in summaries.items():
        md += f"- **{n}**: {s}\n"
    (outp / "insights.md").write_text(md, encoding="utf-8")
    html = "<!doctype html><html><head><meta charset='utf-8'><title>Insights</title></head><body>"
    html += "<h1>自动化见解（LLM 输出）</h1><pre style='white-space:pre-wrap;'>" + final_resp + "</pre><h2>每文件摘要</h2><ul>"
    for n,s in summaries.items():
        html += f"<li><strong>{n}</strong>: {s}</li>"
    html += "</ul></body></html>"
    (outp / "insights.html").write_text(html, encoding="utf-8")
    print("已保存输出到：", str(outp.resolve()))
    return outp

# ---------- CLI ----------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir","-d", required=True)
    p.add_argument("--graph-dir","-g", default="")
    p.add_argument("--model","-m", default=os.environ.get("MODEL_NAME","gpt-4"))
    p.add_argument("--output","-o", default="insights_output")
    p.add_argument("--run-once", action="store_true")
    args = p.parse_args()
    if args.run_once:
        run_two_step(args.data_dir, args.graph_dir, args.model, args.output)
        return
    print("交互式模式：当前脚本推荐使用 --run-once。")

if __name__ == "__main__":
    main()