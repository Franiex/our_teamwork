import os
import csv
import base64
import time
from openai import OpenAI

# =======================
# 0. 开关（非常重要）
# =======================
USE_MOCK = True   # ←←← 先设为 True，100% 能跑通

# =======================
# 1. 配置
# =======================
API_KEY = os.getenv("OPENAI_API_KEY") or "sk-xxxx"
API_BASE = os.getenv("OPENAI_API_BASE") or "https://api3.wlai.vip/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout=20  # 防止无限等待
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "For_AI")
OUTPUT_HTML = os.path.join(BASE_DIR, "ai_insight_report.html")

# =======================
# 2. AI 洞察
# =======================
def generate_insight(content: str, filename: str) -> str:
    print(f"  -> 生成洞察: {filename}")

    if USE_MOCK:
        return f"""• 该文件已完成预处理，可直接用于分析
• 数据结构清晰，适合趋势与对比研究
• 建议结合其他文件进行综合解读"""

    prompt = f"""
你是一名数据分析专家，请基于以下内容总结 3–5 条关键发现：

{content}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ⚠️ 稳定可用
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


# =======================
# 3. CSV 预览
# =======================
def read_csv_preview(path, max_rows=5):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= max_rows:
                break
    return rows


# =======================
# 4. 图片 base64
# =======================
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# =======================
# 5. HTML 生成
# =======================
def generate_html():
    print("开始生成 HTML 报告...")

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>AI 自动化洞察报告</title>",
        "<style>",
        "body{font-family:Arial;margin:40px}",
        "table{border-collapse:collapse}",
        "td,th{border:1px solid #aaa;padding:6px}",
        ".insight{background:#f2f2f2;padding:10px;margin-bottom:30px}",
        "img{max-width:900px}",
        "</style></head><body>",
        "<h1>AI 自动化数据洞察报告</h1>"
    ]

    file_count = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".csv"):
                print(f"处理 CSV: {file}")
                preview = read_csv_preview(path)
                text = "\n".join([", ".join(r) for r in preview])
                insight = generate_insight(text, file)

                html.append(f"<h2>{file}</h2><table>")
                for r in preview:
                    html.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
                html.append("</table>")
                html.append(f"<div class='insight'><pre>{insight}</pre></div>")
                file_count += 1

            elif file.endswith(".png"):
                print(f"处理 PNG: {file}")
                img64 = image_to_base64(path)
                insight = generate_insight("图像分析", file)

                html.append(f"<h2>{file}</h2>")
                html.append(f"<img src='data:image/png;base64,{img64}'/>")
                html.append(f"<div class='insight'><pre>{insight}</pre></div>")
                file_count += 1

    html.append("</body></html>")

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print("HTML 生成完成")


# =======================
# 6. 主入口
# =======================
if __name__ == "__main__":
    print("程序启动")
    generate_html()
    print(f"报告路径: {OUTPUT_HTML}")
