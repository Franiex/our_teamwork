import os
import csv
import base64
import time
from openai import OpenAI

# =======================
# 0. Switch (Very Important)
# =======================
USE_MOCK = False  # Set to True for offline testing; guaranteed to run

# =======================
# 1. Configuration
# =======================
API_KEY = 'sk-KcKAu5U9JBEbbq0wReeTeFVm60lMqvrmYqjDYQgQ4MHF6Jl2'
API_BASE = 'https://api3.wlai.vip/v1'

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout=20  # Prevent infinite waiting
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "For_AI")
OUTPUT_HTML = os.path.join(BASE_DIR, "ai_insight_report.html")

# =======================
# 2. AI Insight Generation (English Output)
# =======================
def generate_insight(content: str, filename: str) -> str:
    print(f"  -> Generating insight (EN): {filename}")

    if USE_MOCK:
        return (
            "• The file has been successfully preprocessed and is ready for analysis\n"
            "• The data structure is clear and suitable for trend and comparison studies\n"
            "• It is recommended to combine this dataset with other files for deeper insights"
        )

    prompt = f"""
You are a professional data analyst.

Based on the following content, extract 3 to 5 key analytical insights.
Your response must:
- Be written in English
- Use concise and professional business language
- Focus on patterns, trends, data quality, or analytical value
- Avoid repeating raw data

Content:
{content}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# =======================
# 3. CSV Preview Reader
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
# 4. Convert Image to Base64
# =======================
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# =======================
# 5. HTML Report Generation
# =======================
def generate_html():
    print("Starting HTML report generation...")

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>AI Automated Insight Report</title>",
        "<style>",
        "body{font-family:Arial;margin:40px}",
        "table{border-collapse:collapse}",
        "td,th{border:1px solid #aaa;padding:6px}",
        ".insight{background:#f2f2f2;padding:10px;margin-bottom:30px}",
        "img{max-width:900px}",
        "</style></head><body>",
        "<h1>AI Automated Data Insight Report</h1>"
    ]

    file_count = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".csv"):
                print(f"Processing CSV file: {file}")
                preview = read_csv_preview(path)
                text = "\n".join([", ".join(r) for r in preview])
                insight = generate_insight(text, file)

                html.append(f"<h2>{file}</h2><table>")
                for r in preview:
                    html.append(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                    )
                html.append("</table>")
                html.append(f"<div class='insight'><pre>{insight}</pre></div>")
                file_count += 1

            elif file.endswith(".png"):
                print(f"Processing PNG image: {file}")
                img64 = image_to_base64(path)
                insight = generate_insight("Image-based analytical context", file)

                html.append(f"<h2>{file}</h2>")
                html.append(f"<img src='data:image/png;base64,{img64}'/>")
                html.append(f"<div class='insight'><pre>{insight}</pre></div>")
                file_count += 1

    html.append("</body></html>")

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"HTML generation completed. Files processed: {file_count}")


# =======================
# 6. Program Entry Point
# =======================
if __name__ == "__main__":
    print("Program started")
    generate_html()
    print(f"Report saved at: {OUTPUT_HTML}")

