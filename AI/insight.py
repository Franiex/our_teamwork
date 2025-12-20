import os
import pandas as pd
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = '你的key'
os.environ['OPENAI_API_BASE'] = 'https://api3.wlai.vip/v1'

TABLE_DIR = "../data/final_data/globe/final_data/yearly_globe_data/tables"

def load_table_text():
    text = ""
    for file in os.listdir(TABLE_DIR):
        if file.endswith(".csv"):
            path = os.path.join(TABLE_DIR, file)
            df = pd.read_csv(path)
            text += f"\n【{file}】\n"
            text += df.head(10).to_string(index=False)
            text += "\n"
    return text

def build_prompt(table_text):
    return f"""
任务：自动化见解生成（Automated Insight Generation）

你是一名气候趋势分析专家。
请基于以下统计结果，生成关键洞察。

要求：
- 3–5 条结论
- 描述长期趋势
- 使用通俗语言
- 不复述表格

数据如下：
{table_text}
"""

def main():
    table_text = load_table_text()
    prompt = build_prompt(table_text)

    llm = ChatOpenAI(model_name='gpt-5')
    response = llm.invoke([HumanMessage(content=prompt)])

    print("\n=== 自动生成的数据洞察 ===\n")
    print(response.content)

if __name__ == "__main__":
    main()
