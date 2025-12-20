import os
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# environment
os.environ['OPENAI_API_KEY'] = 'sk-KcKAu5U9JBEbbq0wReeTeFVm60lMqvrmYqjDYQgQ4MHF6Jl2'
os.environ['OPENAI_API_BASE'] = 'https://api3.wlai.vip/v1'

# LLM
llm = ChatOpenAI(model_name='gpt-5')
messages = []

# 交互循环
while True:
    user_input = input('你：')
    if user_input.lower() == 'exit':
        break
    # add history
    messages.append(HumanMessage(content=user_input))
    # use
    ai_output = llm.invoke(messages)
    ai_response = ai_output.content
    # input and keep response
    print(f'AI：{ai_response}')
    messages.append(AIMessage(content=ai_response))