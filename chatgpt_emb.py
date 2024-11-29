import openai
import pandas as pd
import requests
import numpy as np
import json
from openai import OpenAI

client = OpenAI(
    api_key='Your GPT api key.',
    base_url='https://api.chatanywhere.tech/v1'
)
def gpt_embeddings(texts):

    # 设置API密钥和自定义的API基础URL
    api_key = 'Your GPT api key.'  # 付费
    base_url = 'https://api.chatanywhere.tech/v1/embeddings'

    embedding_dimensions = 256  # 统一embedding的尺寸

    # 定义请求的Header和Body
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # embedding_list = []
    # for text in texts:
    data = {
        # "input": "gender: 1, age: 26, high blood pressure: 1, diabetes: 0, smoke: 0, alcoholic: 1",
        "input": f"{texts}",
        # "model": "text-embedding-ada-002"
        "model": "text-embedding-3-large",
        "dimensions": embedding_dimensions
    }

    # 发送POST请求
    response = requests.post(base_url, headers=headers, json=data)

    # 检查响应状态并处理数据
    if response.status_code == 200:
        # print("Embeddings retrieved successfully!")
        embeddings = response.json()
        embeddings = embeddings['data'][0]['embedding']
        # embedding_list.append(embeddings)
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings
    else:
        print(f"Failed to retrieve embeddings: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    test_texts = [
        "gender: 1, age: 26, high blood pressure: 1, diabetes: 0, smoke: 0, alcoholic: 1",
        "gender: 0, age: 34, high blood pressure: 0, diabetes: 1, smoke: 1, alcoholic: 0",
    ]
    gpt_embs = gpt_embeddings(test_texts)
    print(gpt_embs)