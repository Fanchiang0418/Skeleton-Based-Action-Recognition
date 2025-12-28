import requests
import json

def llama2_generate(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama2",  # 也可換成 "mistral"
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json().get("response", "").strip()

# 測試
text = llama2_generate("請用中文寫一句描述舞者跳躍的詩意句子，語氣要優雅且富有詩意。")
print("🎨 Llama2 中文描述：", text)
