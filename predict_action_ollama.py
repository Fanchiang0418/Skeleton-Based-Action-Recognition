import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
import requests

# ✅ LSTM 模型
model = tf.keras.models.load_model("lstm_action_model.h5")

# ✅ 動作標籤
with open("action_dict.json", "r") as f:
    action_dict = json.load(f)
reverse_dict = {v: k for k, v in action_dict.items()}

# ✅ MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

time_steps = 30
video_folder = "data"

# ✅ 本地模板（Fallback）
local_templates = {
    "jump": "舞者輕盈地跳躍，宛如風中的羽毛。",
    "wave": "舞者揮動雙手，如水波般柔美。",
    "run": "舞者疾速奔跑，散發強烈的生命力。"
}

# ✅ 呼叫 Ollama（支持 qwen/llama2/mistral）
def ollama_generate(prompt, model_name="qwen"):
    try:
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Ollama 生成失敗：{e}")
    return None

# ✅ 提取骨架特徵
def extract_features(frames):
    diffs = np.linalg.norm(np.diff(frames[:, :2], axis=0), axis=1)
    return {
        "avg_speed": round(float(np.mean(diffs)), 4),
        "max_speed": round(float(np.max(diffs)), 4)
    }

# ✅ 生成描述（優先 Ollama → Fallback 模板）
def get_description(action, features, model_name="qwen"):
    prompt = f"請用中文寫1-2句優雅且富有詩意的舞蹈評論，描述舞者正在表演「{action}」，平均速度{features['avg_speed']}，最大速度{features['max_speed']}。"
    text = ollama_generate(prompt, model_name)
    if text:
        return text
    return local_templates.get(action, f"舞者正在表演 {action}，動作流暢自然。")

# 🔥 處理影片
for video in os.listdir(video_folder):
    if not video.endswith(".mp4"):
        continue

    print(f"\n🎬 測試影片： {video}")
    frames = []
    cap = cv2.VideoCapture(os.path.join(video_folder, video))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks += [lm.x, lm.y, lm.z, lm.visibility]
            frames.append(landmarks)
    cap.release()

    frames = np.array(frames)
    if len(frames) < time_steps and len(frames) > 0:
        pad = np.repeat(frames[-1][np.newaxis, :], time_steps - len(frames), axis=0)
        frames = np.vstack([frames, pad])

    if len(frames) >= time_steps:
        clip = frames[:time_steps].reshape(1, time_steps, 132)
        pred = model.predict(clip)
        predicted_action = reverse_dict[np.argmax(pred)]
        features = extract_features(frames[:time_steps])

        # 🎨 生成詩意描述
        poetic_text = get_description(predicted_action, features, model_name="qwen")
        print(f"🎯 AI 動作： {predicted_action}（置信度 {np.max(pred):.2f}）")
        print(f"🎨 Qwen 詩意描述： {poetic_text}")
    else:
        print("⚠️ 無法預測（影片無骨架）")
