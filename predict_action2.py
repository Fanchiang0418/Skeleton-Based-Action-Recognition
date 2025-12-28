import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os

# ✅ 讀取模型
model = tf.keras.models.load_model("lstm_action_model.h5")

# ✅ 讀取動作標籤
with open("action_dict.json", "r") as f:
    action_dict = json.load(f)
reverse_dict = {v: k for k, v in action_dict.items()}

# ✅ 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

time_steps = 30
video_folder = "data"

# 🔥 批量處理所有影片
for video in os.listdir(video_folder):
    if not video.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, video)
    print(f"\n🎬 測試影片： {video}")

    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    print(f"✅ 偵測到骨架幀數： {len(frames)}/{total_frames}")

    # 👉 Padding: 如果幀數不足 30，自動補最後一幀
    if len(frames) < time_steps and len(frames) > 0:
        pad = np.repeat(frames[-1][np.newaxis, :], time_steps - len(frames), axis=0)
        frames = np.vstack([frames, pad])

    # 👉 AI 預測
    if len(frames) >= time_steps:
        clip = frames[:time_steps].reshape(1, time_steps, 132)
        pred = model.predict(clip)
        predicted_class = np.argmax(pred)
        predicted_action = reverse_dict[predicted_class]
        print(f"🎯 AI 預測動作： {predicted_action}（{pred[0][predicted_class]:.2f} 機率）")
    else:
        print("⚠️ 無法預測（影片無骨架）")
