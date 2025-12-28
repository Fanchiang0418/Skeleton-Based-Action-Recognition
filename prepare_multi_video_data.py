import cv2
import mediapipe as mp
import numpy as np
import os
import re

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ 影片資料夾
video_folder = "data"
time_steps = 30
X_list, y_list = [], []
action_dict = {}  # 動作名稱 → ID

for video in os.listdir(video_folder):
    if not video.endswith(".mp4"):
        continue

    # ✅ 只取動作主名稱 (jump1.mp4 → jump)
    action = re.match(r"[a-zA-Z]+", video).group(0).lower()

    if action not in action_dict:
        action_dict[action] = len(action_dict)

    cap = cv2.VideoCapture(os.path.join(video_folder, video))
    frames = []
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

    # 切成 time_steps 片段
    for i in range(len(frames) - time_steps):
        X_list.append(frames[i:i+time_steps])
        y_list.append(action_dict[action])

# 轉成 numpy
X = np.array(X_list)  # (samples, 30, 132)
y = np.array(y_list)  # (samples, )

# ✅ 儲存資料
np.save("X.npy", X)
np.save("y.npy", y)

import json
json.dump(action_dict, open("action_dict.json", "w"))

print("✅ 產生 X.npy & y.npy")
print("✅ 動作標籤：", action_dict)
