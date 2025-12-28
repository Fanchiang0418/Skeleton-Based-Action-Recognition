import cv2
import mediapipe as mp
import csv
import os

# ====【設定】====
video_path = "your_video.mp4"   # <-- 替換成你的影片檔案路徑
output_csv = "skeleton_data.csv"  # 輸出的 CSV 檔案名稱

# ====【初始化 MediaPipe】====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ====【讀取影片】====
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 無法讀取影片，請確認路徑是否正確")
    exit()

frame_idx = 0
landmark_names = [f"{i}_{c}" for i in range(33) for c in ("x","y","z","v")]

# ====【建立 CSV 檔案】====
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 寫入標題列：frame + 33個關節(x,y,z,可見度)
    writer.writerow(["frame"] + landmark_names)

    # ====【逐幀處理影片】====
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            row = [frame_idx]
            for lm in results.pose_landmarks.landmark:
                row += [lm.x, lm.y, lm.z, lm.visibility]
            writer.writerow(row)

        frame_idx += 1

cap.release()
print(f"✅ 完成！骨架資料已存為 {os.path.abspath(output_csv)}")
