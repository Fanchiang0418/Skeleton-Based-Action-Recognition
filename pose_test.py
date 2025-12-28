import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # 使用攝影機，改成 "video.mp4" 可讀影片

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC鍵退出
        break

cap.release()
cv2.destroyAllWindows()
