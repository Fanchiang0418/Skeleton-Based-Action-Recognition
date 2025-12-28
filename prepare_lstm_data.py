import pandas as pd
import numpy as np

# 讀取骨架CSV
csv_path = "skeleton_data.csv"  # <-- 你的CSV路徑
df = pd.read_csv(csv_path)

# 去掉 frame 欄位，只保留骨架特徵
features = df.drop(columns=["frame"]).values  # shape = (frames, 132)

# 設定時間步（例如每30幀當一個序列）
time_steps = 30  
X = []
for i in range(len(features) - time_steps):
    X.append(features[i:i+time_steps])
X = np.array(X)  # shape = (samples, 30, 132)

# 建立假動作標籤 (例如都標成 0，後面你可換成真實標籤)
y = np.zeros(len(X))

# ✅ 儲存為 .npy，方便 LSTM 直接載入
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ LSTM 輸入 X 形狀：", X.shape)
print("✅ 資料已儲存為 X.npy 和 y.npy")