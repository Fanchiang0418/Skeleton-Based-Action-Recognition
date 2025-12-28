import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ✅ 讀取資料
X = np.load("X.npy")
y = np.load("y.npy")

# ✅ 自動判斷分類數
num_classes = len(np.unique(y))
print("✅ 動作類別數：", num_classes)

# 建立 LSTM 模型
model = models.Sequential([
    layers.LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # ✅ 根據類別數動態設定
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ 訓練模型
model.fit(X, y, epochs=10, batch_size=32)  # 建議 epochs 增加到 10

# ✅ 測試預測
pred = model.predict(X[:1])
print("第一個樣本預測類別 ID：", np.argmax(pred))

# ✅ 儲存模型
model.save("lstm_action_model.h5")
print("✅ 模型已儲存為 lstm_action_model.h5")
