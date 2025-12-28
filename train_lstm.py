import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ✅ 直接讀取剛剛準備好的資料
X = np.load("X.npy")       # LSTM 輸入
y = np.load("y.npy")       # 標籤（目前全是 0，可日後替換成真實標籤）

# 建立 LSTM 模型
model = models.Sequential([
    layers.LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 假設分類 2 種動作
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 訓練模型
model.fit(X, y, epochs=5, batch_size=32)

# 測試預測
pred = model.predict(X[:1])
print("第一個樣本預測類別：", np.argmax(pred))

# 儲存模型
model.save("lstm_action_model.h5")
print("✅ 模型已儲存為 lstm_action_model.h5")
