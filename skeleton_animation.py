import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ✅ 載入骨架資料
X = np.load("X.npy")  # shape = (samples, 30, 132)

# ✅ 取第一個樣本（30幀）
sequence = X[0]  # shape = (30, 132)

# ✅ 定義骨架連線（根據 MediaPipe Pose 關節）
connections = [
    (11,13),(13,15),(12,14),(14,16),  # 手臂
    (11,12),(11,23),(12,24),          # 上身
    (23,25),(25,27),(24,26),(26,28),  # 腿
    (23,24)                           # 髖部
]

# ✅ 準備繪圖
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=50)
lines = [ax.plot([], [], 'b-', lw=2)[0] for _ in connections]
ax.set_xlim(0,1)
ax.set_ylim(-1,0)  # y 反轉，符合視覺
ax.set_title("Skeleton Animation")

# ✅ 更新每幀骨架
def update(frame_idx):
    frame = sequence[frame_idx].reshape(33,4)  # (33 joints, 4 features)
    x, y = frame[:,0], -frame[:,1]
    scat.set_offsets(np.c_[x, y])
    for i, (a,b) in enumerate(connections):
        lines[i].set_data([x[a], x[b]], [y[a], y[b]])
    return scat, *lines

# ✅ 建立動畫
ani = animation.FuncAnimation(fig, update, frames=len(sequence), interval=200, blit=True)

plt.show()
