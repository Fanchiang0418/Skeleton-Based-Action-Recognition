import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ✅ 載入骨架資料
X = np.load("X.npy")  # (samples, time_steps, 132)

# ✅ 參數設定
sample_index = 0     # 選擇要播放的樣本（0=第一個）
fps = 10             # 播放速度（每秒幀數）

# ✅ 選擇要播放的序列
sequence = X[sample_index]  # (time_steps, 132)
time_steps = sequence.shape[0]

# ✅ MediaPipe Pose 骨架連線
connections = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(11,23),(12,24),
    (23,25),(25,27),(24,26),(26,28),
    (23,24)
]

# ✅ 準備 Matplotlib 繪圖
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=50)
lines = [ax.plot([], [], 'b-', lw=2)[0] for _ in connections]
ax.set_xlim(0,1)
ax.set_ylim(-1,0)  # y 反轉
ax.set_title(f"Skeleton Animation - Sample {sample_index}")

# ✅ 更新每幀
def update(frame_idx):
    frame = sequence[frame_idx].reshape(33,4)  # (33 joints, 4 features)
    x, y = frame[:,0], -frame[:,1]
    scat.set_offsets(np.c_[x, y])
    for i, (a,b) in enumerate(connections):
        lines[i].set_data([x[a], x[b]], [y[a], y[b]])
    return scat, *lines

# ✅ 動畫控制
ani = animation.FuncAnimation(
    fig, update, frames=time_steps, interval=1000/fps, blit=True, repeat=True
)

plt.show()
