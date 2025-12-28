import numpy as np
import matplotlib.pyplot as plt

# 載入骨架資料
X = np.load("X.npy")

# 取第一個序列的第一幀
#frame = X[0,0].reshape(33,4)  # 33 關節 (x,y,z,v)

# 畫 2D 骨架 (x,y)
#plt.scatter(frame[:,0], -frame[:,1])  # y 取反，符合視覺
#plt.title("First Frame Skeleton")
#plt.show()

print(X)