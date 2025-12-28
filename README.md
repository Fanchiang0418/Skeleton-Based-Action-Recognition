# Skeleton-Based-Action-Recognition
基於骨架的動作辨識系統，使用 MediaPipe 擷取人體關節序列，並以 LSTM 進行時序動作分類。

1.MediaPipe > 鏡頭抓取關節節點 (pose_test.py)

2.MediaPipe > 影片抓取關節節點(影片 → 骨架座標 → CSV) (extract_pose.py) (skeleton_data) (your_video.mp4)

3.CSV → LSTM 輸入格式的程式 (prepare_lstm_data.py) (X.npy) (y.npy) //pip install pandas numpy

4.完整 LSTM 訓練腳本 (train_lstm.py) //py -3.10 -m pip install tensorflow

5.模型 (lstm_action_model.h5)

6.Matplotlib 會打開一個視窗，顯示影片第一幀的骨架點(33個關節) (plot_skeleton.py)

7.播放第一個樣本30幀的骨架動畫 (skeleton_animation.py)

8.sample_index → 可以改 0 → 1, 2, 3 來播放不同動作樣本、fps → 調整播放速度(數字越大播放越快) (skeleton_animation_all.py)

9.準備多個影片資料 > 自動提取骨架 + 產生 (X, y) > 讀取數值產生模型 > 測試AI (prepare_multi_video_data.py) (data) > (train_lstm2.py) > (predict_action2.py)

10.新增自然語言處理 (test_llama2.py) (predict_action_ollama.py) //pip install ollama
