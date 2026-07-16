# 1_config.py
import os

# -------------------- 路径设置 --------------------
NORMAL_TRAIN_DIR = '/root/autodl-tmp/前排遮阳帘打开运行音频/ok'      # 训练用的正常样本文件夹
TEST_NORMAL_DIR = '/root/autodl-tmp/前排遮阳帘打开运行音频/test_normal'  # 测试集正常样本（用于评估）
TEST_ABNORMAL_DIR = '/root/autodl-tmp/前排遮阳帘打开运行音频/ng/88831-异响-静态-前排遮阳帘打开运行噪声-CH/Run 1'  # 测试集卡顿样本

# -------------------- 音频参数 --------------------
SAMPLE_RATE = 16000      # 采样率（Hz）


# -------------------- 特征参数 --------------------
N_MFCC = 13              # MFCC维数
N_MELS = 128             # Mel频谱图维数
HOP_LENGTH = 512
N_FFT = 1024

# -------------------- 模型训练参数 --------------------
AE_EPOCHS = 50
BATCH_SIZE = 32
GMM_COMPONENTS = 1
OCSVM_NU = 0.1
IFOREST_CONTAMINATION = 0.1

# -------------------- 输出 --------------------
OUTPUT_SCORES_PATH = './model_scores.npy'   # 保存5个模型的得分矩阵
OUTPUT_LABELS_PATH = './true_labels.npy'    # 保存真实标签