# 3_ae_models.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.ndimage import zoom

# ---------- 模型1：MFCC + 全连接自编码器 ----------
def build_mfcc_ae(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_mfcc_ae(normal_mfccs, epochs=50, batch_size=32):
    # 对每个样本的MFCC按时间帧取平均，得到固定长度向量
    X_train = np.array([np.mean(mfcc, axis=0) for mfcc in normal_mfccs])
    model = build_mfcc_ae(X_train.shape[1])
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, X_train

def predict_mfcc_ae(model, train_features, test_mfccs):
    test_features = np.array([np.mean(mfcc, axis=0) for mfcc in test_mfccs])
    recon = model.predict(test_features, verbose=0)
    mse = np.mean((test_features - recon) ** 2, axis=1)
    # 用训练集最大误差归一化
    train_recon = model.predict(train_features, verbose=0)
    train_mse = np.mean((train_features - train_recon) ** 2, axis=1)
    max_train = train_mse.max() if train_mse.max() > 0 else 1e-6
    scores = np.clip(mse / max_train, 0, 1)
    return scores

# ---------- 模型2：Mel谱图 + 卷积自编码器 ----------
def build_conv_ae(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(4, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    # Decoder
    x = layers.Conv2D(4, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    outputs = layers.Conv2D(1, (3,3), activation='linear', padding='same')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_mel_ae(normal_mels, target_size=(128,128), epochs=30, batch_size=16):
    # 缩放所有Mel谱图到固定尺寸（支持任意原始T）
    X_train = []
    for mel in normal_mels:
        mel_t = mel.T  # (n_mels, T)
        # 计算缩放因子
        zoom_factors = (target_size[0] / mel_t.shape[0], target_size[1] / mel_t.shape[1])
        resized = zoom(mel_t, zoom_factors, order=1)
        X_train.append(resized)
    X_train = np.array(X_train)[..., np.newaxis]  # (n, 128, 128, 1)
    # 标准化
    mean = X_train.mean()
    std = X_train.std() + 1e-6
    X_train = (X_train - mean) / std
    model = build_conv_ae((target_size[0], target_size[1], 1))
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, mean, std

def predict_mel_ae(model, mean, std, test_mels, target_size=(128,128)):
    X_test = []
    for mel in test_mels:
        mel_t = mel.T
        zoom_factors = (target_size[0] / mel_t.shape[0], target_size[1] / mel_t.shape[1])
        resized = zoom(mel_t, zoom_factors, order=1)
        X_test.append(resized)
    X_test = np.array(X_test)[..., np.newaxis]
    X_test = (X_test - mean) / std
    recon = model.predict(X_test, verbose=0)
    mse = np.mean((X_test - recon) ** 2, axis=(1,2,3))
    # 简单min-max归一化
    scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-6)
    scores = np.clip(scores, 0, 1)
    return scores