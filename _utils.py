# 2_audio_utils.py
import numpy as np
import librosa

def load_audio(file_path, sr=16000):
    """加载音频，不固定长度，返回原始波形"""
    y, _ = librosa.load(file_path, sr=sr)
    return y

def extract_mfcc(waveform, sr, n_mfcc=13, hop_length=512, n_fft=1024):
    """返回 (T, n_mfcc)，T 随音频长度变化"""
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=hop_length, n_fft=n_fft)
    return mfccs.T  # (T, n_mfcc)

def extract_mel_spectrogram(waveform, sr, n_mels=128, hop_length=512, n_fft=1024):
    """返回 (T, n_mels)，T 随音频长度变化"""
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels,
                                         hop_length=hop_length, n_fft=n_fft)
    log_mel = librosa.power_to_db(mel)
    return log_mel.T  # (T, n_mels)


def add_noise(waveform, snr_db=30):
    """添加高斯白噪声（信噪比30dB，轻微噪声）"""
    noise = np.random.randn(len(waveform))
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)
    # 计算噪声缩放系数
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    noisy_waveform = waveform + scale * noise
    return noisy_waveform

def time_stretch(waveform, rate=0.9):
    """时间拉伸（0.9倍速变慢，1.1倍速变快），改变长度"""
    # librosa 的时间拉伸会改变长度，需要重新resize到原始长度
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    # 截断或补零到原始长度（训练时固定长度可保持batch一致）
    if len(stretched) > len(waveform):
        stretched = stretched[:len(waveform)]
    else:
        stretched = np.pad(stretched, (0, len(waveform) - len(stretched)), mode='constant')
    return stretched

def pitch_shift(waveform, sr, n_steps=2):
    """音高偏移（+2 或 -2 个半音）"""
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

def augment_waveform(waveform, sr):
    """
    对单个音频生成 3 个增强版本
    返回一个列表，包含原始波形 + 3个增强波形
    """
    augmented = [waveform]  # 保留原始版本
    
    # 增强1：加轻微噪声
    augmented.append(add_noise(waveform, snr_db=25))
    
    # 增强2：时间拉伸变慢一点（模拟升降阻力变化）
    augmented.append(time_stretch(waveform, rate=0.9))
    
    # 增强3：音高微调（模拟不同负载下的声音频率变化）
    augmented.append(pitch_shift(waveform, sr, n_steps=2))
    
    return augmented