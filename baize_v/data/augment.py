import random
import numpy as np
from scipy.signal import resample


def augment_wav(wav, sr=16000):
    # 随机变速(0.7~1.6x)：改变音频时长和音调，覆盖快/慢语速
    if random.random() < 0.5:
        speed = random.uniform(0.7, 1.6)
        wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
    # 随机加噪：叠加轻微高斯白噪声，模拟录音环境差异
    if random.random() < 0.3:
        noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
        wav = wav + noise
    # 随机音量：缩放振幅0.8~1.2倍，模拟说话音量变化
    if random.random() < 0.3:
        wav = wav * random.uniform(0.8, 1.2)
    # 随机时间遮蔽：将0.25秒片段置零，模拟短暂静音/丢包
    if random.random() < 0.2 and len(wav) > sr:
        start = random.randint(0, len(wav) - sr // 4)
        wav[start:start + sr // 4] = 0
    # 随机低通滤波：移动平均模糊高频，模拟电话/低质量麦克风
    if random.random() < 0.2:
        k = random.choice([3, 5, 7])
        wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
    # 随机混响：合成指数衰减脉冲响应卷积，模拟房间反射/回声
    if random.random() < 0.3:
        ir_len = int(sr * random.uniform(0.05, 0.2))
        ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
        ir[0] = 1.0
        ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
        wav = np.convolve(wav, ir, mode='same').astype(np.float32)
    # 随机粉红噪声：1/f噪声模拟房间环境底噪（空调/远处人声）
    if random.random() < 0.2:
        pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
        pink /= np.max(np.abs(pink)) + 1e-6
        wav = wav + pink * random.uniform(0.003, 0.015)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

def augment_mel(fbank):
    # fbank: (T, 560) — SenseVoice LFR 后的特征（时间维在前，频率维在后）
    T, D = fbank.shape
    # SpecAugment频率遮蔽：随机抹掉1~64维，防止对特定频段过拟合
    if random.random() < 0.5:
        f = random.randint(1, 64)
        f0 = random.randint(0, D - f)
        fbank[:, f0:f0 + f] = 0
    # SpecAugment时间遮蔽：随机抹掉1~min(10,T)帧，提升对不完整输入的容错
    if random.random() < 0.5 and T > 1:
        t = random.randint(1, min(10, T))
        t0 = random.randint(0, T - t)
        fbank[t0:t0 + t, :] = 0
    return fbank
