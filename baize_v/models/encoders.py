from types import SimpleNamespace
import numpy as np
import torch


class SenseVoiceAudioProcessor:
    def __init__(self, frontend): self.frontend = frontend
    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(input_features=fbank, attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0))


