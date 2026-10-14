import os
import io
import json
import random

import torch
import numpy as np
import soundfile as sf
import librosa
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .chat import pre_processing_chat, post_processing_chat
from .augment import augment_wav, augment_mel
from .labels import generate_text_labels, apply_scheduled_sampling
from ..tokens import AUDIO_STOP_TOKEN, AUDIO_PAD_TOKEN, AUDIO_SPK_TOKEN, AUDIO_VOCAB_SIZE, NUM_AUDIO_LAYERS

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OmniDataset(Dataset):
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_stop_token=AUDIO_STOP_TOKEN,
                 audio_pad_token=AUDIO_PAD_TOKEN,
                 audio_spk_token=AUDIO_SPK_TOKEN,
                 audio_vocab_size=AUDIO_VOCAB_SIZE,
                 scheduled_sampling=0.05,
                 image_token_len=64):
        super().__init__()
        tables = [pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches()) for p in data_path.split(',')]
        tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in t.schema])) for t in tables]
        self.table = pa.concat_tables(tables, promote_options='default')
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        """加载音频并预处理成fbank，返回 (fbank (T,560), valid_len=encoder输出帧数)"""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def load_audio_inputs(self, audio_bytes):
        if not audio_bytes: return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return augment_mel(inputs.input_features.squeeze(0)), valid_len

    def load_image_inputs(self, image_bytes):
        if not image_bytes or self.vision_processor is None: return None
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt")
        if hasattr(inputs, 'keys'): return {k: v for k, v in inputs.items()}
        return inputs.pixel_values

    def create_chat_prompt(self, conversations, audio_features_length=0):
        conversations = pre_processing_chat(conversations)
        messages = []
        is_last_user = lambda i: i == max(j for j, t in enumerate(conversations) if t['role'] == 'user')
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4: content = ap
                elif r < 0.6: content = content
                elif r < 0.8: content = ap + '\n\n' + content
                else: content = content + '\n\n' + ap
            if '<image>' in content:
                r = random.random()
                if r < 0.2: content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4: content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6: content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else: content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    
    def __getitem__(self, index: int):
        cols = self._read_row(index)
        conversations = self._truncate_turns(cols['conversations'])
        n_user = sum(1 for t in conversations if t['role'] == 'user')
        n_asst = sum(1 for t in conversations if t['role'] == 'assistant')

        pixel_values = self._load_pixel_values(cols['image_bytes'])
        audio_inputs, audio_len, audio_features_length = self._load_question_audio(cols['question_audios'], n_user)
        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)   # 无音频样本：dummy tensor，保持 batch 对齐（SenseVoice: T x 560）
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}

        last_audio_codes = self._answer_codes(cols['answer_audios'], n_asst)

        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None: prompt = prompt.replace('<image>', self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        text_labels, assistant_ranges = generate_text_labels(input_ids, self.bos_id, self.eos_id, self.max_length)
        for start, end in assistant_ranges[:-1]:
            mask_end = min(end + len(self.eos_id), self.max_length)
            text_labels[start:mask_end] = [-100] * (mask_end - start)

        Y_audio_layers, audio_labels = self._build_audio_targets(input_ids, assistant_ranges, last_audio_codes,
                                                                 cols['ref_audios'], bool(cols['spk_emb']))

        # 9 路输入 = 8 路音频 + 1 路文本，去掉最后一个位置作为输入，标签整体左移一位
        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)
        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)

        input_ids = apply_scheduled_sampling(input_ids, audio_labels, text_labels, self.scheduled_sampling_prob,
                                             self.audio_vocab_size, self.text_vocab_size, self.image_token_id)
        spk_emb = torch.tensor(cols['spk_emb'], dtype=torch.float32) if cols['spk_emb'] else torch.zeros(192)
        return input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb

    def _read_row(self, index):
        """读取一行的各列；缺失的列返回空值。不涉及随机数。"""
        names = self.table.column_names

        def col(name, default):
            return self.table[name][index].as_py() if name in names else default

        image_bytes = col('image_bytes', [])
        if image_bytes and not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        return {
            'conversations': json.loads(self.table['conversations'][index].as_py()),
            'question_audios': col('question_audios', []),
            'answer_audios': col('answer_audios', []),
            'image_bytes': image_bytes,
            'ref_audios': col('ref_audios', []),
            'spk_emb': col('spk_emb', []),
        }

    def _truncate_turns(self, conversations):
        """随机截断到某一轮（每轮 = user + assistant），并回退直到长度安全。"""
        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if len(asst_indices) <= 1:
            return conversations
        rand_idx = random.randint(0, len(asst_indices) - 1)
        for i in range(rand_idx, -1, -1):
            conversations = conversations[:asst_indices[i] + 1]
            test_prompt = self.create_chat_prompt(conversations, 0)
            if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                break
        return conversations

    def _load_pixel_values(self, image_bytes):
        """只取第一张图；没有图像或没有视觉处理器时返回 None。"""
        if image_bytes and len(image_bytes) > 0 and self.vision_processor:
            return self.load_image_inputs(image_bytes[0])
        return None

    def _load_question_audio(self, question_audios, n_user):
        """只加载最后一个 user 轮次的音频。返回 (audio_inputs, audio_len, 占位长度)。"""
        if question_audios and n_user > 0 and n_user <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[n_user - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    return mel.unsqueeze(0), valid_len, valid_len or 1
        return None, 0, 0

    def _answer_codes(self, answer_audios, n_asst):
        """取最后一个 assistant 轮次的音频码，整理为 8 层列表并追加 stop token。"""
        if not (answer_audios and n_asst > 0 and n_asst <= len(answer_audios)):
            return None
        tokens = answer_audios[n_asst - 1]
        if not tokens:
            return None
        layers = [[] for _ in range(NUM_AUDIO_LAYERS)]
        for i in range(0, len(tokens) - 7, 8):
            for j in range(NUM_AUDIO_LAYERS): layers[j].append(tokens[i + j])
        for layer in layers: layer.append(self.audio_stop_token)
        return layers

    def _build_audio_targets(self, input_ids, assistant_ranges, last_audio_codes, ref_audios, has_spk):
        """填充 8 层音频目标：[说话人占位][参考音频 codes][目标 codes]，只有最后一个 assistant 参与 loss。"""
        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(NUM_AUDIO_LAYERS)]
        audio_labels = [[-100] * self.max_length for _ in range(NUM_AUDIO_LAYERS)]
        if not (assistant_ranges and last_audio_codes):
            return Y_audio_layers, audio_labels

        assistant_start, assistant_end = assistant_ranges[-1]
        for pos in range(assistant_start, min(assistant_end, assistant_start + 50)):
            if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                assistant_start = pos + len(self.think_end_ids)
                break

        # 50% 概率丢弃参考音频，只保留说话人向量
        has_ref = bool(ref_audios) and random.random() > 0.5
        spk_reserve = 1 if has_spk else 0
        if has_ref:
            ref_codes = [[] for _ in range(NUM_AUDIO_LAYERS)]
            for i in range(0, len(ref_audios) - 7, 8):
                for j in range(NUM_AUDIO_LAYERS): ref_codes[j].append(ref_audios[i + j])
            ref_len = len(ref_codes[0])
            ref_start = max(spk_reserve, assistant_start - ref_len)
            for layer_idx in range(NUM_AUDIO_LAYERS):
                codes = ref_codes[layer_idx][-(assistant_start - ref_start):] if ref_len > (assistant_start - ref_start) else ref_codes[layer_idx]
                for i, code in enumerate(codes):
                    Y_audio_layers[layer_idx][ref_start + i] = code
        else:
            ref_start = assistant_start
        if has_spk and ref_start > 0:
            for layer_idx in range(NUM_AUDIO_LAYERS):
                Y_audio_layers[layer_idx][ref_start - 1] = self.audio_spk_token

        for layer_idx in range(NUM_AUDIO_LAYERS):
            start_pos = assistant_start + layer_idx + 1
            for i, code in enumerate(last_audio_codes[layer_idx]):
                if start_pos + i < self.max_length:
                    Y_audio_layers[layer_idx][start_pos + i] = code
                    audio_labels[layer_idx][start_pos + i] = code
        return Y_audio_layers, audio_labels
