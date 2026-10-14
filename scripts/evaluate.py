"""Baize-V 离线评测：文本 / 多轮 / 语音 / 音色克隆 / 图像 / 混合输入。

用法（仓库根目录）：uv run python scripts/evaluate.py --load_from torch --weight sft_omni --mode -1
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel

from baize_v.config import OmniConfig
from baize_v.models import BaizeOmni
from baize_v.data import OmniDataset
from baize_v.inference.mimi import frames_to_codes, mask_special, decode_codes
from baize_v.models.summary import log_model_params
from baize_v.paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR, MIMI_DIR, OUT_DIR, EVAL_AUDIO_DIR, SPEAKER_DIR
from baize_v.train.dist import setup_seed

warnings.filterwarnings('ignore')

AUDIO_EXTS = ('.mp3', '.wav')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def init_model(args):
    is_torch = args.load_from == 'torch'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR if is_torch else args.load_from)
    if is_torch:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
        model = BaizeOmni(
            OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR,
        )
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.audio_encoder, model.audio_processor = BaizeOmni.load_sensevoice(SENSEVOICE_DIR)
        model.vision_encoder, model.vision_processor = BaizeOmni.load_vision(SIGLIP_DIR)
    log_model_params(model)
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    model.mimi_model = MimiModel.from_pretrained(MIMI_DIR).eval()
    return model.half().eval().to(args.device), tokenizer


def eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, output_name, pixel_values=None, history=None, audio_lens=None, ref_codes=None, spk_emb=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
    x = torch.tensor(tokenizer(inputs_text).data['input_ids'], dtype=torch.long, device=args.device)[None, ...]

    audio_frames = []
    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p, stream=True,
                               return_audio_codes=True, open_thinking=bool(args.open_thinking),
                               audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values,
                               ref_codes=ref_codes, spk_emb=spk_emb)
        print('📒 [Thinker]: ', end='', flush=True)
        history_idx = 0
        for y, audio_frame in res_y:
            if y is not None:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if answer and answer[-1] != '�':
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()

        if audio_frames:
            print(f'🎹 [Talker]: {len(audio_frames)} frames', end=" ")
            if args.decode_audio:
                try:
                    codes = frames_to_codes(audio_frames, device=args.device)
                    if codes is None:
                        print('⚠️  生成的Mimi codes为空，跳过保存。')
                        return
                    audio = decode_codes(model.mimi_model, mask_special(codes)).astype(np.float32)
                    output_path = os.path.join(args.output_dir, output_name)
                    wav_path = output_path.rsplit('.', 1)[0] + '.wav'
                    sf.write(wav_path, audio, 24000)
                    AudioSegment.from_wav(wav_path).export(output_path, format='mp3', bitrate='64k')
                    os.remove(wav_path)
                    print(f'| Audio decoded to: {output_path}')
                except Exception as e:
                    print(f'⚠️  保存音频失败: {str(e)}')
            else:
                print("(decode_audio=off)\n")


def _list_files(directory, prefix=None, exts=None):
    """按前缀和扩展名列出文件（排序后返回）。"""
    names = []
    for f in os.listdir(directory):
        if prefix is not None and not f.startswith(prefix):
            continue
        if exts is not None and not f.lower().endswith(exts):
            continue
        names.append(f)
    return sorted(names)


def _audio_inputs(model, audio_path, device):
    """读取音频并返回 (audio_inputs, audio_lens, 占位 token 数)。"""
    mel, valid_len = OmniDataset.process_audio(audio_path, model.audio_processor)
    audio_inputs = mel.unsqueeze(0).to(device)
    audio_lens = torch.tensor([valid_len], device=device)
    return audio_inputs, audio_lens, valid_len or 1


def _pixel_values(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    return {k: v.to(device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}


def run_text(model, tokenizer, args):
    print('\n\n==================== text -> {text, audio} ====================')
    test_prompts_en = [
        "Tell me an interesting fact about space.", "How do I make a cup of coffee?", "What's the weather like today?",
        "Will it rain tomorrow?", "Tell me a joke.", "Can you sing a song for me?", "Please introduce yourself."
    ]
    test_prompts_zh = [
        "告诉我一个关于太空的有趣事实。", "如何制作一杯咖啡？", "今天的天气怎么样？",
        "明天会下雨吗？", "给我讲个笑话吧", "你能为我唱首歌吗？", "介绍一下你自己"
    ]
    test_prompts = [test_prompts_en, test_prompts_zh, test_prompts_en + test_prompts_zh][args.prompt_lang]
    for idx, prompt in enumerate(test_prompts):
        print(f'\n📝 [text-{idx+1}]: {prompt}')
        eval_sample(model, tokenizer, args, idx, prompt, None, f"text-{idx:02d}.mp3")


MULTI_TURN_ZH = [
    {"history": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}],
     "prompt": "我想找点事做，你有什么建议吗？"},
    {"history": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                 {"role": "user", "content": "我想找点事做，你有什么建议吗？"}, {"role": "assistant", "content": "可以听听音乐或者看看书，放松一下心情。"}],
     "prompt": "好的，那我去照做了，谢谢你"},
]
MULTI_TURN_EN = [
    {"history": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help you?"}],
     "prompt": "I want to find something to do. Do you have any suggestions?"},
    {"history": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help you?"},
                 {"role": "user", "content": "I want to find something to do. Do you have any suggestions?"},
                 {"role": "assistant", "content": "You can listen to music or read a book to relax a little."}],
     "prompt": "Okay, I will try that. Thank you."},
]


def run_multi(model, tokenizer, args):
    print('\n\n==================== multi-turn -> {text, audio} ====================')
    tests = [MULTI_TURN_EN, MULTI_TURN_ZH, MULTI_TURN_EN + MULTI_TURN_ZH][args.prompt_lang]
    for idx, test in enumerate(tests):
        print(f'\n💬 [multi-{idx+1}]')
        for msg in test["history"]: print(f'   {msg["role"]}: {msg["content"]}')
        print(f'   user: {test["prompt"]}')
        eval_sample(model, tokenizer, args, idx, test["prompt"], None, f"multi-{idx:02d}.mp3", history=test["history"])


def run_audio(model, tokenizer, args):
    print('\n\n==================== audio -> {text, audio} ====================')
    audio_en = _list_files(args.audio_dir, prefix='audio-en-', exts=AUDIO_EXTS)
    audio_zh = _list_files(args.audio_dir, prefix='audio-zh-', exts=AUDIO_EXTS)
    audio_files = [audio_en, audio_zh, audio_en + audio_zh][args.prompt_lang]
    for idx, audio_file in enumerate(audio_files):
        print(f'\n🎤 [audio-{idx+1}]: {audio_file}')
        audio_inputs, audio_lens, token_len = _audio_inputs(model, os.path.join(args.audio_dir, audio_file), args.device)
        prompt = model.config.audio_special_token * token_len
        eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"audio-{idx:02d}-{os.path.splitext(audio_file)[0]}.mp3", audio_lens=audio_lens)


def run_clone(model, tokenizer, args):
    print('\n\n==================== clone voice -> {text, audio} ====================')
    clone_en = ["Hello, please introduce yourself.", "What's the weather like today?", "Tell me a joke."]
    clone_zh = ["你好，请介绍一下你自己。", "今天天气怎么样？", "给我讲个笑话吧。"]
    clone_prompts = [clone_en, clone_zh, clone_en + clone_zh][args.prompt_lang]
    voices_pt = os.path.join(SPEAKER_DIR, 'voices_unseen.pt')
    voices = [('default', None, None)]
    if os.path.exists(voices_pt):
        voice_data = torch.load(voices_pt, map_location=args.device)
        for speaker, v in sorted(voice_data.items()):
            rc = v['ref_codes'].unsqueeze(0).to(args.device)
            se = v['spk_emb'].half().unsqueeze(0).to(args.device) if 'spk_emb' in v else None
            voices.append((speaker, rc, se))
    for speaker, rc, se in voices:
        info = f'ref_codes: {rc.shape[2]} frames, spk_emb: {"+" if se is not None else "-"}' if rc is not None else ('spk_emb only' if se is not None else 'default')
        print(f'\n🎵 [clone: {speaker}] {info}')
        for idx, prompt in enumerate(clone_prompts):
            print(f'  📝 [text-{idx+1}]: {prompt}')
            history = [{"role": "system", "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。请尽量详细地回答，给出有价值的信息。"}]
            eval_sample(model, tokenizer, args, idx, prompt, None, f"clone-{speaker}-{idx:02d}.mp3", ref_codes=rc, history=history, spk_emb=se)


def run_image(model, tokenizer, args):
    print('\n\n==================== image -> {text, audio} ====================')
    image_files = _list_files(args.image_dir, exts=IMAGE_EXTS)
    for idx, image_file in enumerate(image_files):
        print(f'\n🖼️ [image-{idx+1}]: {image_file}')
        pixel_values = _pixel_values(model, os.path.join(args.image_dir, image_file), args.device)
        prompts = [["Please describe this image."], ["请描述这张图片"], ["Please describe this image.", "请描述这张图片"]][args.prompt_lang]
        for lang_idx, prompt_text in enumerate(prompts):
            prompt = prompt_text + "\n\n" + model.config.image_special_token * model.config.image_token_len
            eval_sample(model, tokenizer, args, idx, prompt, None, f"image-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values)


def run_mix(model, tokenizer, args):
    print('\n\n==================== text+audio+image -> {text, audio} ====================')
    img_audio_files = _list_files(args.audio_dir, prefix='img-', exts=AUDIO_EXTS)
    image_files = _list_files(args.image_dir, exts=IMAGE_EXTS)
    text_hints = [["Please answer me: "], ["请回答我："], ["Please answer me: ", "请回答我："]][args.prompt_lang]
    for idx, image_file in enumerate(image_files):
        audio_file = random.choice(img_audio_files)
        pixel_values = _pixel_values(model, os.path.join(args.image_dir, image_file), args.device)
        for lang_idx, text_hint in enumerate(text_hints):
            print(f'\n🌀 [mix-{idx+1}-{lang_idx}]: {text_hint} | {audio_file} | {image_file}')
            audio_inputs, audio_lens, token_len = _audio_inputs(model, os.path.join(args.audio_dir, audio_file), args.device)
            prompt = text_hint + model.config.audio_special_token * token_len + "\n\n" + model.config.image_special_token * model.config.image_token_len
            eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"mix-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3",
                        pixel_values=pixel_values, audio_lens=audio_lens)


RUNNERS = {'0': run_text, '1': run_multi, '2': run_audio, '3': run_clone, '4': run_image, '5': run_mix}


def parse_args():
    parser = argparse.ArgumentParser(description="Baize-V Chat")
    parser.add_argument('--load_from', default='torch', type=str, help="模型加载路径（torch=原生torch权重，否则为 HF 目录）")
    parser.add_argument('--save_dir', default=OUT_DIR, type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_omni', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="Thinker生成温度")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument('--output_dir', default='./output_audio/', type=str, help="输出音频保存目录")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--audio_dir', default=EVAL_AUDIO_DIR, type=str, help="测试音频目录")
    parser.add_argument('--image_dir', default=EVAL_AUDIO_DIR, type=str, help="测试图像目录")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启思考模式（0=否，1=是）（思考模式下禁用audio输出）")
    parser.add_argument('--decode_audio', default=1, type=int, help="是否解码音频输出（0=否，1=是）")
    parser.add_argument('--mode', default='0', type=str, help="评估模式：-1=all 0=text 1=multi 2=audio 3=clone 4=image 5=mix（逗号组合，如 2,5）")
    parser.add_argument('--prompt_lang', default=0, type=int, choices=[0, 1, 2], help="问题语言：0=英文 1=中文 2=英文+中文")
    return parser.parse_args()


def main():
    args = parse_args()
    modes = set(args.mode.replace(',', '').replace('-1', '012345'))

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = init_model(args)
    setup_seed(int(time.time()) % 31415926)

    for key, runner in RUNNERS.items():
        if key in modes:
            runner(model, tokenizer, args)


if __name__ == "__main__":
    main()
