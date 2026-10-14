import argparse, os, json, time, math, torch, threading, queue, base64, io, logging, contextlib
import numpy as np
import torchaudio
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image
from pydub import AudioSegment
from baize_v.inference.realtime import RealtimeSession
from baize_v.inference.loader import scan_hf_models, load_hf_model, clear_remote_code_cache
from baize_v.inference.mimi import frames_to_codes, mask_special, decode_codes
from baize_v.models.encoders import load_sensevoice, load_vision
from baize_v.models.summary import log_model_params
from voices import VoiceRegistry
from baize_v.paths import SENSEVOICE_DIR, SIGLIP_DIR, MIMI_DIR, CAMPPLUS_DIR, SPEAKER_DIR, VAD_PATH
from baize_v.tokens import AUDIO_PAD_TOKEN
logging.getLogger().setLevel(logging.ERROR)
with contextlib.redirect_stdout(io.StringIO()):
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)
sock = Sock(app)
M = {}  # model / tokenizer / device / mimi / asr / cfg
VOICES = VoiceRegistry()
MODEL_LOCK = threading.Lock()
SAMPLES_PER_FRAME = 1920
REF_FRAMES = 300
CLONE_VOICE = 'voice_clone'
CLONE_FILE = 'voice_clone.pt'

# -------- helpers --------
def sse(d): return f"data: {json.dumps(d)}\n\n"


def asr_run(samples):
    r = M['asr'].generate(input=samples, cache={}, language='auto', use_itn=True)
    return rich_transcription_postprocess(r[0]['text']).strip() if r else ''

def prep_audio(samples):
    m = M['model']
    proc = m.audio_processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    mel = proc.input_features.squeeze(0).unsqueeze(0).to(M['device'])
    vlen = proc.attention_mask.sum().item()
    prompt = m.config.audio_special_token * (vlen or 1)
    return mel, torch.tensor([vlen], device=M['device']), prompt

def prep_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return {k: v.to(M['device']) for k, v in M['model'].vision_processor(images=img, return_tensors="pt").items()}

def build_ids(prompt, history):
    tok, dev, n = M['tokenizer'], M['device'], M['cfg'].max_history_turns
    hist = history[-n:] if n > 0 else []
    msgs = hist + [{"role": "user", "content": prompt}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor(tok(t).data['input_ids'], dtype=torch.long, device=dev)[None, ...]

def _mimi_decode(frames):
    mc = frames_to_codes(frames, M['device'])
    if mc is None or not M['mimi']: return None
    mc = mask_special(mc)
    return decode_codes(M['mimi'], mc), mc.shape[-1]

def pcm_bytes(frames, ov):
    r = _mimi_decode(frames)
    if r is None: return None
    au, T = r
    if ov > 0: au = au[int(ov * len(au) / T):]
    return (au * 32767).astype('int16').tobytes()

def stream_pcm(frames, flush=False):
    """yield (pcm_bytes,) on chunk boundaries or on final flush."""
    if not M['mimi']: return
    cf, ov_max, n = M['cfg'].audio_chunk_frames, M['cfg'].audio_overlap, len(frames)
    if not flush and n >= cf and n % cf == 0:
        ov = min(ov_max, n - cf)
        p = pcm_bytes(frames[-(cf + ov):], ov)
        if p: yield p
    elif flush:
        rem = n % cf
        if rem:
            ov = min(ov_max, n - rem)
            p = pcm_bytes(frames[-(rem + ov):], ov)
            if p: yield p

def voice_args(name):
    if name and name != 'default' and name in VOICES:
        v = VOICES[name]
        dev = M['device']
        rc = v['ref_codes'].unsqueeze(0).to(dev)
        se = v['spk_emb'].half().unsqueeze(0).to(dev) if 'spk_emb' in v else None
        return {'ref_codes': rc, 'spk_emb': se}
    return {}

def register_voice(name, value, group='manual'):
    VOICES.register(name, value, group)

def clone_voice_path():
    return os.path.join(SPEAKER_DIR, CLONE_FILE)

def delete_manual_voice(name):
    if not VOICES.in_group(name, 'manual'):
        raise RuntimeError('只能删除手动克隆的音色')
    out_path = clone_voice_path()
    saved = torch.load(out_path, map_location='cpu') if os.path.exists(out_path) else {}
    if name in saved:
        saved.pop(name)
        torch.save(saved, out_path)
    VOICES.remove_manual(name)

def normalize_voice_name(name):
    name = ' '.join(str(name or '').split())
    if not name:
        name = CLONE_VOICE
    if len(name) > 24:
        raise RuntimeError('音色名太长，建议控制在 24 个字以内')
    if name.lower() == 'default':
        raise RuntimeError('default 是保留名称，请换一个')
    if VOICES.in_group(name, 'builtin') or VOICES.in_group(name, 'unseen'):
        raise RuntimeError('该名称已被现有音色占用，请换一个')
    return name

def validate_clone_audio(w16):
    if w16.numel() < int(16000 * 1.8):
        raise RuntimeError('录音太短，请把整句话读完')
    peak = w16.abs().max().item()
    frame, hop = 800, 400
    if w16.numel() >= frame:
        rms = w16.unfold(0, frame, hop).pow(2).mean(dim=1).sqrt().cpu().numpy()
    else:
        rms = np.array([w16.pow(2).mean().sqrt().item()])
    hi = float(np.quantile(rms, 0.95))
    lo = float(np.quantile(rms, 0.2))
    if hi < 0.008:
        raise RuntimeError('录音太轻，请靠近麦克风一点')
    if hi > 0 and lo / hi > 0.45:
        raise RuntimeError('环境噪声太大，请换安静一点的环境')
    if peak > 0.995:
        raise RuntimeError('录音有爆音，请离麦克风远一点')

def build_clone_voice(audio_b64):
    if M.get('mimi') is None or M.get('campplus') is None or M.get('mel_fn') is None:
        raise RuntimeError('Mimi 或 CAM++ 未加载')
    seg = AudioSegment.from_file(io.BytesIO(base64.b64decode(audio_b64))).set_channels(1).set_sample_width(2)
    if len(seg) < 1000:
        raise RuntimeError('录音太短，至少读 1 秒')
    try:
        seg = seg.speedup(playback_speed=1.5, chunk_size=150, crossfade=25)
    except Exception:
        seg = seg.speedup(playback_speed=1.5)
    seg24 = seg.set_frame_rate(24000)
    seg16 = seg.set_frame_rate(16000)
    w24 = torch.tensor(np.frombuffer(seg24.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)
    w16 = torch.tensor(np.frombuffer(seg16.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)
    validate_clone_audio(w16)
    mimi_dev = next(M['mimi'].parameters()).device
    mimi_dtype = torch.float16 if mimi_dev.type != 'cpu' else torch.float32
    with torch.inference_mode():
        t = w24.unsqueeze(0).unsqueeze(0).to(device=mimi_dev, dtype=mimi_dtype)
        codes = M['mimi'].encode(t).audio_codes
        nf = math.ceil(w24.shape[-1] / SAMPLES_PER_FRAME)
        ref_codes = codes[0, :8, :nf].cpu()[:, :min(nf, REF_FRAMES)]
    with torch.no_grad():
        mel = M['mel_fn'](w16.unsqueeze(0).to(M['device']))
        feat = mel.clamp(min=1e-10).log().transpose(1, 2)
        feat = feat - feat.mean(dim=1, keepdim=True)
        spk_emb = M['campplus'](feat).squeeze(0).cpu()
    return {'ref_codes': ref_codes, 'spk_emb': spk_emb}

def run_generate(x, audio_inputs, audio_lens, pixel_values, **kw):
    with MODEL_LOCK, torch.no_grad():
        yield from M['model'].generate(
            x, M['tokenizer'].eos_token_id, stream=True, return_audio_codes=True,
            audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, **kw)

def load_main_model(model_path, model_name):
    with MODEL_LOCK:
        clear_remote_code_cache()
        M.pop('model', None); M.pop('tokenizer', None)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        tok, m = load_hf_model(model_path, M['device'], audio=load_sensevoice(SENSEVOICE_DIR), vision=load_vision(SIGLIP_DIR))
        M['tokenizer'], M['model'], M['model_name'] = tok, m, model_name
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f'Loaded model: {model_name} ({params:.2f}M)')
        return round(params, 2)


def prepare_turn(text, samples, image_b64, do_asr_for_image):
    """返回 (audio_inputs, audio_lens, pixel_values, prompt_for_model, user_text_for_history, asr_thread, asr_result)"""
    audio_inputs = audio_lens = pixel_values = None
    prompt = text or ''
    user_text = text or ''
    asr_thread, asr_result = None, [None]
    if samples is not None:
        if image_b64 and do_asr_for_image:
            user_text = asr_run(samples)
            prompt = user_text
        else:
            audio_inputs, audio_lens, prompt = prep_audio(samples)
            if M['cfg'].max_history_turns > 0:
                sa = samples.copy()
                def _a(): asr_result[0] = asr_run(sa)
                asr_thread = threading.Thread(target=_a); asr_thread.start()
    if image_b64:
        pixel_values = prep_image(image_b64)
        m = M['model']
        prompt = (prompt + "\n\n" if prompt else "") + "请描述这张图片\n\n" + m.config.image_special_token * m.config.image_token_len
    return audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_thread, asr_result

# -------- routes --------
@app.route('/')
def index(): return send_from_directory(STATIC_DIR, 'web_demo.html')
@app.route('/call')
def call_page(): return send_from_directory(STATIC_DIR, 'web_demo.html')

@app.route('/voices')
def get_voices():
    return json.dumps(VOICES.summary())

@app.route('/models')
def get_models():
    return json.dumps({'models': list(M.get('models', {}).keys()), 'current': M.get('model_name')})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    name = (request.json or {}).get('name')
    if name not in M.get('models', {}):
        return Response(json.dumps({'ok': False, 'error': 'unknown model'}), status=400, mimetype='application/json')
    try:
        params = load_main_model(M['models'][name], name)
        return Response(json.dumps({'ok': True, 'model': name, 'params': params}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    d = request.json or {}
    if not d.get('audio'):
        return Response(json.dumps({'ok': False, 'error': 'missing audio'}), status=400, mimetype='application/json')
    try:
        name = normalize_voice_name(d.get('name'))
        value = build_clone_voice(d['audio'])
        out_path = clone_voice_path()
        saved = torch.load(out_path, map_location='cpu') if os.path.exists(out_path) else {}
        saved[name] = value
        torch.save(saved, out_path)
        register_voice(name, value, group='manual')
        return Response(json.dumps({'ok': True, 'voice': name, 'path': 'assets/speaker/' + CLONE_FILE}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/delete_voice', methods=['POST'])
def delete_voice():
    d = request.json or {}
    name = ' '.join(str(d.get('name') or '').split())
    if not name:
        return Response(json.dumps({'ok': False, 'error': 'missing name'}), status=400, mimetype='application/json')
    try:
        delete_manual_voice(name)
        return Response(json.dumps({'ok': True, 'voice': name}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'ok': False, 'error': str(e)}), status=500, mimetype='application/json')

@app.route('/chat', methods=['POST'])
def chat():
    d = request.json
    history = d.get('history', [])
    samples = None
    if d.get('audio'):
        seg = AudioSegment.from_file(io.BytesIO(base64.b64decode(d['audio']))).set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    va = voice_args(d.get('voice', 'default'))

    def gen():
        audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_th, asr_res = prepare_turn(
            d.get('text', ''), samples, d.get('image'), do_asr_for_image=True)
        x = build_ids(prompt, history)
        asr_sent = False
        if user_text and samples is not None and d.get('image'):
            yield sse({'type': 'user_prompt', 'content': user_text}); asr_sent = True
        frames, text_ttft, audio_ttft = [], None, None
        t0 = time.time(); hi = 0
        for y, af in run_generate(x, audio_inputs, audio_lens, pixel_values,
                                   max_new_tokens=d.get('max_tokens', 512),
                                   temperature=d.get('temperature', 1), top_p=0.85, **va):
            if not asr_sent and asr_th and not asr_th.is_alive():
                asr_th.join()
                if asr_res[0]: yield sse({'type': 'user_prompt', 'content': asr_res[0]})
                asr_sent = True
            if y is not None:
                if text_ttft is None:
                    text_ttft = (time.time() - t0) * 1000
                    yield sse({'type': 'ttft', 'text_ttft': round(text_ttft, 1)})
                ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                if ans and ans[-1] != '\ufffd' and len(ans) > hi:
                    yield sse({'type': 'text', 'content': ans[hi:]}); hi = len(ans)
            if af:
                if audio_ttft is None:
                    audio_ttft = (time.time() - t0) * 1000
                    yield sse({'type': 'ttft', 'audio_ttft': round(audio_ttft, 1)})
                frames.append(af)
                for pcm in stream_pcm(frames):
                    b64 = base64.b64encode(pcm).decode()
                    for i in range(0, len(b64), 2000):
                        yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
        for pcm in stream_pcm(frames, flush=True):
            b64 = base64.b64encode(pcm).decode()
            for i in range(0, len(b64), 2000):
                yield sse({'type': 'pcm', 'c': b64[i:i+2000], 'd': i+2000 >= len(b64)})
        if not asr_sent:
            if asr_th:
                asr_th.join()
                if asr_res[0]: yield sse({'type': 'user_prompt', 'content': asr_res[0]})
            else:
                yield sse({'type': 'user_prompt', 'content': prompt})
        yield sse({'type': 'done'})

    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@sock.route('/ws/realtime')
class RealtimeConnection:
    """一条 WebSocket 实时连接：接收音频，VAD 判定语音段后生成回复并流式回传。"""

    def __init__(self, ws):
        self.ws = ws
        self.session = RealtimeSession(M['vad_path'])
        self.q = queue.Queue()
        self.alive = [True]
        self.state = {'history': [], 'image': None}
        self.n_hist = M['cfg'].max_history_turns

    def _send(self, payload):
        self.ws.send(json.dumps(payload))

    def _send_vad(self):
        self._send({'type': 'vad', 'speaking': self.session.speaking})

    def _send_pcm(self, pcm):
        self._send({'type': 'pcm', 'data': base64.b64encode(pcm).decode()})

    def push_audio(self, data):
        return self.session.push_chunk(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

    def set_ctx(self, msg):
        h = msg.get('history') or []
        self.state['history'] = h[-self.n_hist:] if self.n_hist > 0 else []
        if 'image' in msg: self.state['image'] = msg.get('image')
        if 'voice' in msg: self.state['voice'] = msg.get('voice', 'default')

    def poll_interrupt(self):
        """生成过程中非阻塞地查看新消息；语音打断或停止时返回 True。"""
        while True:
            try: data = self.q.get_nowait()
            except queue.Empty: return False
            if isinstance(data, bytes):
                if self.push_audio(data) == 'interrupt': return True
                self._send_vad()
            else:
                m = json.loads(data)
                if m.get('type') == 'context': self.set_ctx(m)
                elif m.get('type') in ('stop', 'end'):
                    if m['type'] == 'end': self.alive[0] = False
                    self.session.interrupt = True
                    return True

    def recv_loop(self):
        while self.alive[0]:
            try:
                data = self.ws.receive(timeout=1)
                if data is None: self.alive[0] = False; break
                self.q.put(data)
            except: self.alive[0] = False; break

    def handle_control(self, m):
        """空闲时处理文本控制消息。返回 False 表示结束连接。"""
        if m.get('type') == 'context': self.set_ctx(m)
        elif m.get('type') == 'stop': self.session.interrupt = True
        elif m.get('type') == 'end': return False
        return True

    def respond(self):
        """对一段完整语音生成回复，流式返回文本与音频。"""
        s = self.session
        s.generating = True
        audio = s.get_audio()
        self._send({'type': 'generating'})
        audio_inputs, audio_lens, pixel_values, prompt, user_text, asr_th, asr_res = prepare_turn(
            '', audio, self.state['image'], do_asr_for_image=True)
        if self.state['image']: self.state['image'] = None
        x = build_ids(prompt, self.state['history'])
        va_rt = voice_args(self.state.get('voice', 'default'))

        frames, full_text, interrupted = [], '', False
        for y, af in run_generate(x, audio_inputs, audio_lens, pixel_values, max_new_tokens=512, temperature=0.7, **va_rt):
            if self.poll_interrupt() or s.interrupt: interrupted = True; break
            if y is not None:
                ans = M['tokenizer'].decode(y[0].tolist(), skip_special_tokens=True)
                if ans and ans[-1] != '\ufffd' and len(ans) > len(full_text):
                    self._send({'type': 'text', 'content': ans[len(full_text):]})
                    full_text = ans
            if af:
                frames.append(af)
                for pcm in stream_pcm(frames): self._send_pcm(pcm)
        if not interrupted:
            for pcm in stream_pcm(frames, flush=True): self._send_pcm(pcm)
        if asr_th:
            asr_th.join(); user_text = asr_res[0] or user_text
        self._remember(user_text, full_text)
        self._send({'type': 'done', 'interrupted': interrupted or s.interrupt})
        s.generating = False; s.interrupt = False

    def _remember(self, user_text, assistant_text):
        if self.n_hist > 0:
            if user_text: self.state['history'].append({'role': 'user', 'content': user_text})
            if assistant_text: self.state['history'].append({'role': 'assistant', 'content': assistant_text})
            self.state['history'] = self.state['history'][-self.n_hist:]

    def run(self):
        threading.Thread(target=self.recv_loop, daemon=True).start()
        try:
            while self.alive[0]:
                try: data = self.q.get(timeout=0.05)
                except queue.Empty: continue
                if isinstance(data, str):
                    if not self.handle_control(json.loads(data)): break
                    continue
                if self.session.generating:
                    self.push_audio(data); self._send_vad(); continue
                status = self.push_audio(data)
                self._send_vad()
                if status == 'speech_end': self.respond()
        finally:
            self.alive[0] = False


def realtime(ws):
    RealtimeConnection(ws).run()

def init_model(args):
    M['cfg'] = args; M['device'] = args.device
    with contextlib.redirect_stdout(io.StringIO()):
        M['asr'] = AutoModel(model=SENSEVOICE_DIR, trust_remote_code=True, device=args.device, disable_update=True)
    M['models'] = scan_hf_models(args.load_from)
    if not M['models']:
        raise RuntimeError(f"未在 {os.path.abspath(args.load_from)} 找到 transformers 模型")
    model_name = next(iter(M['models']))
    load_main_model(M['models'][model_name], model_name)
    try:
        from transformers import MimiModel
        M['mimi'] = MimiModel.from_pretrained(MIMI_DIR).eval().to(args.device)
        if args.device != 'cpu': M['mimi'] = M['mimi'].half()
        print('Mimi model loaded')
    except: M['mimi'] = None
    try:
        from modelscope.models.audio.sv.DTDNN import CAMPPlus
        M['campplus'] = CAMPPlus(feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4,
                                 init_channels=128, config_str='batchnorm-relu', memory_efficient=True)
        sd = torch.load(os.path.join(CAMPPLUS_DIR, 'campplus_cn_common.pt'), map_location='cpu')
        M['campplus'].load_state_dict({k: v.float() for k, v in sd.items()})
        M['campplus'] = M['campplus'].eval().to(args.device)
        M['mel_fn'] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            n_mels=80, f_min=20, f_max=7600, norm='slaney', mel_scale='slaney',
        ).to(args.device)
        print('CAM++ loaded')
    except Exception as e:
        M['campplus'], M['mel_fn'] = None, None
        print(f'CAM++ load failed: {e}')
    M['vad_path'] = VAD_PATH
    spk_dir = SPEAKER_DIR
    for fn, group in [('voices.pt', 'builtin'), ('voices_unseen.pt', 'unseen'), (CLONE_FILE, 'manual')]:
        fp = os.path.join(spk_dir, fn)
        if os.path.exists(fp):
            for speaker, v in torch.load(fp, map_location=args.device).items():
                if speaker not in VOICES or fn == CLONE_FILE:
                    register_voice(speaker, v, group=group)
    if len(VOICES): print(f'Loaded {len(VOICES)} voices: {VOICES.describe()}')
    log_model_params(M['model'])
    print('Warmup...')
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]], device=args.device)
        au = torch.full((1, 8, 3), AUDIO_PAD_TOKEN, dtype=torch.long, device=args.device)
        M['model'].forward(torch.cat((au, ids.unsqueeze(1)), dim=1))
        if M['model'].audio_encoder: M['model'].audio_encoder(torch.zeros(1, 100, 560, device=args.device), torch.tensor([100], device=args.device))
        if M['mimi']: M['mimi'].decode(torch.zeros(1, 8, 1, dtype=torch.long, device=args.device))
    print('Warmup done!')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--load_from', default='./', help='模型权重搜索目录；目录下可放多个 HF 格式模型，WebUI 会自动扫描并允许切换。')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='推理设备；CUDA 可用时默认 cuda。显存不足或排查环境问题时可改为 cpu。')
    p.add_argument('--port', default=7860, type=int, help='WebUI 服务端口；端口被占用或需要同时启动多个实例时调整。')
    p.add_argument('--audio_chunk_frames', default=4, type=int, help='流式播放每次解码的 Mimi frame 数；默认 4 约 320ms。WebUI 播放卡顿时可调大到 8/12，低延迟优先时保持 4。')
    p.add_argument('--audio_overlap', default=2, type=int, help='分块 Mimi 解码的重叠帧数；默认 2 用于缓解块边界断裂。一般不需要调整，边界杂音明显时可适当增大。')
    p.add_argument('--max_history_turns', default=0, type=int, help='对话历史轮数；默认 0 不带历史以降低延迟和显存。需要多轮上下文时调大，但会增加 prefill 成本。')
    args = p.parse_args()
    init_model(args)
    app.run(host='0.0.0.0', port=args.port, threaded=True)
