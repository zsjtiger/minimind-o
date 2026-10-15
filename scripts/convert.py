import os
import json
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from baize_v.config import OmniConfig
from baize_v.models import BaizeOmni
from baize_v.models.summary import report_load_result
from baize_v.models.llm import remap_legacy_state_dict
from baize_v.paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR, OUT_DIR

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(torch_path, transformers_path, lm_config, dtype=torch.bfloat16):
    OmniConfig.register_for_auto_class()
    BaizeOmni.register_for_auto_class("AutoModelForCausalLM")
    model = BaizeOmni(lm_config, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR)
    state_dict = torch.load(torch_path, map_location='cpu')
    state_dict = remap_legacy_state_dict(state_dict, lm_config)
    report_load_result(model.load_state_dict(state_dict, strict=False), label=os.path.basename(torch_path))
    model = model.to(dtype)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'模型参数: {params:.2f}M')
    del model.audio_encoder
    del model.vision_encoder
    model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.save_pretrained(transformers_path)
    config_path = os.path.join(transformers_path, "config.json")
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    config['tie_word_embeddings'] = True
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; config.pop('rope_parameters', None)
    json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"已保存为 PyTorch 格式: {torch_path}")


if __name__ == '__main__':
    lm_config = OmniConfig(hidden_size=768, use_moe=False, prelude_layers=2, coda_layers=2, max_loop_iters=4)
    torch_path = os.path.join(OUT_DIR, f"sft_omni_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth")
    transformers_path = os.path.join(OUT_DIR, 'baize-v')
    convert_torch2transformers(torch_path, transformers_path, lm_config)
