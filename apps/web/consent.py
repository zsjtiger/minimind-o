"""音色克隆的授权确认与审计日志。

- 克隆前必须提交与当前 CONSENT_VERSION 一致的授权确认，否则拒绝。
- 每次克隆、拒绝、删除都追加一行 JSON 到审计日志；日志只记录哈希与元数据，不保存录音本身。
"""
import datetime
import hashlib
import json
import os
import threading

from baize_v.paths import OUT_DIR

CONSENT_VERSION = '2026-10-v1'
CONSENT_TEXT = (
    '我确认：这段录音是我本人，或录音者已明确授权我使用其声音；'
    '生成的音色只用于合法用途，不用于冒充他人、诈骗或制作误导性内容。'
)
AUDIT_LOG = os.environ.get('BAIZE_AUDIT_LOG', os.path.join(OUT_DIR, 'audit', 'voice_clone.jsonl'))

_lock = threading.Lock()


class ConsentError(RuntimeError):
    pass


def check_consent(payload):
    """payload 需包含 consent=True 与 consent_version=CONSENT_VERSION。"""
    if payload.get('consent') is not True:
        raise ConsentError('请先勾选授权确认后再克隆音色')
    if payload.get('consent_version') != CONSENT_VERSION:
        raise ConsentError('授权条款已更新，请刷新页面后重新确认')


def audio_digest(audio_bytes):
    return hashlib.sha256(audio_bytes).hexdigest()


def audit(event, path=None, **fields):
    """追加一条审计记录。写入失败不影响主流程，但会打印到 stderr。"""
    path = path or AUDIT_LOG
    record = {'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'event': event, **fields}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock, open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except OSError as e:
        print(f'[audit] write failed: {e}', flush=True)
