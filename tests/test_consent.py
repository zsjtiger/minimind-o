import base64
import json
import os
import sys

import pytest

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'apps', 'web')


def _web_import_path():
    if WEB_DIR not in sys.path:
        sys.path.insert(0, WEB_DIR)


def test_check_consent_requires_flag_and_current_version():
    _web_import_path()
    from consent import CONSENT_VERSION, ConsentError, check_consent
    check_consent({'consent': True, 'consent_version': CONSENT_VERSION})
    with pytest.raises(ConsentError):
        check_consent({'consent_version': CONSENT_VERSION})
    with pytest.raises(ConsentError):
        check_consent({'consent': 'yes', 'consent_version': CONSENT_VERSION})  # 必须是布尔 True
    with pytest.raises(ConsentError):
        check_consent({'consent': True, 'consent_version': 'old'})


def test_audit_appends_json_lines_without_audio(tmp_path):
    _web_import_path()
    from consent import audio_digest, audit
    log = tmp_path / 'audit' / 'voice.jsonl'
    audit('clone_created', path=str(log), voice='a', audio_sha256=audio_digest(b'x'))
    audit('clone_deleted', path=str(log), voice='a')
    lines = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    assert [r['event'] for r in lines] == ['clone_created', 'clone_deleted']
    assert lines[0]['audio_sha256'] == audio_digest(b'x') and 'ts' in lines[0]
    assert 'audio' not in lines[0]


def test_clone_route_rejects_without_consent(tmp_path, monkeypatch):
    _web_import_path()
    import consent
    import server
    log = tmp_path / 'voice.jsonl'
    monkeypatch.setattr(consent, 'AUDIT_LOG', str(log))
    client = server.app.test_client()
    audio = base64.b64encode(b'not-really-audio').decode()
    resp = client.post('/clone_voice', json={'audio': audio, 'name': 'x'})
    assert resp.status_code == 403
    assert json.loads(resp.data)['ok'] is False
    events = [json.loads(line)['event'] for line in log.read_text(encoding='utf-8').splitlines()]
    assert events == ['clone_rejected']


def test_consent_route_returns_version_and_text():
    _web_import_path()
    import server
    from consent import CONSENT_VERSION
    data = json.loads(server.app.test_client().get('/clone_consent').data)
    assert data['version'] == CONSENT_VERSION and data['text']
