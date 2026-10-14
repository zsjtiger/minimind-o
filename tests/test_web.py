import json
import os
import sys

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'apps', 'web')


def _web_import_path():
    if WEB_DIR not in sys.path:
        sys.path.insert(0, WEB_DIR)


def test_voice_registry_moves_between_groups():
    _web_import_path()
    from voices import VoiceRegistry
    reg = VoiceRegistry()
    reg.register('a', 1, 'builtin')
    reg.register('a', 2, 'manual')
    assert reg['a'] == 2
    assert reg.in_group('a', 'manual') and not reg.in_group('a', 'builtin')
    assert reg.summary() == {'builtin': [], 'unseen': [], 'manual': ['a']}
    reg.remove_manual('a')
    assert 'a' not in reg and len(reg) == 0


def test_voices_route_lists_groups():
    _web_import_path()
    import server
    resp = server.app.test_client().get('/voices')
    assert resp.status_code == 200
    assert set(json.loads(resp.data)) == {'builtin', 'unseen', 'manual'}
