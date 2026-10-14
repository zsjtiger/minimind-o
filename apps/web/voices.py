"""音色注册表：内置音色、未见音色与用户克隆音色。"""

GROUPS = ('builtin', 'unseen', 'manual')


class VoiceRegistry:
    def __init__(self):
        self._voices = {}                       # 名称 -> {ref_codes, spk_emb}
        self._groups = {g: [] for g in GROUPS}  # 分组 -> 名称列表

    def __contains__(self, name):
        return name in self._voices

    def __getitem__(self, name):
        return self._voices[name]

    def __len__(self):
        return len(self._voices)

    def in_group(self, name, group):
        return name in self._groups[group]

    def register(self, name, value, group='manual'):
        """注册音色；同名音色会从其他分组移出，归入 group。"""
        self._voices[name] = value
        if name not in self._groups[group]:
            self._groups[group].append(name)
        for g, names in self._groups.items():
            if g != group and name in names:
                names.remove(name)

    def remove_manual(self, name):
        self._voices.pop(name, None)
        if name in self._groups['manual']:
            self._groups['manual'].remove(name)

    def summary(self):
        return {g: sorted(self._groups[g]) for g in GROUPS}

    def describe(self):
        s = self.summary()
        return ', '.join(f'{g}={s[g]}' for g in GROUPS)
