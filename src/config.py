"""配置与国际化。

DiceConfig 为不可变实例级配置（frozen dataclass），由 DiceSimulator 在构造时持有，
替代旧实现的全局可变类属性。国际化消息从 src/locales/<lang>.json 读取并缓存，
便于开源社区直接添加新语言文件提交翻译，无需改动代码。
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiceConfig:
	"""引擎限制、默认值与展示阈值。所有限制都是对单个 DiceSimulator 实例生效。"""

	# 解析限制
	max_recursion_depth: int = 50

	# 投掷限制
	max_dice_number: int = 1000
	max_dice_faces: int = 10000
	max_output_length: int = 1000
	max_simulation_steps: int = 100

	# 默认值
	default_dice_faces: int = 100

	# 展示阈值
	threshold_sort_rolls: int = 20
	threshold_sum_rolls: int = 50


class I18nManager:
	"""国际化资源管理器：消息表从 locales 目录按语言读取并缓存。

	添加新语言只需在 src/locales/ 下放置 <lang>.json（键与已有文件一致），
	即可通过 I18nManager.t(key, lang='<lang>') 使用。

	运行时修改翻译文件后不会自动生效，需手动调用 reload(lang) 强制重读；
	进程重启自然重新加载。
	"""

	# 默认语言（不可变常量）：调用方未指定 lang 时用哪个。
	# lang 不分大小写，统一以小写作为规范形式（如 'zh_cn'）
	DEFAULT_LANG = 'zh_cn'

	_LOCALE_DIR = Path(__file__).resolve().parent / 'locales'
	_cache = {}  # lang -> {message_key: template}

	@classmethod
	def _canonical_lang(cls, lang: str) -> str:
		"""把 lang 大小写不敏感地归一为小写规范形式（'zh_CN'/'ZH_CN'/'zh_cn' → 'zh_cn'）。

		内部缓存键、对外输出统一用小写；文件名大小写无关，读取时单独定位。
		"""
		return lang.lower()

	@classmethod
	def _resolve_path(cls, lang: str) -> Path:
		"""大小写不敏感地在 locales 目录中定位语言文件，找不到返回 None。"""
		target = cls._canonical_lang(lang)
		try:
			paths = list(cls._LOCALE_DIR.iterdir())
		except OSError:
			return None
		for p in paths:
			if p.suffix == '.json' and p.stem.lower() == target:
				return p
		return None

	@classmethod
	def _load(cls, lang: str) -> dict:
		"""读取并缓存某语言的模板表；文件缺失时返回空表（由 t 回退）。

		缓存键统一为小写规范形式（如 'zh_cn'），与文件名实际大小写无关。
		"""
		lang = cls._canonical_lang(lang)
		table = cls._cache.get(lang)
		if table is None:
			path = cls._resolve_path(lang)
			try:
				table = json.loads(path.read_text(encoding='utf-8')) if path else {}
			except OSError:
				table = {}
			cls._cache[lang] = table
		return table

	@classmethod
	def available_langs(cls) -> list:
		"""当前可用的语言（统一小写规范形式，如 ['en_us', 'zh_cn']）。"""
		try:
			names = [p.stem.lower() for p in cls._LOCALE_DIR.iterdir() if p.suffix == '.json']
		except OSError:
			return []
		return sorted(names)

	@classmethod
	def reload(cls, lang: str = None) -> None:
		"""强制重读消息表：清除缓存，下次 t() 时重新从文件加载。

		lang 指定时只重读该语言（大小写不敏感）；为 None 时重读所有可用语言。
		"""
		if lang is None:
			for name in cls.available_langs():
				cls._cache.pop(name, None)
		else:
			cls._cache.pop(cls._canonical_lang(lang), None)

	@classmethod
	def t(cls, key: str, lang: str = None, **kwargs) -> str:
		lang = cls._canonical_lang(lang or cls.DEFAULT_LANG)
		msg = cls._load(lang).get(key)
		if msg is None:
			# 缺失键回退默认语言，再回退到键名本身
			if lang != cls.DEFAULT_LANG:
				return cls.t(key, lang=cls.DEFAULT_LANG, **kwargs)
			return key
		return msg.format(**kwargs)
