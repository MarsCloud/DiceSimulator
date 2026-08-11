"""配置与国际化。

DiceConfig 为不可变实例级配置（frozen dataclass），由 DiceSimulator 在构造时持有，
替代旧实现的全局可变类属性。国际化消息从 src/locales/<lang>.json 读取并缓存，
便于开源社区直接添加新语言文件提交翻译，无需改动代码。
"""

import json
import os
from dataclasses import dataclass


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

	# 默认语言（不可变常量）；DiceSimulator(lang=...) 可按实例覆盖
	DEFAULT_LANG = 'zh_CN'

	_LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locales')
	_cache = {}  # lang -> {message_key: template}

	@classmethod
	def _load(cls, lang: str) -> dict:
		"""读取并缓存某语言的模板表；文件缺失时返回空表（由 t 回退）。"""
		table = cls._cache.get(lang)
		if table is None:
			path = os.path.join(cls._LOCALE_DIR, f'{lang}.json')
			try:
				with open(path, encoding='utf-8') as f:
					table = json.load(f)
			except OSError:
				table = {}
			cls._cache[lang] = table
		return table

	@classmethod
	def available_langs(cls) -> list:
		"""当前可用的语言（按 locales 目录中存在的 .json 文件）。"""
		try:
			names = os.listdir(cls._LOCALE_DIR)
		except OSError:
			return []
		return sorted(name[:-5] for name in names if name.endswith('.json'))

	@classmethod
	def reload(cls, lang: str = None) -> None:
		"""强制重读消息表：清除缓存，下次 t() 时重新从文件加载。

		lang 指定时只重读该语言；为 None 时重读所有可用语言
		（含新添加的语言文件）。
		"""
		if lang is None:
			for name in cls.available_langs():
				cls._cache.pop(name, None)
		else:
			cls._cache.pop(lang, None)

	@classmethod
	def t(cls, key: str, lang: str = None, **kwargs) -> str:
		lang = lang or cls.DEFAULT_LANG
		msg = cls._load(lang).get(key)
		if msg is None:
			# 缺失键回退英文，再回退到键名本身
			if lang != 'en_US':
				return cls.t(key, lang='en_US', **kwargs)
			return key
		return msg.format(**kwargs)
