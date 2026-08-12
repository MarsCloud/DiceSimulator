"""模拟器：DiceResult / DiceSimulator + 执行循环与步骤过滤。

规约内部是"朴素逐步规约"（runtime.reduce_once），会产出全部中间状态；
本层在展示时过滤掉纯算术中间步（无 Roll 且非首/末/折叠步），
从而在内部不依赖 cluster/结合律优化的情况下，仍输出紧凑的步骤 trace。
"""

import random
import secrets
from dataclasses import dataclass
from typing import List, Optional, Dict

from .config import DiceConfig, I18nManager
from .errors import DiceError
from .parser import Tokenizer, Parser
from .runtime import (Num, to_runtime, reduce_once, render,
					  is_meaningful, collapse_rolls)


@dataclass
class DiceResult:
	"""标准 API 返回结构。"""
	raw_input: str
	steps: List[str]  # 过滤后的步骤列表
	result: Optional[int]  # 最终结果
	is_success: bool  # 是否执行成功
	seed: int  # 随机数种子（用该 seed 重放可复现结果）
	error: Optional[Dict]  # 错误信息字典（含 lang，见 DiceError.to_dict）


class DiceSimulator:
	"""可复用的掷骰模拟器：配置一次，多次执行。

	用法：
		sim = DiceSimulator(config=cfg, lang='zh_CN')
		sim.execute('3d6')                  # 每次自动用 secrets 生成不可预测种子
		sim.execute('3d6', seed=42)         # 本次指定 seed，结果可复现
		sim.execute('1d20+5')               # 复用同一实例

	Tokenizer 与 Parser 由本实例持有并复用（无全局单例）。
	"""

	def __init__(self, *,
				 config: DiceConfig = None, lang: str = None, rng=None):
		self.config = config or DiceConfig()
		# lang 大小写不敏感，统一归一为小写规范形式
		self.lang = (lang or I18nManager.DEFAULT_LANG).lower()

		# 词法/语法层实例持有并复用；Parser 在 parse() 时重置状态
		self._tokenizer = Tokenizer()
		self._parser = Parser(self.config)

		self._rng = rng  # 显式注入随机源（测试等确定性场景），否则按 seed 构造

	def execute(self, expr_str: str, seed: int = None) -> DiceResult:
		if not isinstance(expr_str, str):
			raise ValueError("execute() 需要表达式字符串参数，例如 sim.execute('3d6')")

		# 种子策略：
		# - 传入 seed → 本次执行基于该 seed，结果稳定可重放；
		# - 未传入 → 本次用 secrets 生成不可预测种子，result.seed 记录该种子。
		if seed is None:
			seed = secrets.randbits(32)
		rng = self._rng if self._rng is not None else random.Random(seed)

		# 错误以 DiceError（或未知错误 dict）原样携带，到最后组装点才转 dict
		steps = []
		final_val = None
		error = None

		try:
			tokens = self._tokenizer.tokenize(expr_str)
			ast = self._parser.parse(tokens)
		except DiceError as e:
			error = e
		except Exception as e:
			# 仅在 API 最外层兜底未知错误
			error = {"error_code": "err_unknown",
					 "message": str(e), "params": {}, "lang": self.lang}

		if error is None:
			rt = to_runtime(ast)
			raw_steps = [render(rt)]
			keep = [True]  # 首步总是保留

			step_count = 0
			while step_count < self.config.max_simulation_steps:
				step_count += 1
				try:
					rt2 = reduce_once(rt, self.config, rng)
				except DiceError as e:
					error = e
					break

				text = render(rt2)
				collapsed = False
				if len(text) > self.config.max_output_length:
					rt2 = collapse_rolls(rt2)
					text = render(rt2)
					collapsed = True

				if text != raw_steps[-1]:
					raw_steps.append(text)
					keep.append(is_meaningful(rt2) or collapsed)

				if isinstance(rt2, Num):
					final_val = rt2.value
					break
				rt = rt2
			else:
				# 循环自然结束（达到最大步数）→ 熔断
				error = DiceError('err_steps_limit')

			if final_val is not None:
				# 终值步始终保留
				keep[-1] = True

			steps = [s for s, k in zip(raw_steps, keep) if k]

		# 单一组装点：DiceError 在此统一转 dict（带上原始表达式以便指示位置的错误
		# 截取上下文窗口），其余（未知错误 dict / None）原样透传
		error_dict = error.to_dict(lang=self.lang, source=expr_str) if isinstance(error, DiceError) else error

		return DiceResult(
			raw_input=expr_str,
			steps=steps,
			result=final_val,
			is_success=(error_dict is None),
			seed=seed,
			error=error_dict,
		)
