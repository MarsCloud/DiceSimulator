"""模拟器：DiceResult / DiceSimulator + 执行循环与步骤过滤。

规约内部是"朴素逐步规约"（runtime.reduce_once），会产出全部中间状态；
本层在展示时过滤掉纯算术中间步（无 Roll 且非首/末/折叠步），
从而在内部不依赖 cluster/结合律优化的情况下，仍输出紧凑的步骤 trace。
"""

import random
import time
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
    steps: List[str]          # 过滤后的步骤列表
    result: Optional[int]     # 最终结果
    is_success: bool          # 是否执行成功
    seed: int                 # 随机数种子（用该 seed 重放可复现结果）
    error: Optional[Dict]     # 错误信息字典


class DiceSimulator:
    def __init__(self, expr_str: str, seed=None, *,
                 config: DiceConfig = None, lang: str = None, rng=None):
        self.expr_str = expr_str
        self.config = config or DiceConfig()
        self.lang = lang or I18nManager._LANG

        # 修复旧 bug：必须用 self.seed 初始化 RNG，保证返回的 seed 可复现结果
        self.seed = seed
        if self.seed is None:
            self.seed = int(time.time())
        self._random = rng if rng is not None else random.Random(self.seed)

    def _format_error(self, err: DiceError) -> Dict:
        return {
            "error_code": err.message_key,
            "position": err.pos,
            "message": I18nManager.t(err.message_key, lang=self.lang,
                                     pos=err.pos, **err.params),
            "params": err.params,
        }

    def execute(self) -> DiceResult:
        try:
            tokens = Tokenizer().tokenize(self.expr_str)
            ast = Parser(tokens, self.config).parse()
        except DiceError as e:
            return DiceResult(self.expr_str, [], None, False, self.seed, self._format_error(e))
        except Exception as e:
            # 仅在 API 最外层兜底未知错误
            return DiceResult(self.expr_str, [], None, False, self.seed,
                              {"error_code": "err_unknown", "position": None,
                               "message": str(e), "params": {}})

        rt = to_runtime(ast)
        raw_steps = [render(rt)]
        keep = [True]  # 首步总是保留

        step_count = 0
        final_val = None
        error_info = None

        while step_count < self.config.max_simulation_steps:
            step_count += 1
            try:
                rt2 = reduce_once(rt, self.config, self._random)
            except DiceError as e:
                error_info = self._format_error(e)
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
            error_info = self._format_error(DiceError('err_steps_limit'))

        if final_val is not None:
            # 终值步始终保留
            keep[-1] = True

        steps = [s for s, k in zip(raw_steps, keep) if k]

        return DiceResult(
            raw_input=self.expr_str,
            steps=steps,
            result=final_val,
            is_success=(error_info is None),
            seed=self.seed,
            error=error_info,
        )
