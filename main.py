"""掷骰模拟器交互入口。

用法：python main.py
"""

import json
from dataclasses import asdict

from src.dice_engine import DiceSimulator

# 复用同一实例：配置一次（此处用默认配置），可反复执行不同表达式；
# 未指定 seed，每次 execute() 自动生成不可预测种子
_sim = DiceSimulator()


def run_test(expr):
	print(f"\n>>> Input: {expr}")
	result = _sim.execute(expr)
	print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
	if asdict(result)["error"]:
		print(f"\n>>> {asdict(result)['error']['message']}")


def main():
	while True:
		try:
			run_test(input(">>> "))
		except (KeyboardInterrupt, EOFError):
			break


if __name__ == "__main__":
	main()
