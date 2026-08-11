"""掷骰模拟器交互入口。

用法：python main.py
"""

from src.dice_engine import DiceSimulator

# 复用同一实例：配置一次（此处用默认配置），可反复执行不同表达式
_sim = DiceSimulator()


def run_test(expr):
	print(f"\n>>> Input: {expr}")
	result = _sim.execute(expr)
	print("直接展示：")
	print("{}\n".format('\n='.join(result.steps)) if result.steps else "", end="")
	print(f"发生错误：{result.error['message']}\n" if result.error else "", end="")


def main():
	while True:
		try:
			run_test(input(">>> "))
		except (KeyboardInterrupt, EOFError):
			break


if __name__ == "__main__":
	main()
