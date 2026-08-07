"""掷骰模拟器交互入口。

用法：python main.py
"""

from src.dice_engine import DiceSimulator


def run_test(expr):
    print(f"\n>>> Input: {expr}")
    result = DiceSimulator(expr).execute()
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
