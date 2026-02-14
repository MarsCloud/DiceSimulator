# 🎲 Dice Simulator Engine (AST-Based)

A secure, step-by-step dice rolling engine based on Abstract Syntax Tree (AST) parsing.

基于抽象语法树 (AST) 的安全、分步展示的掷骰模拟引擎。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

## ✨ Features / 特性

*   **🛡️ Secure & Safe**: Uses a custom Recursive Descent Parser instead of `eval()`. Safe against code injection.
    *   **安全**: 使用自定义递归下降解析器，而非 `eval()`，防止代码注入攻击。
*   **📝 Step-by-Step History**: Returns not just the result, but the calculation process (e.g., `3d6` -> `1+4+2` -> `7`).
    *   **过程展示**: 不仅返回结果，还保留计算过程（如 `3d6` 展开为 `1+4+2`）。
*   **⚡ Optimized Math**: Includes Associativity Optimization and Cluster Optimization to handle complex expressions efficiently.
    *   **数学优化**: 内置结合律优化与聚类优化，高效处理复杂算式。
*   **🌍 Internationalization (I18n)**: Built-in support for English (`en_US`) and Chinese (`zh_CN`) error messages.
    *   **国际化**: 内置中英文错误提示支持。
*   **⚙️ Highly Configurable**: Adjustable limits for recursion depth, max dice count, and max faces to prevent abuse.
    *   **高度可配置**: 可限制递归深度、最大骰子数和面数，防止资源滥用。

## 🚀 Quick Start / 快速开始

### Installation / 安装

Clone the repository and import the engine:

克隆仓库并导入引擎：

```bash
git clone https://github.com/MarsCloud/dice-simulator.git
cd dice-simulator
```

### Usage / 使用示例

```python
from src.dice_engine import DiceSimulator

# 1. Simple Roll / 简单投掷
sim = DiceSimulator("3D6 + 5")
result = sim.execute()

if result.is_success:
	print(f"Result: {result.result}")       # Output: 16
	print(f"Steps: {result.steps}")         # Output: ['3D6+5', '(1+4+6)+5', '16']
else:
	print(f"Error: {result.error}")

# 2. Nested Logic / 嵌套逻辑
# Roll (1d4) dice, each having 6 faces
sim = DiceSimulator("(1d4)d6")
result = sim.execute()

if result.is_success:
	print(f"Result: {result.result}")       # Output: 14
	print(f"Steps: {result.steps}")         # Output: ['(1D4)D6', '4D6', '3+6+2+3', '14']
```

### API Output Structure / API 返回结构

The engine returns a structured `DiceResult` object, perfect for JSON serialization (REST APIs / Bots).

引擎返回结构化的对象，适合用于 REST API 或 机器人开发。

```json
{
  "result": 13,
  "is_success": true,
  "steps": [
    "2d10+5",
    "(3+5)+5",
    "13"
  ],
  "error": null
}
```

## ⚙️ Configuration / 配置

You can customize the engine limits in `DiceConfig` class:

你可以在 `DiceConfig` 类中自定义限制：

```python
from src.dice_engine import DiceConfig, I18nManager

# Set limits / 设置限制
DiceConfig.MAX_DICE_NUMBER = 100   # Max dice at once
DiceConfig.MAX_DICE_FACES = 1000   # Max faces per die
DiceConfig.MAX_RECURSION_DEPTH = 20

# Switch Language / 切换语言
I18nManager._LANG = 'zh_CN'  # 'en_US' or 'zh_CN'
```

## 🧪 Testing / 测试

The project includes a comprehensive test suite covering arithmetic, syntax errors, and edge cases.
本项目包含完整的测试套件，覆盖了算术、语法错误和边界情况。

```bash
# Run all tests
python -m unittest tests/test_dice.py
```

## 📂 Project Structure / 项目结构

```text
dice-simulator/
├── src/
│   ├── __init__.py
│   └── dice_engine.py    # Core logic (Parser, AST, Simulator)
├── tests/
│   ├── __init__.py
│   └── test_dice.py      # Unit tests with Mocking
├── .gitignore
└── README.md
```

## 📜 License

This project is licensed under the MIT License.

本项目采用 MIT 许可证。