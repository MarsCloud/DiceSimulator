# 🎲 Dice Simulator Engine (AST-Based)

A secure, step-by-step dice rolling engine based on Abstract Syntax Tree (AST) parsing.

基于抽象语法树 (AST) 的安全、分步展示的掷骰模拟引擎。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

## ✨ Features / 特性

*   **🛡️ Secure & Safe**: Uses a custom Recursive Descent Parser instead of `eval()`. Safe against code injection.
    *   **安全**: 使用自定义递归下降解析器，而非 `eval()`，防止代码注入攻击。
*   **📝 Step-by-Step History**: Returns not just the result, but the calculation process (e.g., `3d6` -> `1+4+2` -> `7`).
    *   **过程展示**: 不仅返回结果，还保留计算过程（如 `3d6` 展开为 `1+4+2`）。
*   **🪶 Compact Traces**: Internally reduces naively step-by-step; the display layer filters out pure-arithmetic intermediate steps, keeping the trace compact.
    *   **紧凑轨迹**: 内部朴素逐步规约，展示层过滤纯算术中间步，trace 保持紧凑。
*   **♻️ Reusable Instance**: Configure a `DiceSimulator` once, then `execute()` many different expressions on the same instance.
    *   **可复用实例**: `DiceSimulator` 配置一次，即可在同一实例上反复执行不同表达式。
*   **🌍 Internationalization (I18n)**: English (`en_US`) and Chinese (`zh_CN`) built-in; messages are loaded from `src/locales/*.json`, so the community can add a new language without touching code.
    *   **国际化**: 内置中英文错误提示；文案从 `src/locales/*.json` 读取，社区可直接添加语言文件提交翻译。
*   **⚙️ Highly Configurable**: Per-instance limits for recursion depth, max dice count, and max faces to prevent abuse.
    *   **高度可配置**: 实例级限制，可配置递归深度、最大骰子数和面数，防止资源滥用。

## 🚀 Quick Start / 快速开始

### Prerequisites / 环境要求

*   Python 3.10+
*   [uv](https://docs.astral.sh/uv/)（包管理与虚拟环境）

### Setup / 安装

```bash
git clone https://github.com/MarsCloud/dice-simulator.git
cd dice-simulator

# 创建虚拟环境并安装项目（生成 .venv 与 uv.lock）
uv sync
```

### Usage / 使用示例

```python
from src.dice_engine import DiceSimulator

# 1. Simple Roll / 简单投掷
sim = DiceSimulator()
result = sim.execute("3d6 + 5")

if result.is_success:
    print(f"Result: {result.result}")   # 例如 20
    print(f"Steps: {result.steps}")     # 例如 ['3D6+5', '(4+5+6)+5', '20']
    print(f"Seed: {result.seed}")       # 用该 seed 重放可复现结果
else:
    print(f"Error: {result.error}")

# 2. Nested Logic / 嵌套逻辑
# 掷 1d4 颗骰子，每颗 6 面
result = sim.execute("(1d4)d6")
print(f"Result: {result.result}")

# 3. Switch Language / 切换语言（按实例生效，互不影响）
zh = DiceSimulator(lang='zh_CN')
en = DiceSimulator(lang='en_US')
print(zh.execute("10/0").error['message'])  # 除数为零
print(en.execute("10/0").error['message'])  # Division by zero
```

> 同一实例的多次 `execute()` 调用每次使用独立推导的种子，`result.seed` 均可单独重放。

### Interactive Demo / 交互式演示

```bash
uv run python main.py
```

### API Output Structure / API 返回结构

The engine returns a structured `DiceResult` object, perfect for JSON serialization (REST APIs / Bots).

引擎返回结构化的对象，适合用于 REST API 或 机器人开发。

```json
{
  "raw_input": "2d10+5",
  "steps": ["2D10+5", "(3+5)+5", "13"],
  "result": 13,
  "is_success": true,
  "seed": 1786435139,
  "error": null,
  "lang": "zh_CN"
}
```

## ⚙️ Configuration / 配置

`DiceConfig` is a frozen dataclass passed via the constructor; limits only apply to that instance.

`DiceConfig` 是 frozen dataclass，通过构造参数传入，限制只对单个实例生效。

```python
from src.dice_engine import DiceSimulator, DiceConfig, I18nManager

sim = DiceSimulator(config=DiceConfig(
    max_dice_number=100,      # 单次最大骰子数
    max_dice_faces=1000,      # 单颗骰子最大面数
    max_recursion_depth=20,   # 最大递归深度
    max_simulation_steps=100, # 最大规约步数（熔断）
    default_dice_faces=100,   # 未写面数时的默认面数，如 "d6"
))

# 默认语言与可用语言
print(I18nManager.DEFAULT_LANG)       # zh_CN
print(I18nManager.available_langs())  # ['en_US', 'zh_CN']
```

### I18n / 国际化

*   语言按实例指定：`DiceSimulator(lang='zh_CN')`。
*   消息表从 `src/locales/<lang>.json` 读取；新增语言只需添加键结构与现有文件一致的 JSON 文件。
*   语言名大小写不敏感：`zh_cn` / `zh_CN` 均可。
*   缺失键回退到默认语言（`DEFAULT_LANG`），再回退到键名本身。
*   修改翻译文件后需调用 `I18nManager.reload(lang)` 手动重读（进程重启也会自动重新加载）。

## 🧪 Testing / 测试

The project includes a comprehensive test suite covering arithmetic, syntax errors, and edge cases.
本项目包含完整的测试套件，覆盖了算术、语法错误和边界情况。

```bash
uv run python -m unittest test.test_dice
```

## 📂 Project Structure / 项目结构

```text
dice-simulator/
├── src/
│   ├── config.py        # DiceConfig（frozen dataclass）与 I18nManager
│   ├── ast.py           # AST 节点定义
│   ├── parser.py        # 词法 + 递归下降解析器
│   ├── runtime.py       # 运行时规约（朴素逐步 + 展示层过滤）
│   ├── simulator.py     # DiceSimulator / DiceResult（公共 API）
│   ├── errors.py        # DiceError（结构化错误）
│   ├── dice_engine.py   # 门面：重新导出公共 API
│   └── locales/         # 多语言消息表（*.json）
├── test/
│   └── test_dice.py     # 单元测试
├── pyproject.toml       # uv / 打包配置
├── uv.lock
├── main.py              # 交互式演示入口
└── README.md
```

## 📜 License

This project is licensed under the MIT License.

本项目采用 MIT 许可证。
