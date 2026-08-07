"""门面模块：保持公共导入 API 不变。

历史版本为单文件实现；现拆分为 config / ast / parser / runtime / simulator 等
模块，此文件只负责重新导出公共 API。

注意：本文件使用相对导入，不能直接 `python src/dice_engine.py` 运行；
交互式演示入口在项目根目录的 main.py。
"""

from .config import DiceConfig, I18nManager
from .errors import DiceError
from .simulator import DiceSimulator, DiceResult

__all__ = ['DiceConfig', 'I18nManager', 'DiceError', 'DiceSimulator', 'DiceResult']
