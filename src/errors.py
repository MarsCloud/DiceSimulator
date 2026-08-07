"""异常类型。

DiceError 只承载结构化信息（错误码 + 位置 + 参数），消息文本由调用方按实例语言
通过 to_dict(lang=...) 格式化——避免"构造时定死语言"的问题。
"""

from .config import I18nManager


class DiceError(Exception):
    def __init__(self, message_key: str, pos: int = None, **params):
        self.message_key = message_key
        self.pos = pos
        self.params = params
        super().__init__(message_key)

    @property
    def message(self) -> str:
        # 兼容旧用法：用默认语言即时格式化
        return I18nManager.t(self.message_key, pos=self.pos, **self.params)

    def to_dict(self, lang: str = None) -> dict:
        return {
            "error_code": self.message_key,
            "position": self.pos,
            "message": I18nManager.t(self.message_key, lang=lang,
                                     pos=self.pos, **self.params),
            "params": self.params,
        }
