"""配置与国际化。

DiceConfig 为不可变实例级配置（frozen dataclass），由 DiceSimulator 在构造时持有，
替代旧实现的全局可变类属性；I18nManager._LANG 仅为默认语言，可被实例级 lang 覆盖。
"""

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


_MESSAGES = {
    'zh_CN': {
        'err_illegal_char': "非法字符 '{char}' (位置: {pos})",
        'err_unexpected_end': "语句未结束，期待更多输入",
        'err_syntax': "语法错误: 期待 '{expected}', 但得到 '{token}' (位置: {pos})",
        'err_unparsed': "无法解析的剩余字符: '{token}' (位置: {pos})",
        'err_depth_limit': "表达式嵌套过深",
        'err_missing_paren': "括号不匹配，缺少 ')'",
        'err_missing_atom': "语句突然结束，期待数字或括号",
        'err_invalid_syntax': "无效的语法标记: '{token}'",
        'err_dice_neg': "骰子数量不能为负数: {val}",
        'err_face_min': "骰子面数必须大于0: {val}",
        'err_dice_max': "骰子数量过大 ({val} > {limit})",
        'err_face_max': "骰子面数过大 ({val} > {limit})",
        'err_div_zero': "除数不能为零",
        'err_steps_limit': "计算步骤过多，强制停止",
        'err_unknown': "发生未知错误",
    },
    'en_US': {
        'err_illegal_char': "Illegal character '{char}' at {pos}",
        'err_unexpected_end': "Unexpected end of input",
        'err_syntax': "Syntax error: expected '{expected}', got '{token}' at {pos}",
        'err_unparsed': "Unparsed characters remaining: '{token}' at {pos}",
        'err_depth_limit': "Expression recursion depth exceeded",
        'err_missing_paren': "Mismatched parentheses, missing ')'",
        'err_missing_atom': "Unexpected end, expected number or '('",
        'err_invalid_syntax': "Invalid syntax token: '{token}'",
        'err_dice_neg': "Dice count cannot be negative: {val}",
        'err_face_min': "Dice faces must be > 0: {val}",
        'err_dice_max': "Too many dice ({val} > {limit})",
        'err_face_max': "Dice faces too large ({val} > {limit})",
        'err_div_zero': "Division by zero",
        'err_steps_limit': "Computation steps exceeded limit",
        'err_unknown': "Unknown error",
    },
}


class I18nManager:
    """国际化资源管理器。

    _LANG 仅为"默认语言"，可被 DiceSimulator(lang=...) 按实例覆盖。
    """

    _LANG = 'zh_CN'

    @classmethod
    def t(cls, key: str, lang: str = None, **kwargs) -> str:
        lang = lang or cls._LANG
        msg_tpl = _MESSAGES.get(lang, _MESSAGES['en_US']).get(key, key)
        return msg_tpl.format(**kwargs)
