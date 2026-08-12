"""异常类型。

DiceError 只承载结构化信息（错误码 + 参数），消息文本由调用方按实例语言
通过 to_dict(lang=..., source=...) 格式化——避免"构造时定死语言"的问题。

指示位置的错误（err_illegal_char / err_unparsed / err_missing_atom）会在 dict 里
多带一个 context 字段（position 也合并进来，顶层不再带 position）：
  position:    出错位置
  text:        上下文窗口原文
  widths:      单字符显示宽度（系统字符均为 ASCII，故恒为 1）
  arrow_text:  对齐好的箭头行（不含换行，\n 由消费方拼接）
message 保持纯描述，由消费方按需拼接（如 message + '\n' + text + '\n' + arrow_text）。
"""

from .config import I18nManager


def build_context(source: str, pos: int, length: int = 11) -> dict:
	"""从原表达式截取上下文窗口（默认总长 11，可自定义，最小 5），返回 context 结构。

	text 按出错位置取三种形态之一：
	  ..xxxxxxx..   出错点在中部（两侧截断）
	  xxxxxxxxx..   出错点靠近开头（仅右侧截断）
	  ..xxxxxxxxx   出错点靠近末尾（仅左侧截断）
	'..' 只在该方向确实有内容被截掉时才出现——源串不够长时直接把内容
	整个贴出来，不会假装还有未完的输入。

	系统只展示 0-9/D/+ - * / ( ) 这些 ASCII 字符，单字符显示宽度一致
	（widths=1）；widths 保留下来供前端按所在平台的空格宽度做对齐适配。
	arrow_text 是后端按自身空格宽度（每空格 1 列）预拼好的箭头行。
	"""
	length = max(5, length)
	ellipsis = '..'  # 截断标记
	content_len = length - 2 * len(ellipsis)  # 窗口里内容区占用的长度
	before = content_len // 2  # 出错字符前的内容长度
	after = content_len - before - 1  # 出错字符后的内容长度

	total = len(source)
	pos = max(0, min(pos, total))

	# 确定内容切片边界（出错字符尽量居中，不足则贴边）
	if pos >= before and total - pos > after:
		start, end = pos - before, pos + after + 1  # 中部
	elif pos < before:
		start, end = 0, length - len(ellipsis)  # 靠近开头
	else:
		start, end = total - (length - len(ellipsis)), total  # 靠近末尾

	start = max(0, start)
	end = min(total, end)
	content = source[start:end]

	# 截断方向的 '..' 仅在确实截掉了内容时才加
	left = ellipsis if start > 0 else ''
	right = ellipsis if end < total else ''
	text = left + content + right

	# 出错字符在窗口内的下标（pos 为末尾时指向最后一个字符之后）
	err_idx = len(left) + (pos - start)
	width = 1
	return {
		"position": pos,
		"text": text,
		"widths": width,
		"arrow_text": ' ' * width * err_idx + '^',
	}


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

	def to_dict(self, lang: str = None, source: str = None) -> dict:
		lang = (lang or I18nManager.DEFAULT_LANG).lower()
		result = {
			"error_code": self.message_key,
			"message": I18nManager.t(self.message_key, lang=lang, pos=self.pos, **self.params),
			"params": self.params,
			"lang": lang,  # 只有错误信息需要语言，lang 属于 error 内层
		}
		if self.pos is not None and source is not None:
			# 指示位置的错误：context 承载 position/text/widths/arrow_text
			result["context"] = build_context(source, self.pos)
		return result
