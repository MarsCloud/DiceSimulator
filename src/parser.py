"""词法与语法层：Tokenizer + Parser，输出纯 AST（见 ast.py）。

词法规则（全角/Emoji/符号替换、token 模式）收在 Tokenizer 类内——它们是"语言语法"
而非可调限制；解析限制（递归深度、默认面数）来自实例级 DiceConfig。
Tokenizer 无状态、Parser 在 parse() 时重置状态，因此二者都可复用
（由 DiceSimulator 各持一份实例）。

tokenize() 返回 (value, src_pos) 元组，src_pos 是 token 在原始表达式中的字符下标
（全角替换是 1:1 的，故可直接映射回用户输入）——指示位置的错误由此拿到真实上下文位置。
"""

import re

from .ast import Number, Dice, BinOp
from .config import DiceConfig
from .errors import DiceError


class Tokenizer:
	"""词法器：无状态，可复用。"""

	_REPLACEMENTS = {
		'（': '(', '）': ')',
		'【': '(', '】': ')',
		'➕': '+', '➖': '-',
		'✖': '*', '×': '*', 'x': '*', 'X': '*',
		'➗': '/', '÷': '/',
		'd': 'D',
	}

	_TOKEN_PATTERN = re.compile(r'\d+|D|[+\-*/()]')

	def tokenize(self, text: str):
		clean_text = text
		for k, v in self._REPLACEMENTS.items():
			clean_text = clean_text.replace(k, v)

		tokens = []
		pos = 0
		length = len(clean_text)

		while pos < length:
			char = clean_text[pos]
			if char.isspace():
				pos += 1
				continue

			match = self._TOKEN_PATTERN.match(clean_text, pos)
			if not match:
				raise DiceError('err_illegal_char', pos=pos, char=char)

			tokens.append((match.group(), pos))
			pos = match.end()

		return tokens


class Parser:
	"""递归下降解析器：配置注入一次，parse(tokens) 可反复调用。"""

	def __init__(self, config: DiceConfig):
		self.config = config
		self.tokens = []
		self.pos = 0
		self.n_tokens = len(self.tokens)

	def peek(self):
		# 返回 (value, src_pos) 或 None
		return self.tokens[self.pos] if self.pos < self.n_tokens else None

	def peek_value(self) -> str:
		token = self.peek()
		return token[0] if token else None

	def consume(self) -> str:
		# 调用方均已保证当前存在 token；consume 只负责取出并前进
		token = self.peek()
		self.pos += 1
		return token[0]

	def src_pos(self) -> int:
		"""当前游标在源串中的位置：还有剩余 token 用其起点，否则用最后一个 token 的终点。"""
		if self.pos < self.n_tokens:
			return self.tokens[self.pos][1]
		if self.tokens:
			value, start = self.tokens[-1]
			return start + len(value)
		return 0

	def parse(self, tokens):
		# 每次 parse 重置状态，使同一实例可复用
		self.tokens = tokens
		self.pos = 0
		self.n_tokens = len(tokens)
		if not self.tokens:
			# 什么都没写默认扔一颗骰子
			return Dice(Number(1), Number(self.config.default_dice_faces))
		node = self.expr(depth=0)
		if self.peek() is not None:
			raise DiceError('err_unparsed', pos=self.peek()[1], token=self.peek_value())
		return node

	def check_depth(self, depth):
		if depth > self.config.max_recursion_depth:
			raise DiceError('err_depth_limit')

	# --- Recursive Descent Logic ---

	def expr(self, depth):
		self.check_depth(depth)
		node = self.term(depth + 1)
		while self.peek_value() in ('+', '-'):
			op = self.consume()
			right = self.term(depth + 1)
			node = BinOp(node, op, right)
		return node

	def term(self, depth):
		self.check_depth(depth)
		node = self.unary(depth + 1)
		while self.peek_value() in ('*', '/'):
			op = self.consume()
			right = self.unary(depth + 1)
			node = BinOp(node, op, right)
		return node

	def unary(self, depth):
		self.check_depth(depth)
		sign = 1
		while self.peek_value() in ('+', '-'):
			token = self.consume()
			if token == '-':
				sign *= -1
		node = self.dice_ops(depth + 1)
		if sign == -1:
			return BinOp(Number(0), '-', node)
		return node

	def dice_ops(self, depth):
		self.check_depth(depth)
		if self.peek_value() == 'D':
			self.consume()
			if self.peek_value() and (self.peek_value().isdigit() or self.peek_value() == '(' or self.peek_value() == 'D'):
				right = self.atom(depth + 1)
			else:
				right = Number(self.config.default_dice_faces)
			node = Dice(Number(1), right)
		else:
			node = self.atom(depth + 1)

		while self.peek_value() == 'D':
			self.consume()
			if self.peek_value() and (self.peek_value().isdigit() or self.peek_value() == '(' or self.peek_value() == 'D'):
				right = self.atom(depth + 1)
			else:
				right = Number(self.config.default_dice_faces)
			node = Dice(node, right)
		return node

	def atom(self, depth):
		self.check_depth(depth)
		token = self.peek()
		if token is None:
			# 输入在此处结束，但这里本应出现数字或括号（如 '1+'、'd*'）
			raise DiceError('err_missing_atom', pos=self.src_pos())

		value = token[0]

		if value.isdigit():
			return Number(self.consume())

		if value == '(':
			self.consume()
			node = self.expr(depth + 1)
			if self.peek() is None:
				# 输入在 ')' 之前就结束了 → 真的缺右括号，位置指向应出现 ')' 的游标处
				raise DiceError('err_missing_paren', pos=self.src_pos())
			if self.peek_value() != ')':
				# 括号内还残留其他内容 → 与顶层一致地报"剩余字符"
				raise DiceError('err_unparsed', pos=self.peek()[1], token=self.peek_value())
			self.consume()
			return node

		if value == 'D':
			return self.dice_ops(depth + 1)

		raise DiceError('err_invalid_syntax', token=value)
