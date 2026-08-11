"""词法与语法层：Tokenizer + Parser，输出纯 AST（见 ast.py）。

词法规则（全角/Emoji/符号替换、token 模式）收在 Tokenizer 类内——它们是"语言语法"
而非可调限制；解析限制（递归深度、默认面数）来自实例级 DiceConfig。
Tokenizer 无状态、Parser 在 parse() 时重置状态，因此二者都可复用
（由 DiceSimulator 各持一份实例）。
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

			tokens.append(match.group())
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
		return self.tokens[self.pos] if self.pos < self.n_tokens else None

	def consume(self) -> str:
		# 调用方均已保证当前存在 token；consume 只负责取出并前进
		token = self.peek()
		self.pos += 1
		return token

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
			raise DiceError('err_unparsed', pos=self.pos, token=self.peek())
		return node

	def check_depth(self, depth):
		if depth > self.config.max_recursion_depth:
			raise DiceError('err_depth_limit')

	# --- Recursive Descent Logic ---

	def expr(self, depth):
		self.check_depth(depth)
		node = self.term(depth + 1)
		while self.peek() in ('+', '-'):
			op = self.consume()
			right = self.term(depth + 1)
			node = BinOp(node, op, right)
		return node

	def term(self, depth):
		self.check_depth(depth)
		node = self.unary(depth + 1)
		while self.peek() in ('*', '/'):
			op = self.consume()
			right = self.unary(depth + 1)
			node = BinOp(node, op, right)
		return node

	def unary(self, depth):
		self.check_depth(depth)
		sign = 1
		while self.peek() in ('+', '-'):
			token = self.consume()
			if token == '-':
				sign *= -1
		node = self.dice_ops(depth + 1)
		if sign == -1:
			return BinOp(Number(0), '-', node)
		return node

	def dice_ops(self, depth):
		self.check_depth(depth)
		if self.peek() == 'D':
			self.consume()
			if self.peek() and (self.peek().isdigit() or self.peek() == '(' or self.peek() == 'D'):
				right = self.atom(depth + 1)
			else:
				right = Number(self.config.default_dice_faces)
			node = Dice(Number(1), right)
		else:
			node = self.atom(depth + 1)

		while self.peek() == 'D':
			self.consume()
			if self.peek() and (self.peek().isdigit() or self.peek() == '(' or self.peek() == 'D'):
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
			raise DiceError('err_missing_atom', pos=self.pos)

		if token.isdigit():
			return Number(self.consume())

		if token == '(':
			self.consume()
			node = self.expr(depth + 1)
			if self.peek() is None:
				# 输入在 ')' 之前就结束了 → 真的缺右括号
				raise DiceError('err_missing_paren')
			if self.peek() != ')':
				# 括号内还残留其他内容 → 与顶层一致地报"剩余字符"
				raise DiceError('err_unparsed', pos=self.pos, token=self.peek())
			self.consume()
			return node

		if token == 'D':
			return self.dice_ops(depth + 1)

		raise DiceError('err_invalid_syntax', token=token)
