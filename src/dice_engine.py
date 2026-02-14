import re
import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


# ==========================================
# 0. 配置与国际化 (Configuration & I18n)
# ==========================================

class DiceConfig:
	"""全局配置类：集中管理魔法数值和常量"""
	# 限制设置
	MAX_RECURSION_DEPTH = 50  # AST最大深度
	MAX_DICE_NUMBER = 1000  # 单次最大投掷骰子数 (防止 9999d6)
	MAX_DICE_FACES = 10000  # 骰子最大面数
	MAX_OUTPUT_LENGTH = 1000  # 结果文本最大长度，超过折叠
	MAX_SIMULATION_STEPS = 100  # 最大规约步数 (防止死循环)

	# 默认值
	DEFAULT_DICE_FACES = 100  # 当输入 'd' 时默认的面数 (d = d100)

	# 阈值策略
	THRESHOLD_SORT_ROLLS = 20  # 超过多少个骰子时，结果进行排序显示
	THRESHOLD_SUM_ROLLS = 50  # 超过多少个骰子时，直接显示总和不显示明细

	# 词法分析配置
	REPLACEMENTS = {
		'（': '(', '）': ')',
		'【': '(', '】': ')',
		'➕': '+', '➖': '-',
		'✖': '*', '×': '*', 'x': '*', 'X': '*',
		'➗': '/', '÷': '/',
		'd': 'D',
	}

	# 正则表达式
	TOKEN_PATTERN = re.compile(r'\d+|D|[\+\-\*\/\(\)]')


class I18nManager:
	"""国际化资源管理器"""
	_LANG = 'zh_CN'

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
		}
	}

	@classmethod
	def t(cls, key: str, **kwargs) -> str:
		msg_tpl = cls._MESSAGES.get(cls._LANG, cls._MESSAGES['en_US']).get(key, key)
		return msg_tpl.format(**kwargs)


class DiceError(Exception):
	"""
	掷骰模拟器的自定义异常。
	message_key: 对应 I18nManager 中的键
	"""

	def __init__(self, message_key: str, pos: int = None, **kwargs):
		self.message_key = message_key
		self.pos = pos
		self.params = kwargs
		# 为了兼容普通打印，立即格式化一条消息
		self.message = I18nManager.t(message_key, pos=pos, **kwargs)
		super().__init__(self.message)

	def to_dict(self):
		return {
			"error_code": self.message_key,
			"position": self.pos,
			"message": self.message,
			"params": self.params
		}


# ==========================================
# 1. AST Node Definitions (数据结构)
# ==========================================

class Node(ABC):
	@abstractmethod
	def render(self) -> str:
		pass


class Number(Node):
	def __init__(self, value):
		self.value = int(value)

	def render(self):
		return str(self.value)


class DiceExpanded(Node):
	"""已投掷但未求和的状态"""

	def __init__(self, rolls: List[int]):
		self.rolls = rolls
		self.collapsed = False

	def render(self):
		if self.collapsed or not self.rolls:
			return str(sum(self.rolls))
		return "+".join(str(r) for r in self.rolls)

	def total(self):
		return sum(self.rolls)


class Dice(Node):
	"""待投掷指令 (ndm)"""

	def __init__(self, num: Node, size: Node):
		self.num = num
		self.size = size

	def render(self):
		n_str = self.num.render()
		s_str = self.size.render()
		if isinstance(self.num, (BinOp, Dice, DiceExpanded)): n_str = f"({n_str})"
		if isinstance(self.size, (BinOp, Dice, DiceExpanded)): s_str = f"({s_str})"
		if isinstance(self.num, Number) and self.num.value < 0: n_str = f"({n_str})"
		if isinstance(self.size, Number) and self.size.value < 0: s_str = f"({s_str})"
		return f"{n_str}D{s_str}"


class BinOp(Node):
	"""二元运算"""

	def __init__(self, left: Node, op: str, right: Node):
		self.left = left
		self.op = op
		self.right = right

	def render(self):
		l_str = self.left.render()
		op_str = self.op
		r_str = self.right.render()

		def needs_parens(node, my_op, is_right=False):
			if isinstance(node, (Number, DiceExpanded, Dice)): return False
			if isinstance(node, BinOp):
				high = {'*', '/'}
				low = {'+', '-'}
				if my_op in high and node.op in low: return True
				if is_right:
					if my_op == '-' and node.op in low: return True
					if my_op == '/' and node.op in high: return True
			return False

		if needs_parens(self.left, op_str): l_str = f"({l_str})"
		if needs_parens(self.right, op_str, is_right=True): r_str = f"({r_str})"

		def is_exp(node):
			return isinstance(node, DiceExpanded) and len(node.rolls) > 1 and not node.collapsed

		if is_exp(self.left): l_str = f"({l_str})"
		if is_exp(self.right): r_str = f"({r_str})"

		# 0 的特殊处理优化显示
		if isinstance(self.left, Number) and self.left.value == 0 and op_str == '+':
			op_str, l_str = '', ''
		if isinstance(self.left, Number) and self.left.value == 0 and op_str == '-':
			l_str = ''

		return f"{l_str}{op_str}{r_str}"


# ==========================================
# 2. Tokenizer & Parser (解析层)
# ==========================================

class Tokenizer:
	def tokenize(self, text: str) -> List[str]:
		clean_text = text
		for k, v in DiceConfig.REPLACEMENTS.items():
			clean_text = clean_text.replace(k, v)

		tokens = []
		pos = 0
		length = len(clean_text)

		while pos < length:
			char = clean_text[pos]
			if char.isspace():
				pos += 1
				continue

			match = DiceConfig.TOKEN_PATTERN.match(clean_text, pos)
			if not match:
				raise DiceError('err_illegal_char', pos=pos, char=char)

			token_str = match.group()
			tokens.append(token_str)
			pos = match.end()

		return tokens


class Parser:
	def __init__(self, tokens: List[str]):
		self.tokens = tokens
		self.pos = 0
		self.n_tokens = len(tokens)

	def peek(self) -> Optional[str]:
		return self.tokens[self.pos] if self.pos < self.n_tokens else None

	def consume(self, expected: str = None) -> str:
		token = self.peek()
		if token is None:
			raise DiceError('err_unexpected_end')
		if expected and token != expected:
			raise DiceError('err_syntax', pos=self.pos, expected=expected, token=token)
		self.pos += 1
		return token

	def parse(self) -> Node:
		if not self.tokens:
			# 什么都没写默认扔一颗骰子
			return Dice(Number(1), Number(DiceConfig.DEFAULT_DICE_FACES))
		node = self.expr(depth=0)
		if self.peek() is not None:
			raise DiceError('err_unparsed', pos=self.pos, token=self.peek())
		return node

	def check_depth(self, depth):
		if depth > DiceConfig.MAX_RECURSION_DEPTH:
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
			if token == '-': sign *= -1

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
				right = Number(DiceConfig.DEFAULT_DICE_FACES)
			node = Dice(Number(1), right)
		else:
			node = self.atom(depth + 1)

		while self.peek() == 'D':
			self.consume()
			if self.peek() and (self.peek().isdigit() or self.peek() == '(' or self.peek() == 'D'):
				right = self.atom(depth + 1)
			else:
				right = Number(DiceConfig.DEFAULT_DICE_FACES)
			node = Dice(node, right)
		return node

	def atom(self, depth):
		self.check_depth(depth)
		token = self.peek()
		if token is None:
			raise DiceError('err_missing_atom')

		if token.isdigit():
			return Number(self.consume())

		if token == '(':
			self.consume('(')
			node = self.expr(depth + 1)
			if self.peek() != ')':
				raise DiceError('err_missing_paren')
			self.consume(')')
			return node

		if token == 'D':
			return self.dice_ops(depth + 1)

		raise DiceError('err_invalid_syntax', token=token)


# ==========================================
# 3. Simulator & API Wrapper (业务逻辑层)
# ==========================================

@dataclass
class DiceResult:
	"""标准 API 返回结构"""
	raw_input: str
	steps: List[str]  # 过程列表 ["3d6", "1+5+2", "8"]
	result: Optional[int]  # 最终结果
	is_success: bool  # 是否执行成功
	seed: int  # 随机数种子
	error: Optional[Dict]  # 错误信息字典


class DiceSimulator:
	def __init__(self, expr_str: str, seed=None):
		self.expr_str = expr_str
		self.ast: Optional[Node] = None
		self.history: List[str] = []
		self.seed = seed

		if self.seed is None:
			self.seed = int(time.time())
		self._random = random.Random(seed)

	def execute(self) -> DiceResult:
		"""API 入口：执行模拟并返回结构化数据"""
		try:
			tokenizer = Tokenizer()
			tokens = tokenizer.tokenize(self.expr_str)
			parser = Parser(tokens)
			self.ast = parser.parse()
		except DiceError as e:
			return DiceResult(self.expr_str, [], None, False, self.seed, e.to_dict())
		except Exception as e:
			# 捕获未知错误
			return DiceResult(self.expr_str, [], None, False, self.seed, {"message": str(e), "type": "Unknown"})

		# 记录初始状态
		initial_render = self.ast.render()
		self.history.append(initial_render)

		last_render = initial_render
		step_count = 0
		final_val = None
		error_info = None

		while step_count < DiceConfig.MAX_SIMULATION_STEPS:
			step_count += 1

			try:
				self.ast = self._transform(self.ast)
			except DiceError as e:
				error_info = e.to_dict()
				break

			current_text = self.ast.render()

			# 文本过长保护：强制折叠
			if len(current_text) > DiceConfig.MAX_OUTPUT_LENGTH:
				self._force_collapse_recursive(self.ast)
				current_text = self.ast.render()

			if current_text != last_render:
				self.history.append(current_text)
				last_render = current_text

			if isinstance(self.ast, Number):
				final_val = self.ast.value
				break
		else:
			# 循环结束（熔断）
			error_info = {
				"message": I18nManager.t("err_steps_limit"),
				"error_code": "err_steps_limit"
			}

		return DiceResult(
			raw_input=self.expr_str,
			steps=self.history,
			result=final_val,
			is_success=(error_info is None),
			seed=self.seed,
			error=error_info
		)

	def _force_collapse_recursive(self, node: Node):
		if isinstance(node, DiceExpanded):
			node.collapsed = True
		elif isinstance(node, BinOp):
			self._force_collapse_recursive(node.left)
			self._force_collapse_recursive(node.right)
		elif isinstance(node, Dice):
			self._force_collapse_recursive(node.num)
			self._force_collapse_recursive(node.size)

	def _transform(self, node: Node) -> Node:
		"""核心规约函数 (Immutable-style)"""

		if isinstance(node, BinOp):
			new_left = self._transform(node.left)
			new_right = self._transform(node.right)
			children_changed = (new_left is not node.left) or (new_right is not node.right)

			node.left = new_left
			node.right = new_right

			# Cluster 优化
			if node.op in ('+', '*', '-'):
				val = self._try_resolve_cluster(node)
				if val is not None:
					return Number(val)

			# 只有当左右都是纯数字时才计算
			if isinstance(node.left, Number) and isinstance(node.right, Number):
				if not children_changed:
					return Number(self._calc(node.left.value, node.op, node.right.value))

			# 结合律优化
			optimized_node = self._optimize_associativity(node)
			if optimized_node is not node:
				return optimized_node

			return node

		elif isinstance(node, Dice):
			new_num = self._transform(node.num)
			new_size = self._transform(node.size)
			children_changed = (new_num is not node.num) or (new_size is not node.size)
			node.num = new_num
			node.size = new_size

			if isinstance(node.num, Number) and isinstance(node.size, Number):
				if not children_changed:
					n = node.num.value
					m = node.size.value

					# 检查限制
					if n < 0: raise DiceError('err_dice_neg', val=n)
					if m < 1: raise DiceError('err_face_min', val=m)
					if n > DiceConfig.MAX_DICE_NUMBER:
						raise DiceError('err_dice_max', val=n, limit=DiceConfig.MAX_DICE_NUMBER)
					if m > DiceConfig.MAX_DICE_FACES:
						raise DiceError('err_face_max', val=m, limit=DiceConfig.MAX_DICE_FACES)

					rolls = [self._random.randint(1, m) for _ in range(n)]

					if n >= DiceConfig.THRESHOLD_SUM_ROLLS:
						return Number(sum(rolls))
					if n >= DiceConfig.THRESHOLD_SORT_ROLLS:
						rolls.sort(reverse=True)

					return DiceExpanded(rolls)
			return node

		elif isinstance(node, DiceExpanded):
			return Number(node.total())

		return node

	# --- 辅助计算逻辑 ---

	def _calc(self, l, op, r):
		if op == '+': return l + r
		if op == '-': return l - r
		if op == '*': return l * r
		if op == '/':
			if r == 0: raise DiceError('err_div_zero')
			return l // r
		return 0

	def _get_val(self, node):
		return node.value if isinstance(node, Number) else None

	# --- 优化算法: Cluster & Associativity ---

	def _try_resolve_cluster(self, node: BinOp):
		target_op = node.op
		leaves = []
		if not self._collect_leaves(node, target_op, leaves):
			return None

		if target_op == '+': return sum(leaves)
		if target_op == '*':
			res = 1
			for x in leaves: res *= x
			return res
		if target_op == '-':
			res = leaves[0]
			for x in leaves[1:]: res -= x
			return res
		return None

	def _collect_leaves(self, node: Node, op: str, leaves: List[int]):
		val = self._get_val(node)
		if val is not None:
			leaves.append(val)
			return True
		if isinstance(node, BinOp) and node.op == op:
			return self._collect_leaves(node.left, op, leaves) and \
				self._collect_leaves(node.right, op, leaves)
		return False

	def _optimize_associativity(self, node: Node):
		if not isinstance(node, BinOp) or node.op not in ('+', '-'):
			return node

		# 左重: (SubTree) op Const
		if isinstance(node.left, BinOp) and self._get_val(node.right) is not None:
			new_node = self._try_merge_constants(
				outer_const=self._get_val(node.right),
				outer_op=node.op,
				subtree=node.left,
				outer_pos='right'
			)
			if new_node: return new_node

		# 右重: Const op (SubTree)
		if self._get_val(node.left) is not None and isinstance(node.right, BinOp):
			new_node = self._try_merge_constants(
				outer_const=self._get_val(node.left),
				outer_op=node.op,
				subtree=node.right,
				outer_pos='left'
			)
			if new_node: return new_node

		return node

	def _try_merge_constants(self, outer_const, outer_op, subtree, outer_pos):
		if subtree.op not in ('+', '-'): return None

		# 识别子树内层常数
		inner_val = self._get_val(subtree.right)
		if inner_val is not None:
			inner_const = inner_val
			expr_node = subtree.left
			inner_pos = 'right'
		else:
			inner_val = self._get_val(subtree.left)
			if inner_val is not None:
				inner_const = inner_val
				expr_node = subtree.right
				inner_pos = 'left'
			else:
				return None

		# 计算外层贡献
		outer_effect = outer_const
		if outer_pos == 'right' and outer_op == '-':
			outer_effect = -outer_const

		# 计算上下文符号
		context_sign = 1
		if outer_pos == 'left' and outer_op == '-':
			context_sign = -1

		# 计算内层贡献
		inner_local_effect = inner_const
		if inner_pos == 'right' and subtree.op == '-':
			inner_local_effect = -inner_const
		inner_effect = inner_local_effect * context_sign

		total_constant = outer_effect + inner_effect

		# 计算表达式符号
		expr_local_sign = 1
		if inner_pos == 'left' and subtree.op == '-':
			expr_local_sign = -1
		final_expr_sign = expr_local_sign * context_sign

		if final_expr_sign == 1:
			if total_constant >= 0:
				return BinOp(expr_node, '+', Number(total_constant))
			else:
				return BinOp(expr_node, '-', Number(abs(total_constant)))
		else:
			return BinOp(Number(total_constant), '-', expr_node)


# ==========================================
# 4. API Usage Example
# ==========================================

if __name__ == "__main__":
	import json


	def run_test(expr):
		print(f"\n>>> Input: {expr}")
		sim = DiceSimulator(expr)
		result = sim.execute()

		# 模拟 API 响应格式化
		response = {
			"success": result.is_success,
			"result": result.result,
			"steps": result.steps,
			"seed": result.seed,
			"error": result.error
		}
		print("API 风格调用：")
		print(json.dumps(response, indent=2, ensure_ascii=False))

		print("直接展示：")
		print("\n=".join(result.steps))
		print("")


	while True:
		try:
			run_test(input(">>> "))
		except KeyboardInterrupt:
			break
