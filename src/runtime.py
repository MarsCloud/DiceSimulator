"""求值层：运行时节点 + 朴素逐步规约 + 渲染。

运行时节点（Num/Roll/DiceNode/Op）不同于语法 AST——它是"正在被规约的文档"。
规约采用朴素的自底向上单步规约（纯函数、不修改原树、无 cluster/结合律优化），
因此内部行为简单可预测；trace 的紧凑性由 simulator 层的"步骤过滤"负责。
"""

from typing import List

from .ast import Number, Dice, BinOp
from .config import DiceConfig
from .errors import DiceError


# ---------- 运行时节点 ----------


class Num:
	"""终值。"""
	__slots__ = ('value',)

	def __init__(self, value: int):
		self.value = value


class Roll:
	"""已掷出、尚未求和的骰子；展示为 '2+5+5'。"""
	__slots__ = ('rolls',)

	def __init__(self, rolls: List[int]):
		self.rolls = rolls


class DiceNode:
	"""待掷骰子 ndm；count/faces 为运行时子表达式。"""
	__slots__ = ('count', 'faces')

	def __init__(self, count, faces):
		self.count = count
		self.faces = faces


class Op:
	"""二元运算。"""

	__slots__ = ('op', 'left', 'right')

	# 运算符优先级：值越大结合越紧。新增二元运算只需在此登记级别，
	# _needs_parens 的加括号规则按数值比较，无需改动。
	PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}

	# 左结合（非交换/非结合）运算符：右侧同优先级子表达式需加括号
	# （如 a-(b+c)、a/(b*c)），否则会改变结合方向。
	NON_ASSOCIATIVE = frozenset({'-', '/'})

	def __init__(self, op: str, left, right):
		self.op = op
		self.left = left
		self.right = right


# ---------- AST → 运行时 ----------


def to_runtime(node):
	if isinstance(node, Number):
		return Num(node.value)
	if isinstance(node, Dice):
		return DiceNode(to_runtime(node.num), to_runtime(node.size))
	if isinstance(node, BinOp):
		return Op(node.op, to_runtime(node.left), to_runtime(node.right))
	raise TypeError(f'unknown AST node: {type(node).__name__}')


# ---------- 规约 ----------


def calc(op: str, l: int, r: int) -> int:
	if op == '+':
		return l + r
	if op == '-':
		return l - r
	if op == '*':
		return l * r
	if op == '/':
		if r == 0:
			raise DiceError('err_div_zero')
		return l // r
	raise ValueError(f'unknown operator: {op}')


def reduce_once(rt, config: DiceConfig, rng):
	"""单次自底向上规约（纯函数）。

	不变量的关键：子节点未变化时**返回原对象**（靠对象同一性判断），这样
	"本轮是否发生了某个子步"才能被父节点感知，从而保留嵌套骰子的叙事
	（如 (1d4)d6 会先显示 2D6）。
	"""
	if isinstance(rt, Num):
		return rt

	if isinstance(rt, Roll):
		return Num(sum(rt.rolls))

	if isinstance(rt, Op):
		new_left = reduce_once(rt.left, config, rng)
		new_right = reduce_once(rt.right, config, rng)
		if new_left is not rt.left or new_right is not rt.right:
			# 子节点本轮刚变化 → 先暴露子步，本轮不合并
			return Op(rt.op, new_left, new_right)
		if isinstance(new_left, Num) and isinstance(new_right, Num):
			return Num(calc(rt.op, new_left.value, new_right.value))
		return rt

	if isinstance(rt, DiceNode):
		new_count = reduce_once(rt.count, config, rng)
		new_faces = reduce_once(rt.faces, config, rng)
		if new_count is not rt.count or new_faces is not rt.faces:
			# 数量/面数本轮刚变化 → 先暴露子步，本轮不掷
			return DiceNode(new_count, new_faces)
		if isinstance(new_count, Num) and isinstance(new_faces, Num):
			return _roll(new_count.value, new_faces.value, config, rng)
		return rt

	return rt


def _roll(n: int, m: int, config: DiceConfig, rng):
	if n < 0:
		raise DiceError('err_dice_neg', val=n)
	if m < 1:
		raise DiceError('err_face_min', val=m)
	if n > config.max_dice_number:
		raise DiceError('err_dice_max', val=n, limit=config.max_dice_number)
	if m > config.max_dice_faces:
		raise DiceError('err_face_max', val=m, limit=config.max_dice_faces)

	rolls = [rng.randint(1, m) for _ in range(n)]
	if n >= config.threshold_sum_rolls:
		# 骰子过多：直接求和，不展示明细
		return Num(sum(rolls))
	if n >= config.threshold_sort_rolls:
		rolls.sort(reverse=True)
	return Roll(rolls)


# ---------- 渲染（把当前运行时树转成步骤文本） ----------


def render(rt) -> str:
	if isinstance(rt, Num):
		return str(rt.value)
	if isinstance(rt, Roll):
		if not rt.rolls:
			# 0dN：空掷骰列表按 0 显示，避免产生空字符串步骤
			return str(sum(rt.rolls))
		return '+'.join(str(r) for r in rt.rolls)
	if isinstance(rt, DiceNode):
		return _render_dice(rt)
	if isinstance(rt, Op):
		return _render_op(rt)
	return str(rt)


def _render_dice(node):
	n_str = render(node.count)
	s_str = render(node.faces)
	if isinstance(node.count, (Op, DiceNode)):
		n_str = f"({n_str})"
	if isinstance(node.faces, (Op, DiceNode)):
		s_str = f"({s_str})"
	if isinstance(node.count, Roll) and len(node.count.rolls) > 1:
		n_str = f"({n_str})"
	if isinstance(node.faces, Roll) and len(node.faces.rolls) > 1:
		s_str = f"({s_str})"
	if isinstance(node.count, Num) and node.count.value < 0:
		n_str = f"({n_str})"
	if isinstance(node.faces, Num) and node.faces.value < 0:
		s_str = f"({s_str})"
	return f"{n_str}D{s_str}"


def _needs_parens(node, my_op, is_right=False):
	"""按数值优先级决定子表达式是否需要加括号。"""
	if not isinstance(node, Op):
		return False
	my_prec = Op.PRECEDENCE[my_op]
	child_prec = Op.PRECEDENCE[node.op]
	if child_prec < my_prec:
		return True
	# 同级且位于右侧：左结合运算符右侧的同优先级子式需加括号
	if is_right and child_prec == my_prec and my_op in Op.NON_ASSOCIATIVE:
		return True
	return False


def _render_op(node):
	op = node.op
	l_str = render(node.left)
	r_str = render(node.right)

	if _needs_parens(node.left, op):
		l_str = f"({l_str})"
	if _needs_parens(node.right, op, is_right=True):
		r_str = f"({r_str})"

	# Roll 展开式子节点加括号，避免优先级歧义
	if isinstance(node.left, Roll) and len(node.left.rolls) > 1:
		l_str = f"({l_str})"
	if isinstance(node.right, Roll) and len(node.right.rolls) > 1:
		r_str = f"({r_str})"

	# 0 的特殊显示优化
	if isinstance(node.left, Num) and node.left.value == 0 and op == '+':
		op, l_str = '', ''
	if isinstance(node.left, Num) and node.left.value == 0 and op == '-':
		l_str = ''

	return f"{l_str}{op}{r_str}"


# ---------- 工具 ----------


def is_meaningful(rt) -> bool:
	"""该运行时树是否构成"有叙事意义的步骤"（用于步骤过滤）。

	有意义的步骤包括：存在未折叠的 Roll（掷骰结果）、存在数量与面数均已解析
	即将投掷的骰子（如 '2D4'，说明骰子规格已就绪）。
	"""
	if isinstance(rt, Roll):
		return True
	if isinstance(rt, DiceNode):
		if isinstance(rt.count, Num) and isinstance(rt.faces, Num):
			return True
		return is_meaningful(rt.count) or is_meaningful(rt.faces)
	if isinstance(rt, Op):
		return is_meaningful(rt.left) or is_meaningful(rt.right)
	return False


def collapse_rolls(rt):
	"""把树中所有 Roll 替换为 Num(sum)（纯函数，供超长输出折叠）。"""
	if isinstance(rt, Roll):
		return Num(sum(rt.rolls))
	if isinstance(rt, Op):
		return Op(rt.op, collapse_rolls(rt.left), collapse_rolls(rt.right))
	if isinstance(rt, DiceNode):
		return DiceNode(collapse_rolls(rt.count), collapse_rolls(rt.faces))
	return rt
