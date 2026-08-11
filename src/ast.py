"""纯语法层 AST 节点。

只描述表达式结构，无渲染方法、无运行时状态、不可变（内部代码从不修改节点，
因此求值层可用对象同一性判断"是否变化"）。
"""


class Node:
	"""AST 节点基类（纯数据标记）。"""
	__slots__ = ()


class Number(Node):
	__slots__ = ('value',)

	def __init__(self, value):
		self.value = int(value)


class Dice(Node):
	"""待投掷指令 ndm：num 为数量表达式，size 为面数表达式。"""
	__slots__ = ('num', 'size')

	def __init__(self, num, size):
		self.num = num
		self.size = size


class BinOp(Node):
	"""二元运算。"""
	__slots__ = ('left', 'op', 'right')

	def __init__(self, left, op, right):
		self.left = left
		self.op = op
		self.right = right
