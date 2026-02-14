import unittest
from unittest.mock import patch

# 假设重构后的代码保存在 dice_engine.py 中
# 如果你在同一个文件中，可以直接使用，不需要 import
from src.dice_engine import DiceSimulator, DiceConfig, I18nManager


class TestDiceRoller(unittest.TestCase):

	def setUp(self):
		"""每个测试前重置配置，防止测试污染"""
		self.original_lang = I18nManager._LANG
		DiceConfig.MAX_DICE_NUMBER = 500
		DiceConfig.MAX_DICE_FACES = 10000

	def tearDown(self):
		I18nManager._LANG = self.original_lang

	# ==========================================
	# 1. 基础算术测试 (Basic Arithmetic)
	# ==========================================

	def test_basic_math(self):
		cases = [
			("1 + 1", 2),
			("10 - 2", 8),
			("2 * 3", 6),
			("10 / 2", 5),
			("2 + 3 * 4", 14),  # 优先级：乘法优先
			("(2 + 3) * 4", 20),  # 括号优先
			("10 - 2 - 2", 6),  # 左结合
			("10 / 2 / 5", 1),  # 左结合
		]
		for expr, expected in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator(expr).execute()
				self.assertTrue(res.is_success, f"Failed: {res.error}")
				self.assertEqual(res.result, expected)

	def test_unary_operators(self):
		"""测试一元运算符 (+x, -x)"""
		cases = [
			("-5", -5),
			("10 + -5", 5),
			("10 * -2", -20),
			("-(-5)", 5),  # 双重负号
			("---5", -5),  # 三重负号
			("+5", 5)
		]
		for expr, expected in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator(expr).execute()
				self.assertEqual(res.result, expected)

	# ==========================================
	# 2. 输入清洗与词法测试 (Tokenizer)
	# ==========================================

	def test_input_sanitization(self):
		"""测试全角符号、中文括号、Emoji替换"""
		cases = [
			("1➕1", 2),  # Emoji 加号
			("1➖1", 0),  # Emoji 减号
			("2✖3", 6),  # Emoji 乘号
			("10➗2", 5),  # Emoji 除号
			("（1+1）", 2),  # 中文括号
			("【2+2】", 4),  # 中文方括号
			("3d6", None),  # 小写d
			("3D6", None),  # 大写D
		]
		for expr, expected in cases:
			res = DiceSimulator(expr).execute()
			self.assertTrue(res.is_success)
			if expected is not None:
				self.assertEqual(res.result, expected)

	# ==========================================
	# 3. 掷骰逻辑测试 (Dice Mechanics)
	# ==========================================

	@patch('random.randint')
	def test_dice_rolls(self, mock_randint):
		"""测试骰子基础逻辑，Mock随机数固定结果"""
		# 设定：第一次调用返回 3，第二次返回 4，第三次返回 5 ...
		mock_randint.side_effect = [3, 4, 5, 1, 1, 1]

		# Case 1: 3d6 -> 3 + 4 + 5 = 12
		res = DiceSimulator("3d6").execute()
		self.assertEqual(res.result, 12)
		# 检查步骤中是否包含展开过程 "3+4+5"
		self.assertTrue(any("3+4+5" in step for step in res.steps))

		# Case 2: d100 (默认d) -> 下一个随机数是 1
		res = DiceSimulator("d").execute()
		self.assertEqual(res.result, 1)  # mock returns 1

	@patch('random.randint')
	def test_dice_complex_syntax(self, mock_randint):
		mock_randint.return_value = 5

		# 测试 (表达式)d(表达式)
		# (1+1)d(2+2) -> 2d4
		res = DiceSimulator("(1+1)d(2+2)").execute()
		self.assertEqual(res.result, 10)  # 2个骰子，每个都是5 => 10

	# ==========================================
	# 4. 优化算法测试 (Optimization)
	# ==========================================

	def test_cluster_optimization(self):
		"""测试纯数字连加/连乘的合并优化"""
		# 1+1+1+1 应该在极少的步骤内完成，而不是分步规约
		sim = DiceSimulator("1+1+1+1+1")
		res = sim.execute()
		self.assertEqual(res.result, 5)
		# 如果优化生效，步骤数应该很少
		self.assertLess(len(res.steps), 5)

	@patch('random.randint')
	def test_associativity_optimization(self, mock_randint):
		"""测试结合律优化: (d10 + 5) + 5 -> d10 + 10"""
		mock_randint.return_value = 1

		sim = DiceSimulator("(d10 + 5) + 5")
		res = sim.execute()

		# 检查是否发生了常数合并
		# 如果优化生效，历史记录中应该出现 "1D100+10" 或类似的结构（在骰子出结果前）
		# 或者在骰子变成 "1" 之后，直接变成 "11" 而不是 "1+5+5"
		self.assertEqual(res.result, 11)

		# 更严格的检查：在AST渲染历史中寻找合并痕迹
		# 原始解析: ((1D10)+5)+5
		# 优化后应有一步: 1D10+10
		has_optimized_step = any("1+10" in step for step in res.steps)
		self.assertTrue(has_optimized_step, "结合律优化未生效，未发现合并后的常数 10")

	# ==========================================
	# 5. 边界与限制测试 (Limits & Edges)
	# ==========================================

	def test_limits_dice_count(self):
		"""测试最大骰子数量限制"""
		DiceConfig.MAX_DICE_NUMBER = 10
		res = DiceSimulator("11d6").execute()
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_dice_max')

	def test_limits_dice_faces(self):
		"""测试最大面数限制"""
		DiceConfig.MAX_DICE_FACES = 100
		res = DiceSimulator("1d101").execute()
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_face_max')

	def test_negative_dice_params(self):
		"""测试负数骰子参数"""
		res = DiceSimulator("(-1)d6").execute()
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_dice_neg')

		res = DiceSimulator("1d(-5)").execute()
		self.assertFalse(res.is_success)  # 运行时错误：面数不能小于1
		self.assertEqual(res.error['error_code'], 'err_face_min')

	def test_recursion_depth(self):
		"""测试递归深度限制"""
		# 构造超深嵌套 (((((...)))))
		deep_expr = "(" * 60 + "1" + ")" * 60
		res = DiceSimulator(deep_expr).execute()
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_depth_limit')

	def test_step_limit_circuit_breaker(self):
		"""测试死循环熔断 (通过Mock强制不收敛来模拟，或者设置极低的步数限制)"""
		DiceConfig.MAX_SIMULATION_STEPS = 2
		res = DiceSimulator("1*2+1d6d6").execute()
		# 2步肯定算不完
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'warn_steps_limit')

	# ==========================================
	# 6. 错误处理测试 (Error Handling)
	# ==========================================

	def test_syntax_errors(self):
		cases = [
			("3d6 5", "err_unparsed"),  # 语句未结束
			("(1+1", "err_missing_paren"),  # 括号不匹配
			("d*", "err_missing_atom"),  # d后面跟非法字符
		]
		for expr, err_code in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator(expr).execute()
				self.assertFalse(res.is_success)
				self.assertEqual(res.error['error_code'], err_code)

	def test_runtime_errors(self):
		res = DiceSimulator("10 / 0").execute()
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_div_zero')

	# ==========================================
	# 7. 国际化测试 (I18n)
	# ==========================================

	def test_i18n_switching(self):
		"""测试切换语言是否影响错误消息"""
		expr = "10 / 0"

		# Test Chinese
		I18nManager._LANG = 'zh_CN'
		res_cn = DiceSimulator(expr).execute()
		self.assertIn("除数不能为零", res_cn.error['message'])

		# Test English
		I18nManager._LANG = 'en_US'
		res_en = DiceSimulator(expr).execute()
		self.assertIn("Division by zero", res_en.error['message'])

	# ==========================================
	# 8. 输出保护测试
	# ==========================================

	@patch('random.randint')
	def test_output_collapse(self, mock_randint):
		"""测试结果过长时的折叠"""
		DiceConfig.MAX_OUTPUT_LENGTH = 10  # 设得极小
		mock_randint.return_value = 1

		# 10d6 展开应该是 "1+1+1..." 肯定超过10个字符
		sim = DiceSimulator("10d6")
		res = sim.execute()

		self.assertTrue(res.is_success)
		# 检查最后一步是否被折叠成数字，而不是长字符串
		# 注意：DiceExpanded 如果 collapsed=True，render直接返回 sum
		self.assertEqual(res.result, 10)

		# 检查历史记录中是否存在某种状态表明被强制求和了
		# 在代码中，如果长度超标，会强制调用 _force_collapse_recursive
		# 这通常意味着中间那些很长的 "1+1+1+..." 字符串不会出现在最终的 steps 里，
		# 或者只出现一次就被下一轮变成数字了。
		# 我们可以检查 raw history 长度
		for step in res.steps:
			if len(step) > DiceConfig.MAX_OUTPUT_LENGTH + 20:  # 给一点buffer
				self.fail(f"Step text too long: {step}")


if __name__ == '__main__':
	unittest.main()
