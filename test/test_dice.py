# -*- coding: utf-8 -*-
"""DiceSimulator 单元测试。

针对重构后的模块化引擎（config / ast / parser / runtime / simulator）重写。
与旧测试的关键差异：
1. DiceConfig 是 frozen 实例级配置，不再用可变类属性；限制通过构造时传入
   `DiceSimulator(config=DiceConfig(...))` 生效。
2. RNG 是注入式的（`DiceSimulator(rng=...)`），不再 mock 模块级 random.randint。
3. 内部改为"朴素逐步规约 + 展示层步骤过滤"：cluster / 结合律优化已移除，
   原优化测试替换为验证"纯算术中间步被过滤、trace 仍紧凑"。
4. 熔断错误码为 err_steps_limit（旧名 warn_steps_limit 已不存在）。
5. DiceSimulator 可复用：配置一次、execute(expr) 多次；表达式必须传给 execute。
6. 词法常量、消息表、优先级表均已收进类内；消息表从 src/locales/*.json 读取。

测试中用 FakeRNG 注入确定性随机数，使掷骰结果可精确断言。
"""

import json
import tempfile
from pathlib import Path
import unittest

from src.dice_engine import DiceSimulator, DiceConfig, I18nManager
from src.errors import build_context


class FakeRNG:
	"""按序返回预设值的确定性 RNG，替代对 random.randint 的 mock。"""

	def __init__(self, values):
		self.values = list(values)
		self.i = 0

	def randint(self, lo, hi):
		value = self.values[self.i]
		self.i += 1
		return value


# ==========================================
# 1. 基础算术
# ==========================================

class TestBasicMath(unittest.TestCase):
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
				res = DiceSimulator().execute(expr)
				self.assertTrue(res.is_success, res.error)
				self.assertEqual(res.result, expected)

	def test_unary_operators(self):
		cases = [
			("-5", -5),
			("+5", 5),
			("---5", -5),  # 三重负号折叠
			("10 + -5", 5),
			("10 * -2", -20),
			("-(-5)", 5),  # 双重负号
		]
		for expr, expected in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator().execute(expr)
				self.assertTrue(res.is_success, res.error)
				self.assertEqual(res.result, expected)


# ==========================================
# 2. 词法清洗
# ==========================================

class TestTokenizer(unittest.TestCase):
	def test_symbol_replacements(self):
		cases = [
			("1➕1", 2),  # Emoji 加号
			("1➖1", 0),  # Emoji 减号
			("2✖3", 6),  # Emoji 乘号
			("2×3", 6),
			("2x3", 6),  # 字母 x
			("2X3", 6),
			("10➗2", 5),  # Emoji 除号
			("10÷2", 5),
			("（1+1）", 2),  # 中文括号
			("【2+2】", 4),  # 中文方括号
		]
		for expr, expected in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator().execute(expr)
				self.assertTrue(res.is_success, res.error)
				self.assertEqual(res.result, expected)

	def test_whitespace_tolerance(self):
		for expr in ["d6", "d 6", "  1d6  ", "2d 4", "1d6 + 5"]:
			with self.subTest(expr=expr):
				res = DiceSimulator().execute(expr)
				self.assertTrue(res.is_success, res.error)

	def test_lowercase_d(self):
		res = DiceSimulator(rng=FakeRNG([3, 3, 3])).execute('3d6')
		self.assertEqual(res.result, 9)  # 3 次掷出 3
		res = DiceSimulator(rng=FakeRNG([3, 3, 3])).execute('3D6')
		self.assertEqual(res.result, 9)

	def test_illegal_char(self):
		res = DiceSimulator().execute('a')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_illegal_char')


# ==========================================
# 3. 掷骰逻辑（注入确定性 RNG）
# ==========================================

class TestDiceMechanics(unittest.TestCase):
	def test_dice_roll_expands_and_sums(self):
		res = DiceSimulator(rng=FakeRNG([3, 4, 5])).execute('3d6')
		self.assertEqual(res.result, 12)
		self.assertIn('3+4+5', res.steps)  # 展开明细被保留

	def test_default_die_is_d100(self):
		res = DiceSimulator(rng=FakeRNG([7])).execute('d')
		self.assertEqual(res.result, 7)
		self.assertIn('1D100', res.steps)

	def test_explicit_count_and_faces(self):
		res = DiceSimulator(rng=FakeRNG([2, 3])).execute('2D6')
		self.assertEqual(res.result, 5)
		self.assertIn('2+3', res.steps)

	def test_dice_with_expression_params(self):
		# (1+1)d(2+2) -> 2D4，两个骰子各掷出 5
		res = DiceSimulator(rng=FakeRNG([5, 5])).execute('(1+1)d(2+2)')
		self.assertEqual(res.result, 10)
		self.assertIn('2D4', res.steps)  # 就绪骰子步骤被保留

	def test_nested_dice_uses_result_as_count(self):
		# (1d4)d6：1d4=3，再掷 3d6=[1,1,1] -> 3
		res = DiceSimulator(rng=FakeRNG([3, 1, 1, 1])).execute('(1d4)d6')
		self.assertEqual(res.result, 3)
		self.assertIn('(1D4)D6', res.steps)
		self.assertIn('3D6', res.steps)

	def test_chained_d_is_nested(self):
		# d6d6 = (1D6)D6
		res = DiceSimulator(rng=FakeRNG([3, 1, 1, 1])).execute('d6d6')
		self.assertEqual(res.result, 3)
		self.assertIn('(1D6)D6', res.steps)

	def test_zero_dice(self):
		res = DiceSimulator(rng=FakeRNG([])).execute('0d6')
		self.assertTrue(res.is_success)
		self.assertEqual(res.result, 0)
		self.assertEqual(res.steps, ['0D6', '0'])  # 不产生空字符串步骤

	def test_empty_input_rolls_default_die(self):
		res = DiceSimulator(rng=FakeRNG([7])).execute('')
		self.assertEqual(res.result, 7)
		self.assertIn('1D100', res.steps)

	def test_sort_threshold(self):
		# 数量超过 threshold_sort_rolls 时降序展示
		cfg = DiceConfig(threshold_sort_rolls=2, threshold_sum_rolls=100)
		res = DiceSimulator(rng=FakeRNG([3, 1, 2]), config=cfg).execute('3d6')
		self.assertEqual(res.result, 6)
		self.assertIn('3+2+1', res.steps)

	def test_sum_threshold(self):
		# 数量超过 threshold_sum_rolls 时直接求和，不展示明细
		cfg = DiceConfig(threshold_sum_rolls=2)
		res = DiceSimulator(rng=FakeRNG([1, 2, 3]), config=cfg).execute('3d6')
		self.assertEqual(res.result, 6)
		self.assertEqual(res.steps, ['3D6', '6'])


# ==========================================
# 4. 步骤过滤（原"优化"测试的替代）
# ==========================================

class TestStepFiltering(unittest.TestCase):
	"""展示层过滤：内部朴素逐步规约，纯算术中间步不输出。

	旧设计有 cluster / 结合律优化；新设计内部不做合并，靠过滤保证 trace 紧凑。
	"""

	def test_pure_arithmetic_intermediates_filtered(self):
		res = DiceSimulator().execute('1+1+1+1+1')
		self.assertEqual(res.result, 5)
		self.assertEqual(res.steps, ['1+1+1+1+1', '5'])  # 无 "2+1+1+1" 之类中间步

	def test_no_constant_merge_but_trace_compact(self):
		# 原结合律优化用例 (d10+5)+5 -> d10+10：新设计不做常数合并
		res = DiceSimulator(rng=FakeRNG([1])).execute('(d10 + 5) + 5')
		self.assertEqual(res.result, 11)
		self.assertEqual(res.steps, ['1D10+5+5', '1+5+5', '11'])
		self.assertNotIn('1D10+10', res.steps)  # 不做合并
		self.assertNotIn('6+5', res.steps)  # 纯算术中间步被过滤

	def test_roll_expansion_step_kept(self):
		res = DiceSimulator(rng=FakeRNG([1, 2, 3])).execute('3d6')
		self.assertIn('1+2+3', res.steps)

	def test_ready_dice_step_kept(self):
		res = DiceSimulator(rng=FakeRNG([5, 5])).execute('(1+1)d(2+2)')
		self.assertIn('2D4', res.steps)

	def test_final_step_always_kept(self):
		res = DiceSimulator(rng=FakeRNG([1, 1, 1])).execute('3d6+5')
		self.assertEqual(res.result, 8)
		self.assertEqual(res.steps[-1], str(res.result))


# ==========================================
# 5. 边界与限制（实例级 DiceConfig）
# ==========================================

class TestLimitsAndEdges(unittest.TestCase):
	def test_max_dice_number(self):
		cfg = DiceConfig(max_dice_number=10)
		res = DiceSimulator(config=cfg).execute('11d6')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_dice_max')

	def test_max_dice_faces(self):
		cfg = DiceConfig(max_dice_faces=100)
		res = DiceSimulator(config=cfg).execute('1d101')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_face_max')

	def test_negative_dice_params(self):
		res = DiceSimulator().execute('(-1)d6')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_dice_neg')

		res = DiceSimulator().execute('1d(-5)')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_face_min')

	def test_recursion_depth(self):
		deep = '(' * 60 + '1' + ')' * 60
		res = DiceSimulator().execute(deep)
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_depth_limit')

		# 适度嵌套应可解析
		ok = '(' * 8 + 'd6' + ')' * 8
		self.assertTrue(DiceSimulator(rng=FakeRNG([3])).execute(ok).is_success)

	def test_step_limit_circuit_breaker(self):
		cfg = DiceConfig(max_simulation_steps=2)
		res = DiceSimulator(config=cfg, rng=FakeRNG([1] * 10)).execute('1*2+1d6d6')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_steps_limit')

	def test_output_collapse(self):
		cfg = DiceConfig(max_output_length=10)
		res = DiceSimulator(config=cfg, rng=FakeRNG([1] * 10)).execute('10d6')
		self.assertTrue(res.is_success)
		self.assertEqual(res.result, 10)
		for step in res.steps:
			self.assertLessEqual(len(step), cfg.max_output_length + 20)


# ==========================================
# 6. 错误处理
# ==========================================

class TestErrors(unittest.TestCase):
	def test_syntax_errors(self):
		cases = [
			("3d6 5", "err_unparsed"),
			("(1+1", "err_missing_paren"),
			("d6)", "err_unparsed"),
			("d*", "err_missing_atom"),
			("()", "err_invalid_syntax"),
			("a", "err_illegal_char"),
		]
		for expr, err_code in cases:
			with self.subTest(expr=expr):
				res = DiceSimulator().execute(expr)
				self.assertFalse(res.is_success)
				self.assertEqual(res.error['error_code'], err_code)

	def test_div_by_zero(self):
		res = DiceSimulator().execute('10 / 0')
		self.assertFalse(res.is_success)
		self.assertEqual(res.error['error_code'], 'err_div_zero')


class TestErrorCodes(unittest.TestCase):
	"""验证每个可达的错误码都能被真实输入触发。"""

	def _err(self, cfg, expr, rng=None):
		res = DiceSimulator(config=cfg, rng=rng).execute(expr)
		return res.error['error_code'] if res.error else None

	def test_all_reachable_error_codes(self):
		cases = {
			'err_illegal_char': (DiceConfig(), 'a'),
			'err_unparsed': (DiceConfig(), '3d6 5'),
			'err_depth_limit': (DiceConfig(), '(' * 60 + '1' + ')' * 60),
			'err_missing_paren': (DiceConfig(), '(1+1'),
			'err_missing_atom': (DiceConfig(), 'd*'),
			'err_invalid_syntax': (DiceConfig(), '()'),
			'err_dice_neg': (DiceConfig(), '(-1)d6'),
			'err_face_min': (DiceConfig(), '1d(-5)'),
			'err_dice_max': (DiceConfig(max_dice_number=10), '11d6'),
			'err_face_max': (DiceConfig(max_dice_faces=100), '1d101'),
			'err_div_zero': (DiceConfig(), '10/0'),
			'err_steps_limit': (DiceConfig(max_simulation_steps=2),
								'1*2+1d6d6', FakeRNG([1] * 10)),
		}
		for code, args in cases.items():
			with self.subTest(code=code):
				got = self._err(*args)
				self.assertEqual(got, code, f'{args[1]!r} 应触发 {code}，实际 {got}')

	def test_locale_keys_match_reachable_errors(self):
		# locale 文案键必须恰好等于「可达错误码 + err_unknown 兜底」，
		# 防止死错误码或假想键混入
		expected = {
			'err_illegal_char', 'err_unparsed', 'err_depth_limit',
			'err_missing_paren', 'err_missing_atom', 'err_invalid_syntax',
			'err_dice_neg', 'err_face_min', 'err_dice_max', 'err_face_max',
			'err_div_zero', 'err_steps_limit', 'err_unknown',
		}
		for lang in ['zh_CN', 'en_US']:
			keys = set(I18nManager._load(lang))
			self.assertEqual(keys, expected, f'{lang} 的文案键不一致')


class TestContextWindow(unittest.TestCase):
	"""指示位置的错误：context 承载结构化上下文（position/text/widths/arrow_text），
	position 合并进 context、顶层不再带；message 保持纯描述。"""

	def _res(self, expr):
		res = DiceSimulator().execute(expr)
		self.assertIsNotNone(res.error)
		return res.error

	def test_context_on_position_errors(self):
		for expr in ['a', '12+abcde', '12345a6789', '3D（6 5']:
			with self.subTest(expr=expr):
				ctx = self._res(expr)['context']
				self.assertLessEqual(len(ctx['text']), 11, ctx)
				self.assertIsInstance(ctx['position'], int, ctx)
				self.assertEqual(ctx['widths'], 1, ctx)  # 系统字符均为 ASCII 宽 1
				# 箭头行只含空格 + '^'，不带换行（\n 由消费方拼接）
				self.assertNotIn('\n', ctx['arrow_text'], ctx)
				self.assertTrue(ctx['arrow_text'].endswith('^'), ctx)

	def test_message_is_pure_description(self):
		# context 独立成字段，message 只含描述 + 位置数字（不拼上下文窗口）
		res = self._res('12+abcde')
		self.assertNotIn('\n', res['message'])
		self.assertNotIn('^', res['message'])
		self.assertEqual(res['message'],
						 I18nManager.t('err_illegal_char', lang='zh_cn', char='a', pos=3))

	def test_pos_in_template_marks_position_errors(self):
		# 模板里带 {pos} 的错误 = 自带位置参数，一眼可辨哪些有 context
		pos_keys = {'err_illegal_char', 'err_unparsed',
					'err_missing_atom', 'err_missing_paren'}
		for lang in ['zh_CN', 'en_US']:
			with self.subTest(lang=lang):
				with_pos = {k for k, v in I18nManager._load(lang).items() if '{pos}' in v}
				self.assertEqual(with_pos, pos_keys, lang)

	def test_no_context_without_position(self):
		# 无 pos 的错误没有 context 字段
		res = self._res('10/0')
		self.assertNotIn('context', res)

	def test_short_source_no_fake_dots(self):
		# 源串短于窗口时不加 '..'，避免假装还有未完的输入
		ctx = self._res('d*')['context']
		self.assertEqual(ctx['text'], 'd*')
		self.assertEqual(ctx['arrow_text'], '  ^')
		self.assertEqual(ctx['position'], 2)

	def test_near_start_form(self):
		# 出错点靠近开头：源串短于 9 → 原样贴出，无右侧截断点
		ctx = self._res('a')['context']
		self.assertEqual(ctx['text'], 'a')
		self.assertEqual(ctx['arrow_text'], '^')
		self.assertEqual(ctx['position'], 0)

	def test_middle_form(self):
		# 出错点在中部，右侧内容被截掉 → 只有右侧补 '..'
		ctx = self._res('12+abcde')['context']
		self.assertEqual(ctx['text'], '12+abcd..')
		self.assertEqual(ctx['arrow_text'], '   ^')
		self.assertEqual(ctx['position'], 3)

	def test_middle_form_both_truncated(self):
		# 源串足够长，出错点两侧都有内容被截掉 → 两侧各补 '..'
		ctx = self._res('12345a6789')['context']
		self.assertEqual(ctx['text'], '..345a678..')
		self.assertEqual(ctx['arrow_text'], '     ^')

	def test_near_end_form(self):
		# 出错点靠近末尾：源串正好 9 个 → 原样贴出，无左侧截断点
		ctx = self._res('12+34567a')['context']
		self.assertEqual(ctx['text'], '12+34567a')
		self.assertEqual(ctx['arrow_text'], '        ^')

	def test_missing_paren_has_context(self):
		# err_missing_paren 也应带位置：箭头指在应出现 ')' 的游标处
		ctx = self._res('(1+1')['context']
		self.assertEqual(ctx['position'], 4)
		self.assertEqual(ctx['text'], '(1+1')
		self.assertEqual(ctx['arrow_text'], '    ^')

	def test_unparsed_has_context(self):
		ctx = self._res('d6)')['context']
		self.assertEqual(ctx['position'], 2)
		self.assertEqual(ctx['text'], 'd6)')
		self.assertEqual(ctx['arrow_text'], '  ^')

	def test_custom_window_length(self):
		# 窗口长度可自定义（默认 11），小于 5 时钳到 5；'..' 仅在真实截断侧出现
		src = '1234567890ab'
		cases = [
			(3, '..7..'),            # 钳到最小长度 5
			(5, '..7..'),
			(7, '..678..'),
			(11, '..4567890..'),
			(15, '..234567890ab'),   # 长度大于源串 → 只截真实溢出侧
		]
		for length, expected in cases:
			with self.subTest(length=length):
				ctx = build_context(src, 6, length=length)
				self.assertEqual(ctx['text'], expected)
				self.assertLessEqual(len(ctx['text']), max(5, length))  # 钳到最小 5


# ==========================================
# 7. 国际化（实例级 lang）
# ==========================================

class TestI18n(unittest.TestCase):
	def _msg(self, lang, key, **params):
		"""从真实 locale 文件读出模板并格式化作为预期值，不硬编码文案。

		这样改翻译文字不会破坏测试——测试只校验"错误码 + 语言"正确
		路由到文件里的消息。
		"""
		return I18nManager._load(lang)[key].format(**params)

	def test_instance_level_lang(self):
		zh = DiceSimulator(lang='zh_CN').execute('10 / 0')
		en = DiceSimulator(lang='en_US').execute('10 / 0')
		self.assertEqual(zh.error['message'], self._msg('zh_CN', 'err_div_zero'))
		self.assertEqual(en.error['message'], self._msg('en_US', 'err_div_zero'))
		self.assertEqual(zh.error['lang'], 'zh_cn')  # 输出统一小写
		self.assertEqual(en.error['lang'], 'en_us')

	def test_default_lang(self):
		res = DiceSimulator().execute('10 / 0')
		self.assertEqual(res.error['message'], self._msg('zh_CN', 'err_div_zero'))

	def test_parametrized_message_from_file(self):
		# 带占位符的错误：验证 val/limit 参数正确传到文件模板
		cfg = DiceConfig(max_dice_number=10)
		res = DiceSimulator(config=cfg).execute('11d6')
		self.assertEqual(res.error['error_code'], 'err_dice_max')
		self.assertEqual(res.error['message'],
						 self._msg('zh_CN', 'err_dice_max', val=11, limit=10))

	def test_messages_loaded_from_files(self):
		langs = I18nManager.available_langs()
		self.assertIn('zh_cn', langs)
		self.assertIn('en_us', langs)

	def test_unknown_lang_falls_back_to_default(self):
		msg = I18nManager.t('err_div_zero', lang='xx_XX')
		self.assertEqual(msg, self._msg('zh_CN', 'err_div_zero'))

	def test_case_insensitive_lang(self):
		msg = I18nManager.t('err_div_zero', lang='zh_cn')
		self.assertEqual(msg, self._msg('zh_CN', 'err_div_zero'))

	def test_unknown_key_falls_back_to_key(self):
		msg = I18nManager.t('no_such_key', lang='en_US')
		self.assertEqual(msg, 'no_such_key')

	def test_manual_reload(self):
		"""改翻译文件后需手动 reload 才生效。"""
		original_dir = I18nManager._LOCALE_DIR
		try:
			with tempfile.TemporaryDirectory() as tmp:
				I18nManager._LOCALE_DIR = Path(tmp)
				path = Path(tmp) / 'test_XX.json'
				path.write_text(json.dumps({'greet': 'hi'}), encoding='utf-8')
				self.assertEqual(I18nManager.t('greet', lang='test_XX'), 'hi')

				# 文件已改但未 reload → 仍是缓存旧值
				path.write_text(json.dumps({'greet': 'hello'}), encoding='utf-8')
				self.assertEqual(I18nManager.t('greet', lang='test_XX'), 'hi')

				# 手动 reload 后读到新值
				I18nManager.reload('test_XX')
				self.assertEqual(I18nManager.t('greet', lang='test_XX'), 'hello')
		finally:
			I18nManager._LOCALE_DIR = original_dir
			I18nManager._cache.pop('test_XX', None)


# ==========================================
# 8. 可复用 API
# ==========================================

class TestReusableSimulator(unittest.TestCase):
	def test_reuse_one_instance(self):
		sim = DiceSimulator()
		r1 = sim.execute('3d6')
		r2 = sim.execute('1d20+5')
		self.assertTrue(r1.is_success)
		self.assertTrue(r2.is_success)
		self.assertEqual(r1.raw_input, '3d6')
		self.assertEqual(r2.raw_input, '1d20+5')

	def test_default_calls_get_fresh_seeds(self):
		sim = DiceSimulator()
		r1 = sim.execute('3d6')
		r2 = sim.execute('3d6')
		self.assertNotEqual(r1.seed, r2.seed)  # secrets 每次生成新种子

	def test_per_call_seed(self):
		sim = DiceSimulator()
		r1 = sim.execute('3d6', seed=42)
		r2 = sim.execute('3d6', seed=42)
		self.assertEqual(r1.seed, 42)
		self.assertEqual(r2.seed, 42)  # 本次指定的 seed 原样记录
		self.assertEqual(r1.result, r2.result)
		self.assertEqual(r1.steps, r2.steps)

	def test_same_seed_same_first_result(self):
		a = DiceSimulator().execute('3d6', seed=42)
		b = DiceSimulator().execute('3d6', seed=42)
		self.assertEqual(a.result, b.result)
		self.assertEqual(a.steps, b.steps)

	def test_seed_replay(self):
		r1 = DiceSimulator().execute('3d6', seed=42)
		replay = DiceSimulator().execute('3d6', seed=r1.seed)
		self.assertEqual(replay.result, r1.result)
		self.assertEqual(replay.steps, r1.steps)

	def test_execute_takes_expression(self):
		res = DiceSimulator().execute('2d4')
		self.assertEqual(res.raw_input, '2d4')
		self.assertTrue(res.is_success)

	def test_independent_calls_on_same_instance(self):
		sim = DiceSimulator()
		self.assertEqual(sim.execute('2+2').result, 4)
		self.assertEqual(sim.execute('3+3').result, 6)
		self.assertEqual(sim.execute('4+4').result, 8)

	def test_execute_requires_expression(self):
		with self.assertRaises(TypeError):
			DiceSimulator().execute()
		with self.assertRaises(ValueError):
			DiceSimulator().execute(None)


# ==========================================
# 9. DiceResult 结构
# ==========================================

class TestResultStructure(unittest.TestCase):
	def test_success_result_fields(self):
		res = DiceSimulator(rng=FakeRNG([1, 1, 1])).execute('3d6')
		self.assertEqual(res.raw_input, '3d6')
		self.assertIsInstance(res.steps, list)
		self.assertTrue(all(isinstance(s, str) for s in res.steps))
		self.assertIsInstance(res.result, int)
		self.assertIs(res.error, None)
		self.assertTrue(res.is_success)
		self.assertIsInstance(res.seed, int)

	def test_error_result_fields(self):
		res = DiceSimulator().execute('10 / 0')
		self.assertFalse(res.is_success)
		self.assertIsNone(res.result)
		self.assertEqual(res.steps, ['10/0'])  # 首步渲染仍被保留，说明出错位置
		self.assertEqual(res.error['error_code'], 'err_div_zero')
		self.assertIsInstance(res.seed, int)
		self.assertEqual(res.error['lang'], 'zh_cn')  # lang 在 error 内层，小写输出


if __name__ == '__main__':
	unittest.main()
