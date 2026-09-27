import re
import unittest

from json_to_gbnf import SchemaConverter


def grammar_regex(schema):
    converter = SchemaConverter(
        prop_order={},
        allow_fetch=False,
        dotall=False,
        raw_pattern=False,
    )
    converter.visit(schema, '')
    line = next(
        line for line in converter.format_grammar().splitlines()
        if line.startswith('root ::= ')
    )
    body = line[len('root ::= '):]
    return re.compile(_gbnf_to_regex(body))


def _gbnf_to_regex(gbnf):
    out = []
    i = 0
    while i < len(gbnf):
        c = gbnf[i]
        if c == '"':
            i += 1
            literal = []
            while i < len(gbnf) and gbnf[i] != '"':
                if gbnf[i] == '\\' and i + 1 < len(gbnf):
                    literal.append(gbnf[i + 1])
                    i += 2
                else:
                    literal.append(gbnf[i])
                    i += 1
            i += 1
            out.append(re.escape(''.join(literal)))
        elif c in ' \n\t':
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


class IntegerMinimumTests(unittest.TestCase):
    def test_minimum_100_does_not_accept_10(self):
        rx = grammar_regex({'type': 'integer', 'minimum': 100})
        self.assertIsNone(rx.fullmatch('10'))
        self.assertIsNone(rx.fullmatch('19'))
        self.assertIsNone(rx.fullmatch('99'))
        self.assertIsNotNone(rx.fullmatch('100'))
        self.assertIsNotNone(rx.fullmatch('101'))
        self.assertIsNotNone(rx.fullmatch('999'))
        self.assertIsNotNone(rx.fullmatch('1000'))

    def test_minimum_200_does_not_accept_20(self):
        rx = grammar_regex({'type': 'integer', 'minimum': 200})
        self.assertIsNone(rx.fullmatch('20'))
        self.assertIsNone(rx.fullmatch('199'))
        self.assertIsNotNone(rx.fullmatch('200'))
        self.assertIsNotNone(rx.fullmatch('1000'))

    def test_minimum_1000_does_not_accept_100(self):
        rx = grammar_regex({'type': 'integer', 'minimum': 1000})
        self.assertIsNone(rx.fullmatch('10'))
        self.assertIsNone(rx.fullmatch('100'))
        self.assertIsNone(rx.fullmatch('999'))
        self.assertIsNotNone(rx.fullmatch('1000'))

    def test_minimum_101_does_not_accept_100(self):
        rx = grammar_regex({'type': 'integer', 'minimum': 101})
        self.assertIsNone(rx.fullmatch('100'))
        self.assertIsNotNone(rx.fullmatch('101'))
        self.assertIsNotNone(rx.fullmatch('1000'))

    def test_minimum_10_still_accepts_10(self):
        rx = grammar_regex({'type': 'integer', 'minimum': 10})
        self.assertIsNone(rx.fullmatch('9'))
        self.assertIsNotNone(rx.fullmatch('10'))
        self.assertIsNotNone(rx.fullmatch('100'))

    def test_closed_range_and_negative_bounds(self):
        closed = grammar_regex({'type': 'integer', 'minimum': 8, 'maximum': 12})
        self.assertIsNone(closed.fullmatch('7'))
        self.assertIsNotNone(closed.fullmatch('8'))
        self.assertIsNotNone(closed.fullmatch('12'))
        self.assertIsNone(closed.fullmatch('13'))

        signed = grammar_regex({'type': 'integer', 'minimum': -5, 'maximum': 5})
        self.assertIsNone(signed.fullmatch('-6'))
        self.assertIsNotNone(signed.fullmatch('-5'))
        self.assertIsNotNone(signed.fullmatch('0'))
        self.assertIsNotNone(signed.fullmatch('5'))
        self.assertIsNone(signed.fullmatch('6'))


if __name__ == '__main__':
    unittest.main()
