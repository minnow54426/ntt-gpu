"""Synthetic parser checks; actual PTX comes from the pinned compiler gate."""
import unittest
from check_barrett_ptx import validate_ptx


class PtxTests(unittest.TestCase):
    def test_scopes_the_check_to_the_barrett_entry(self):
        source = '''.target sm_120
// .entry fake_barrett_kernel() {div.u64;}
.visible .entry other() { div.u64 %x, %a, %b; }
.visible .entry _Zbarrett_kernel(.param .u64 args) .maxntid 256,1,1 {
 { .reg .u64 x; mul.hi.u64 %x, %a, %b; }
 // rem.u64 in this comment is not an instruction
 ret;
}'''
        self.assertEqual(validate_ptx(source), ('sm_120', 1))

    def test_rejects_unsafe_or_missing_instructions(self):
        for body in ['div.u64 %x,%a,%b;', 'rem.u64 %x,%a,%b;', 'call.uni helper;', 'ret;']:
            prefix = '' if body == 'ret;' else 'mul.hi.u64 %x,%a,%b;'
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_ptx('.target sm_89\n.entry barrett_kernel() {'+prefix+body+'}')
        with self.assertRaises(ValueError):validate_ptx('.target sm_89\n.entry barrett_kernel() {')
