#!/usr/bin/env pytest
import torch

import torchdynamo.testing
from torchdynamo.testing import same

from . import test_global_declaration


class Pair(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def Foo():
    return Pair(1, 1)


g_counter = 1
g_list = [0, 1, 2]
g_dict = {"a": 0, "b": 1}
g_object = Foo()
g_tensor = torch.zeros(10)


_name: int = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


def reset_name():
    global _name
    _name = 0


class TestGlobals(torchdynamo.testing.TestCase):
    def test_store_global_1(self):
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_2(self):
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        """Wrap the second call with torchdynamo as well"""
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res2 = opt_fn(x)
        self.assertTrue(same(res2 - res1, 2 * torch.ones(10)))

    def test_store_global_new(self):
        def fn(x):
            # Test create a new global
            global g_counter_new
            g_counter_new = x + 1
            return x + g_counter_new

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        self.assertTrue(same(res1, x + x + 1))

    def test_store_global_list(self):
        def fn(x):
            global g_list
            val = x + g_list[1]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_list[1] += 1
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_list_2(self):
        def fn(x):
            global g_list
            val = x + g_list[1]
            g_list = [x + 1 for x in g_list]
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict(self):
        def fn(x):
            global g_dict
            val = x + g_dict["b"]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_dict["b"] += 1
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict_2(self):
        def fn(x):
            global g_dict
            g_dict = {key: value + 1 for key, value in g_dict.items()}
            val = x + g_dict["b"]
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_object(self):
        def fn(x):
            global g_object
            val = x + g_object.y
            g_object.y += 1
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_cross_file(self):
        def fn(x):
            val = x + test_global_declaration.g_tensor_export
            test_global_declaration.g_tensor_export = (
                test_global_declaration.g_tensor_export + 1
            )
            return val

        x = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_inline_1(self):
        # Borrowed from test_python_autograd.py
        class Variable:
            def __init__(self, value: torch.Tensor, name: str = None):
                self.value = value
                self.name = name or fresh_name()

        def fn(a, b):
            a = Variable(a)
            b = Variable(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()

    def test_store_global_inline_2(self):
        # Borrowed from test_python_autograd.py
        class Variable:
            def __init__(self, value: torch.Tensor, name: str = None):
                self.value = value
                self.name = name or fresh_name()

            @staticmethod
            def constant(value: torch.Tensor, name: str = None):
                return Variable(value, name)

        def fn(a, b):
            a = Variable.constant(a)
            b = Variable.constant(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torchdynamo.testing.CompileCounter()
        opt_fn = torchdynamo.optimize(cnts)(fn)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()
