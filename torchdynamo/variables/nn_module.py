import inspect
import itertools
import re
from typing import Dict
from typing import List

import torch.nn

from .. import skipfiles
from .. import variables
from ..allowed_functions import is_allowed
from ..guards import GuardBuilder
from ..source import AttrSource
from ..source import GetItemSource
from ..source import NNModuleSource
from ..utils import proxy_args_kwargs
from ..utils import unimplemented
from .base import MutableLocal
from .base import VariableTracker
from .base import typestr


class NNModuleVariable(VariableTracker):
    def __init__(self, module_type: type, module_key: str, **kwargs):
        super(NNModuleVariable, self).__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        assert self.source

    def python_type(self):
        return self.module_type

    def unpack_var_sequence(self, tx):
        # implement list/iter/tuple/etc calls
        key = self.module_key
        base = tx.output.get_submodule(self.module_key)
        options = VariableTracker.propagate([self])
        assert isinstance(
            base, (torch.nn.ModuleList, torch.nn.ParameterList, torch.nn.Sequential)
        ), typestr(base)
        assert self.source
        return [
            tx.output.add_submodule(
                submod, key, idx, source=GetItemSource(self.source, idx), **options
            )
            for idx, submod in enumerate(base)
        ]

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        options = VariableTracker.propagate(self)
        mod = tx.output.get_submodule(self.module_key)
        result = hasattr(mod, name)
        return variables.ConstantVariable(result, **options).add_guard(
            NNModuleSource(AttrSource(self.source, name)).create_guard(
                GuardBuilder.HASATTR
            )
        )

    def is_training(self, tx):
        mod = tx.output.get_submodule(self.module_key)
        return getattr(mod, "training", False)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        mod = tx.output.get_submodule(self.module_key)
        if (
            isinstance(mod, torch.nn.Sequential)
            and mod.__class__.forward is torch.nn.Sequential.forward
        ):
            # unroll Sequential()
            assert not kwargs
            (arg,) = args
            for idx, submod in enumerate(mod):
                tx.call_function(
                    tx.output.add_submodule(
                        submod,
                        self.module_key,
                        idx,
                        source=NNModuleSource(GetItemSource(self.source, idx)),
                        **options,
                    ),
                    [arg],
                    {},
                )
                arg = tx.pop()
            return arg
        elif is_allowed(mod.__class__):
            return variables.TensorVariable.create(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_module",
                    self.module_key,
                    *proxy_args_kwargs(args, kwargs),
                ),
                nnmodule=mod,
                **options,
            )
        else:
            forward = mod.__class__.forward
            assert forward is not torch.nn.Module.forward
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(forward, **options),
                [self] + args,
                kwargs,
            )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable
        from . import ListIteratorVariable
        from . import TupleVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        key = self.module_key
        module = tx.output.get_submodule(key)

        if name == "forward":
            return self.call_function(tx, args, kwargs)

        if name == "_check_input_dim" and skipfiles.is_torch_nn(
            inspect.getfile(module.__class__._check_input_dim)
        ):
            return ConstantVariable(True, **options)

        if not all(x.is_python_constant() for x in itertools.chain(args, kwargs)):
            raise unimplemented(f"non-const NNModule method {name}")

        def get_kwargs(*names):
            fn = getattr(module, name)
            bound_args = inspect.signature(fn).bind(
                *([x.as_python_constant() for x in args]),
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            bound_args.apply_defaults()
            bound_args = bound_args.arguments
            return {k: bound_args[k] for k in names}

        def wrap_values(items, getsource=AttrSource):
            result = []
            for name, submod in items:
                # layer.0.foo => layer[0].foo
                name = re.sub(r"[.]([0-9]+)([.]|$)", r"[\1]\2", name)
                src = NNModuleSource(getsource(self.source, name))
                result.append(
                    tx.output.add_submodule(
                        submod,
                        key,
                        name,
                        source=src,
                        **options,
                    )
                )
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)

        if name == "children":
            assert not (args or kwargs)
            return wrap_values(module.named_children())
        elif name == "parameters":
            return wrap_values(module.named_parameters(**get_kwargs("recurse")))
        elif name == "values":
            assert not (args or kwargs)
            return wrap_values(module.items(), GetItemSource)
        elif name == "items":
            assert not (args or kwargs)
            result = []
            for name, submod in module.items():
                result.append(
                    TupleVariable(
                        [
                            ConstantVariable(name, **options),
                            tx.output.add_submodule(
                                submod,
                                key,
                                name,
                                source=NNModuleSource(GetItemSource(self.source, name)),
                                **options,
                            ),
                        ]
                    )
                )
            return ListIteratorVariable(result, mutable_local=MutableLocal(), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable(len(module), **options)
        elif name == "__getitem__":
            assert not kwargs and len(args) == 1
            assert type(module).__getitem__ in (
                torch.nn.ModuleList.__getitem__,
                torch.nn.ParameterList.__getitem__,
            ), typestr(module)
            assert self.source
            key = args[0].as_python_constant()
            submod = module[key]
            return tx.output.add_submodule(
                submod,
                key,
                args[0].as_python_constant(),
                source=NNModuleSource(GetItemSource(self.source, key)),
                **options,
            )
        else:
            return super().call_method(tx, name, args, kwargs)
