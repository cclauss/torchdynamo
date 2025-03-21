import collections
import dataclasses
import enum
import functools
import inspect
import re
import types
from abc import ABCMeta
from typing import Any
from typing import List

import numpy as np
import torch

import torchdynamo

from .. import config
from .. import mutation_guard
from .. import skipfiles
from ..allowed_functions import is_allowed
from ..allowed_functions import is_builtin
from ..allowed_functions import is_numpy
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..side_effects import SideEffects
from ..source import AttrSource
from ..source import GetItemSource
from ..source import GlobalSource
from ..source import GlobalWeakRefSource
from ..source import RandomValueSource
from ..source import Source
from ..source import TupleIteratorGetItemSource
from ..utils import getfile
from ..utils import global_key_name
from ..utils import is_namedtuple
from ..utils import is_numpy_int_type
from ..utils import istensor
from ..utils import istype
from ..utils import tuple_iterator
from ..utils import tuple_iterator_getitem
from ..utils import tuple_iterator_len
from .base import MutableLocal
from .builtin import BuiltinVariable
from .constant import ConstantVariable
from .constant import EnumVariable
from .dicts import ConstDictVariable
from .dicts import DataClassVariable
from .dicts import DefaultDictVariable
from .dicts import HFPretrainedConfigVariable
from .functions import UserFunctionVariable
from .lists import ListIteratorVariable
from .lists import ListVariable
from .lists import NamedTupleVariable
from .lists import RangeVariable
from .lists import SliceVariable
from .lists import TupleVariable
from .misc import AutogradFunctionVariable
from .misc import InspectSignatureVariable
from .misc import LambdaVariable
from .misc import NumpyVariable
from .misc import PythonModuleVariable
from .misc import SkipFilesVariable
from .misc import TypingVariable
from .nn_module import UnspecializedNNModuleVariable
from .tensor import TensorVariable
from .tensor import TensorWithTFOverrideVariable
from .tensor import UnspecializedNumpyVariable
from .tensor import UnspecializedPythonVariable
from .torch import TorchVariable
from .user_defined import UserDefinedClassVariable
from .user_defined import UserDefinedObjectVariable


@dataclasses.dataclass
class GraphArg:
    source: Source
    example: Any
    is_unspecialized: bool

    def load(self, tx):
        return self.source.reconstruct(tx)

    def get_examples(self):
        return [self.example]

    def __len__(self):
        return 1

    def erase(self):
        self.example = None


class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(
        self,
        tx: "torchdynamo.symbolic_convert.InstructionTranslatorBase",
        source: Source,
    ):
        super(VariableBuilder, self).__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        if value in self.tx.output.side_effects:
            # TODO(jansel): add guard for alias relationship
            return self.tx.output.side_effects[value]
        return self._wrap(value).clone(**self.options())

    @staticmethod
    @functools.lru_cache(None)
    def _common_constants():
        return set(range(17)).union(
            {
                20,
                30,
                40,
                32,
                64,
                96,
                128,
                144,
                240,
                256,
                672,
                1024,
                2048,
                4096,
                0.1,
                0.01,
                0.001,
                0.5,
                0.05,
                800,
                1.873536229133606,
                4.135166556742356,  # Work around for vision_maskrcnn where torch.clamp can't be on different devices
            }
        )

    @staticmethod
    def list_type(value):
        if is_namedtuple(value):
            return functools.partial(NamedTupleVariable, tuple_cls=type(value))
        return {
            tuple: TupleVariable,
            list: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
        }[type(value)]

    def get_source(self):
        return self.source

    def options(self):
        return {"source": self.get_source()}

    def make_guards(self, *guards):
        source = self.get_source()
        return {source.create_guard(guard) for guard in guards}

    def _wrap(self, value):
        make_guards = self.make_guards
        if istensor(value):
            return self.wrap_tensor(value)
        elif istype(value, (tuple, list)) or is_namedtuple(value):
            # One can index a tensor with a list/tuple. Therefore, we need to
            # have a stricter match.
            if istype(value, (tuple, list)) and all(
                [isinstance(x, int) or is_numpy_int_type(x) or x is None for x in value]
            ):
                guards = self.make_guards(GuardBuilder.EQUALS_MATCH)
            else:
                guards = self.make_guards(GuardBuilder.LIST_LENGTH)
            output = [
                VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(
                    item
                ).add_guards(guards)
                for i, item in enumerate(value)
            ]
            result = self.list_type(value)(output, guards=guards)
            if istype(value, list):
                return self.tx.output.side_effects.track_list(
                    self.source, value, result
                )
            return result
        elif istype(value, tuple_iterator):
            guards = self.make_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
            output = [
                VariableBuilder(
                    self.tx, TupleIteratorGetItemSource(self.get_source(), i)
                )(tuple_iterator_getitem(value, i)).add_guards(guards)
                for i in range(tuple_iterator_len(value))
            ]
            return ListIteratorVariable(
                output, mutable_local=MutableLocal(), guards=guards
            )
        elif istype(value, range):
            guards = self.make_guards(GuardBuilder.EQUALS_MATCH)
            return RangeVariable(value=value, guards=guards)
        elif istype(
            value, (dict, collections.defaultdict, collections.OrderedDict)
        ) and all(
            map(
                lambda k: ConstantVariable.is_literal(k)
                or isinstance(k, torch.nn.Parameter),
                value.keys(),
            )
        ):
            guards = self.make_guards(GuardBuilder.DICT_KEYS)

            # store key variables in global location for reconstruction
            for key in value.keys():
                if isinstance(key, torch.nn.Parameter):
                    self.tx.store_dict_key(global_key_name(key), key)

            def index_source(key):
                if isinstance(key, torch.nn.Parameter):
                    return GlobalWeakRefSource(global_key_name(key))
                else:
                    return key

            result = dict(
                [
                    (
                        k,
                        VariableBuilder(
                            self.tx, GetItemSource(self.get_source(), index_source(k))
                        )(value[k]).add_guards(guards),
                    )
                    for k in value.keys()
                ]
            )

            if istype(value, collections.defaultdict):
                result = DefaultDictVariable(
                    result, type(value), value.default_factory, guards=guards
                )
            else:
                result = ConstDictVariable(result, type(value), guards=guards)

            return self.tx.output.side_effects.track_dict(self.source, value, result)
        elif isinstance(value, torch.nn.Module):
            if mutation_guard.is_dynamic_nn_module(value):
                # created dynamically, don't specialize on it
                result = UnspecializedNNModuleVariable(
                    value, guards=make_guards(GuardBuilder.TYPE_MATCH)
                )
                if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                    # don't allow STORE_ATTR mutation with custom __setattr__
                    return result
                return self.tx.output.side_effects.track_object_existing(
                    self.source, value, result
                )
            else:
                return self.tx.output.add_submodule(
                    value,
                    self.name,
                    source=self.get_source(),
                    # Guards are added inside add_submodule
                )
        elif ConstantVariable.is_literal(value) or istype(
            value, (torch.Size, torch.device, torch.dtype)
        ):
            if type(value) in (int, float) and not config.specialize_int_float:
                # unspecializing int/float by default, but still
                # specialize for the following conditions
                if (
                    value in self._common_constants()
                    or isinstance(self.source, GlobalSource)
                    or isinstance(self.source, GetItemSource)
                    or (
                        isinstance(self.source, AttrSource)
                        and isinstance(self.source.base, GlobalSource)
                    )
                ):
                    return ConstantVariable(
                        value=value,
                        guards=make_guards(GuardBuilder.CONSTANT_MATCH),
                    )
                else:
                    return self.wrap_unspecialized_primitive(value)
            else:
                return ConstantVariable(
                    value=value,
                    guards=make_guards(GuardBuilder.CONSTANT_MATCH),
                )
        elif isinstance(value, frozenset) and (
            all(is_allowed(x) or ConstantVariable.is_literal(x) for x in value)
        ):
            # For frozenset, we can guard by object ID instead of value
            # equality, this allows us to handle non-literal values
            return ConstantVariable(
                value=value,
                guards=make_guards(GuardBuilder.ID_MATCH),
            )
        elif isinstance(value, enum.Enum):
            return EnumVariable(
                value=value,
                guards=make_guards(GuardBuilder.ID_MATCH),
            )
        elif is_builtin(value):
            return BuiltinVariable(
                value,
                guards=make_guards(GuardBuilder.BUILTIN_MATCH),
            )
        elif is_allowed(value):
            return TorchVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif value is List:
            return TypingVariable(
                value,
                guards=make_guards(GuardBuilder.ID_MATCH),
            )
        elif value is inspect.signature:
            return LambdaVariable(
                InspectSignatureVariable.create,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif value is dataclasses.fields:
            return LambdaVariable(
                _dataclasses_fields_lambda,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif is_numpy(value):
            return NumpyVariable(
                value,
                guards=make_guards(
                    GuardBuilder.FUNCTION_MATCH
                    if callable(value)
                    else GuardBuilder.TYPE_MATCH
                ),
            )
        elif (
            istype(value, (type, types.FunctionType))
            and skipfiles.check(getfile(value), allow_torch=True)
            and not inspect.getattr_static(value, "_torchdynamo_inline", False)
        ):
            return SkipFilesVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif istype(value, (type, ABCMeta)):
            # TODO(whc) the following seems preferable but breaks some tests, debug
            # elif inspect.isclass(value):
            return UserDefinedClassVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif istype(value, types.FunctionType):
            return UserFunctionVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, types.ModuleType):
            return PythonModuleVariable(
                value,
                guards=make_guards(GuardBuilder.PYMODULE_MATCH),
            )
        elif type(value) is torch.autograd.function.FunctionMeta:
            return AutogradFunctionVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif isinstance(value, (int, float, np.number)):
            return self.wrap_unspecialized_primitive(value)
        elif DataClassVariable.is_matching_object(value):
            return DataClassVariable.wrap(self, value).add_guards(
                make_guards(GuardBuilder.TYPE_MATCH)
            )
        elif HFPretrainedConfigVariable.is_matching_object(value):
            return HFPretrainedConfigVariable(
                value, guards=make_guards(GuardBuilder.TYPE_MATCH)
            )
        elif isinstance(value, slice):
            items = [
                VariableBuilder(self.tx, AttrSource(self.get_source(), k))(
                    getattr(value, k)
                )
                for k in ("start", "stop", "step")
            ]
            return SliceVariable(items, guards=make_guards(GuardBuilder.TYPE_MATCH))
        else:
            result = UserDefinedObjectVariable(
                value,
                guards=self.make_guards(GuardBuilder.TYPE_MATCH),
            )
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(
                self.source, value, result
            )

    def wrap_tensor(self, value: torch.Tensor):
        if self.get_source().guard_source().is_nn_module():
            return self.tx.output.add_submodule(
                value,
                self.name,
                source=self.get_source(),
                # Guards are done inside add_submodule
                # guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            )
        else:
            self.tx.output.graphargs.append(GraphArg(self.get_source(), value, False))
            # Disable __torch_function__ to prevent cloning of `value` to hit
            # user code.
            with torch._C.DisableTorchFunction():
                tensor_variable = TensorVariable.create(
                    tx=self.tx,
                    proxy=self.tx.output.create_graph_input(
                        re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(value)
                    ),
                    example_value=value,
                    guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
                )
            if torch.overrides.has_torch_function_unary(value):
                subclass_torch_function__func = value.__torch_function__.__func__
                subclass_type = type(value)
                return TensorWithTFOverrideVariable(
                    tensor_variable,
                    self.get_source(),
                    subclass_torch_function__func,
                    subclass_type,
                )
            return tensor_variable

    def wrap_unspecialized_primitive(self, value):
        if self.name in self.tx.output.unspec_variable_map:
            return self.tx.output.unspec_variable_map[self.name]
        else:
            wrapped_value = torch.tensor(value)
            self.tx.output.graphargs.append(
                GraphArg(self.get_source(), wrapped_value, True)
            )
            if not isinstance(self.get_source(), RandomValueSource):
                guards = {self.get_source().create_guard(GuardBuilder.TYPE_MATCH, True)}
                options = {"guards": guards}
            else:
                options = {}
            options.update({"source": self.get_source()})
            options.update({"raw_value": value})

            proxy = self.tx.output.create_graph_input(
                re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(wrapped_value)
            )

            if isinstance(value, np.number):
                unspec_var = UnspecializedNumpyVariable.create(
                    tx=self.tx,
                    proxy=proxy,
                    example_value=wrapped_value,
                    **options,
                )
            else:
                unspec_var = UnspecializedPythonVariable.create(
                    tx=self.tx,
                    proxy=proxy,
                    example_value=wrapped_value,
                    **options,
                )
            self.tx.output.unspec_variable_map[self.name] = unspec_var
            return unspec_var


def _dataclasses_fields_lambda(obj):
    if isinstance(obj, UserDefinedObjectVariable):
        value = obj.value
    elif isinstance(obj, DataClassVariable):
        value = obj.user_cls
    else:
        unimplemented(f"Dataclass fields handling fails for type {obj}")
    items = []
    for field in dataclasses.fields(value):
        source = None
        if obj.source:
            source = GetItemSource(
                AttrSource(obj.source, "__dataclass_fields__"), field.name
            )
        items.append(UserDefinedObjectVariable(field, source=source).add_options(obj))
    return TupleVariable(items).add_options(obj)
