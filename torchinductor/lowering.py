import functools
import itertools
import logging
import operator
from collections.abc import Iterable
from typing import List

import sympy
import torch
import torch.fx
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common import Number
from torch._prims_common import elementwise_dtypes
from torch._prims_common import is_boolean_dtype
from torch._prims_common import is_integer_dtype

from . import config
from . import ir
from . import overrides
from .decomposition import decompositions
from .decomposition import get_decompositions
from .ir import ExpandView
from .ir import PermuteView
from .ir import Pointwise
from .ir import Reduction
from .ir import SqueezeView
from .ir import TensorBox
from .ir import View
from .utils import ceildiv
from .utils import has_torchvision_roi_align
from .utils import sympy_product
from .virtualized import V
from .virtualized import ops

log = logging.getLogger(__name__)
lowerings = {}
fallbacks = set()
aten = torch.ops.aten
prims = torch.ops.prims
needs_realized_inputs = set()


def add_needs_realized_inputs(fn):
    if isinstance(fn, (list, tuple, set)):
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            needs_realized_inputs.add(getattr(fn, overload))


add_needs_realized_inputs(
    [
        aten.as_strided,
        aten.avg_pool2d,
        aten.avg_pool2d_backward,
        aten.bmm,
        aten.convolution,
        aten.convolution_backward,
        aten.grid_sampler_2d,
        aten.max_pool2d_with_indices,
        aten.max_pool2d_with_indices_backward,
        aten.mm,
        aten.upsample_bilinear2d,
        aten.upsample_nearest2d,
    ]
)

# TODO(jansel): ezyang says we won't need this in the future, try removing it
# based on https://github.com/pytorch/pytorch/blob/9e3eb329df8f701/c10/core/ScalarType.h#L28
DTYPE_ID_LOOKUP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex32,
    11: torch.bool,
    15: torch.bfloat16,
    # TODO(jansel): add quantized types?
    #  _(c10::qint8, QInt8) /* 12 */
    # _(c10::quint8, QUInt8) /* 13 */
    # _(c10::qint32, QInt32) /* 14 */
    # _(c10::quint4x2, QUInt4x2) /* 16 */
    # _(c10::quint2x4, QUInt2x4) /* 17 */
}


def decode_dtype(dtype: int):
    if not isinstance(dtype, int):
        return dtype
    assert dtype in DTYPE_ID_LOOKUP, f"id {dtype} missing from DTYPE_ID_LOOKUP"
    dtype = DTYPE_ID_LOOKUP[dtype]
    return dtype


def is_integer_type(x):
    if isinstance(x, TensorBox):
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, int)


def is_boolean_type(x):
    if isinstance(x, TensorBox):
        return is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, bool)


def decode_device(device):
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", index=torch.cuda.current_device())
    return device


def get_promoted_dtype(*args):
    def construct_input(inp):
        if isinstance(inp, Number):
            return inp
        else:
            assert hasattr(inp, "get_dtype")
            dim = len(inp.get_size())
            # construct a tmp tensor to feed into torch.result_type
            return torch.zeros([1] * dim, dtype=inp.get_dtype())

    inps = [construct_input(arg) for arg in args]
    _, dtype = elementwise_dtypes(
        *inps, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    return dtype


def _register_lowering(aten_fn, decomp_fn, broadcast, type_promote):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promote: True to apply type promotion to tensor inputs
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args = list(args)
        # Only look at args that are Tensors
        indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
        # kwargs tensors not supported yet
        assert not any(isinstance(x, TensorBox) for x in kwargs.values())

        if type_promote and indices:
            # FIXME that's a crude approximation for promoting args
            promoting_args = [
                a for a in args if isinstance(a, Number) or hasattr(a, "get_dtype")
            ]
            dtype = get_promoted_dtype(*promoting_args)
            for i in indices:
                args[i] = to_dtype(args[i], dtype)
            for i in range(len(args)):
                if isinstance(args[i], ir.Constant):
                    args[i] = ir.Constant(
                        args[i].value, dtype, args[indices[0]].get_device()
                    )

        if broadcast and indices:
            for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
                args[i] = x
            for i in range(len(args)):
                if isinstance(args[i], ir.Constant):
                    args[i] = ExpandView.create(
                        args[i], list(args[indices[0]].get_size())
                    )

        return decomp_fn(*args, **kwargs)

    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    lowerings.update({fn: wrapped for fn in aten_fn})
    return wrapped


def register_lowering(aten_fn, broadcast=False, type_promote=True):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering, aten_fn, broadcast=broadcast, type_promote=type_promote
    )


def broadcast_symbolic_shapes(a, b):
    """
    Broadcasting logic based on symbolic shapes.

    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    output = []
    for a, b in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.Integer(1)
    ):
        if b == 1:
            output.append(a)
        elif a == 1:
            output.append(b)
        else:
            V.graph.sizevars.guard_equals(a, b)
            if len(sympy.expand(b).free_symbols) < len(sympy.expand(a).free_symbols):
                output.append(b)  # prefer shorter formula
            else:
                output.append(a)
    return tuple(reversed(output))


def promote_constants(inputs):
    if not any(isinstance(x, (int, float)) for x in inputs):
        return inputs
    ex = next(x for x in inputs if isinstance(x, TensorBox))
    return [
        (
            ExpandView.create(
                ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())
            )
            if isinstance(x, (int, float))
            else x
        )
        for x in inputs
    ]


def make_pointwise(fn, override_dtype=None, override_device=None, override_bool=None):
    def inner(*inputs: List[TensorBox]):
        inputs = promote_constants(inputs)
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_dtype or inputs[0].get_dtype()

        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_bool is not None:
                return override_bool(*[load(index) for load in loaders])
            else:
                return fn(*[load(index) for load in loaders])

        return Pointwise.create(
            device=override_device or inputs[0].get_device(),
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner


@register_lowering(prims.convert_element_type, type_promote=False)
def to_dtype(x: TensorBox, dtype: torch.dtype):
    if x.get_dtype() == dtype:
        return x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_dtype=dtype)(x)


def to_device(x: TensorBox, device: torch.device):
    device = decode_device(device)
    if x.get_device() == device:
        return x
    return TensorBox.create(ir.DeviceCopy.create(x, device))


@register_lowering(aten._to_copy)
def _to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    assert not layout or layout == torch.strided, "TODO"
    assert not pin_memory, "TODO"
    assert not memory_format, "TODO"
    if device:
        device = decode_device(device)
    if device is not None and device != x.get_device():
        if dtype is not None and device.type == "cpu":
            # CPU can do fewer type conversions
            x = to_dtype(x, decode_dtype(dtype))
        x = to_device(x, device)
    if dtype is not None:
        x = to_dtype(x, decode_dtype(dtype))
    return x


@register_lowering(aten.to)
def to(
    x,
    device_or_dtype=None,
    non_blocking=False,
    copy=False,
    memory_format=None,
    device=None,
    dtype=None,
    layout=None,
):
    assert not memory_format, "TODO"
    assert layout in (None, torch.strided)
    if isinstance(device_or_dtype, torch.dtype):
        return to_dtype(x, device_or_dtype)
    elif isinstance(device_or_dtype, torch.device):
        return to_device(x, device_or_dtype)
    else:
        assert device_or_dtype is None, device_or_dtype

    if device is not None:
        x = to_device(x, device)
    if dtype is not None:
        x = to_dtype(x, dtype)
    return x


def ops_wrapper(name):
    assert isinstance(name, str)

    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promote=True,
    override_dtype=None,
    override_device=None,
    override_bool=None,
):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)
    if override_bool is not None:
        override_bool = ops_wrapper(override_bool)

    fn = make_pointwise(
        fn,
        override_dtype=override_dtype,
        override_device=override_device,
        override_bool=override_bool,
    )
    fn = register_lowering(aten_fn, broadcast=broadcast, type_promote=type_promote)(fn)

    if hasattr(prims, name):
        register_lowering(getattr(prims, name), type_promote=False)(fn)
    return fn


@register_lowering(aten.where, broadcast=True, type_promote=False)
def where(cond, a, b):
    def fn(*args):
        return ops.where(*args)

    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)

    dtype = torch.promote_types(a.get_dtype(), b.get_dtype())
    return make_pointwise(fn, override_dtype=dtype)(
        cond, to_dtype(a, dtype), to_dtype(b, dtype)
    )


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promote=False)
def broadcast_tensors(*inputs):
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
    target = functools.reduce(
        broadcast_symbolic_shapes, [x.get_size() for x in inputs], ()
    )
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        if len(sizes) != len(target) or any(
            ((a == 1 and b != 1) or (a != 1 and b == 1)) for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift])
def nop(x):
    return x  # AOT autograd handles this for us


if hasattr(aten, "lift_fresh"):
    register_lowering(aten.lift_fresh)(nop)


@register_lowering(aten.squeeze, type_promote=False)
def squeeze(x, dim=None):
    assert isinstance(x, TensorBox)
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))

    dim = _validate_dim(x, dim, 0)
    new_shape = list(x.get_size())
    removed = new_shape.pop(dim)
    assert removed == 1, removed
    return view(x, new_shape)


@register_lowering([aten.squeeze_])
def squeeze_(x, dim=None):
    val = squeeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x


@register_lowering(aten.expand, type_promote=False)
def expand(x, sizes):
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    if tuple(x.get_size()) == tuple(sizes):
        return x
    x.mark_reuse(
        V.graph.sizevars.size_hint(sympy_product(sizes) / sympy_product(x.get_size()))
    )
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(prims.broadcast_in_dim, type_promote=False)
def broadcast_in_dim(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)

    return expand(v, shape)


@register_lowering(aten.expand_as, type_promote=False)
def expand_as(x, y):
    return expand(x, y.get_size())


@register_lowering(aten.repeat)
def repeat(x, repeats):
    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.Integer(1)] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    assert len(repeats) == len(x.get_size())

    new_size = list(x.get_size())

    for i in range(len(repeats)):
        assert repeats[i] >= 1
        if repeats[i] > 1:
            new_size[i] = new_size[i] * repeats[i]

    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return expand(x, new_size)

    def inner_fn(index):
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] > 1:
                if old_size[i] == 1:
                    index[i] = sympy.Integer(0)
                else:
                    index[i] = ir.ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    x.mark_reuse(
        V.graph.sizevars.size_hint(sympy_product(new_size) / sympy_product(old_size))
    )
    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
    )


@register_lowering(aten._unsafe_view, type_promote=False)
@register_lowering(aten.view, type_promote=False)
@register_lowering(aten.reshape, type_promote=False)
def view(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(View.create(x.data, sizes))


@register_lowering(aten.permute, type_promote=False)
def permute(x, dims):
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


@register_lowering(aten.slice, type_promote=False)
def slice_(x, dim, start, end, step=1):
    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step))


@register_lowering(aten.roll, type_promote=False)
def roll(a, shifts, dims=tuple()):
    """
    This is based on torch._refs.roll(), but uses ir.ModularIndexing().

    We can't use the ref here because it is based on multiple calls to
    torch.cat() that this will result in terrible code.
    """
    # ATen specifies int[1] type for shifts and dims which expands integers to tuples of length 1
    if not isinstance(shifts, Iterable):
        shifts = (shifts,)
    if not isinstance(dims, Iterable):
        dims = (dims,)
    dims = [_validate_dim(a, d) for d in dims]

    if sympy_product(a.get_size()) == 0:
        return clone(a)

    len_shifts = len(shifts)
    len_dims = len(dims)
    if len_shifts != 1 or len_dims != 1:
        if len_shifts == 0:
            raise RuntimeError("`shifts` required")
        # Takes care of the case when dims is not specified (default)
        # By default, the tensor is flattened before shifting, after which the original shape is restored
        if len_dims == 0 and len_shifts == 1:
            flat = view(a, [sympy_product(a.get_size())])
            rolled = roll(flat, shifts, 0)
            return view(rolled, list(a.get_size()))
        if len_shifts != len_dims:
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}"
            )
        tail_shifts = shifts[1:]
        tail_dims = dims[1:]
        first_dim_rolled = roll(a, shifts[0], dims[0])
        return roll(first_dim_rolled, tail_shifts, tail_dims)

    (dim,) = dims
    size = V.graph.sizevars.guard_static_shape(a.get_size()[dim])
    start = (size - shifts[0]) % size
    a_loader = a.make_loader()

    def fn(index):
        index = list(index)
        index[dim] = ir.ModularIndexing(
            index[dim] + start, sympy.Integer(1), sympy.expand(size)
        )
        return a_loader(index)

    return Pointwise.create(
        device=a.get_device(),
        dtype=a.get_dtype(),
        inner_fn=fn,
        ranges=a.get_size(),
    )


@register_lowering(aten.as_strided, type_promote=False)
def as_strided(x, size, stride, storage_offset=None):
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        # as_strided ignores views
        x = x.data.unwrap_view()
    x.realize()
    if not ir.is_contiguous_storage_and_layout(x):
        raise NotImplementedError(f"unrealized as_strided({x}, ...)")
    storage, old_layout = ir.as_contiguous_storage_and_layout(x)
    new_layout = ir.FixedLayout(
        old_layout.device,
        old_layout.dtype,
        [sympy.expand(s) for s in size],
        [sympy.expand(s) for s in stride],
        sympy.expand(storage_offset or 0),
    )
    return TensorBox(ir.ReinterpretView(storage, new_layout))


@register_lowering(aten.as_strided_)
def as_strided_(x, size, stride, storage_offset=None):
    assert isinstance(x, TensorBox)
    x.data = as_strided(x, size, stride, storage_offset).data
    return x


@register_lowering(aten.cat)
def cat(inputs, dim=0):
    if len(inputs) == 1:
        return inputs[0]
    dim = _validate_dim(inputs[0], dim, 0)
    return TensorBox(ir.ConcatKernel.create(inputs, dim))


@register_lowering(aten.select)
def select(x, dim, idx):
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    return squeeze(slice_(x, dim, idx, idx + 1), dim)


@register_lowering(aten.split)
def split(x, sizes, dim=0):
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_static_shape(x.get_size()[dim])
    if isinstance(sizes, int):
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(slice_(x, dim, start, end))
        start = end
    return result


@register_lowering(aten.split_with_sizes)
def split_with_sizes(x, sizes, dim=0):
    return split(x, sizes, dim)


@register_lowering(aten.unbind)
def unbind(x, dim=0):
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_static_shape(x.get_size()[dim])
    result = []
    for i in range(x_size):
        result.append(select(x, dim, i))
    return result


@register_lowering(aten.unsqueeze, type_promote=False)
def unsqueeze(x, dim):
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.Integer(1))
    return view(x, new_shape)


@register_lowering(aten.unsqueeze_, type_promote=False)
def unsqueeze_(x, dim):
    val = unsqueeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x


def _validate_dim(x, dim, offset=0):
    assert isinstance(dim, int)
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    assert 0 <= dim < ndim + offset
    return dim


@register_lowering(aten.glu)
def glu(x, dim=-1):
    dim = _validate_dim(x, dim, 0)
    new_len = V.graph.sizevars.guard_static_shape(x.get_size()[dim]) // 2
    a = slice_(x, dim, 0, new_len)
    b = slice_(x, dim, new_len, new_len * 2)
    return mul(a, sigmoid(b))


@register_lowering(aten.mm)
def mm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.MatrixMultiply.create(a, b))


@register_lowering(aten.addmm)
def addmm(inp: TensorBox, a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.MatrixMultiplyAdd.create(inp, a, b))


@register_lowering(aten.bmm)
def bmm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.BatchMatrixMultiply.create(a, b))


def fallback_handler(kernel):
    fallbacks.add(kernel)

    def handler(*args, **kwargs):
        result = ir.FallbackKernel.create(kernel, *args, **kwargs)
        if isinstance(result, (list, tuple)):
            return list(map(TensorBox.create, result))
        else:
            return TensorBox.create(result)

    return handler


def make_fallback(kernel):
    assert (
        kernel not in decompositions
    ), f"both a fallback and a decomp for same kernel: {kernel}"
    if get_decompositions([kernel]):
        log.warning(
            f"make_fallback({kernel}): a decomposition exists, we should switch to it"
        )

    add_needs_realized_inputs(kernel)
    return register_lowering(kernel, type_promote=False)(fallback_handler(kernel))


@register_lowering(aten.native_dropout, type_promote=False)
def native_dropout(x, p, train):
    assert (
        config.fallback_random
    ), "this should be handled in decomps unless config.fallback_random"
    if train:
        return list(
            map(
                TensorBox.create,
                ir.FallbackKernel.create(aten.native_dropout, x, p, train),
            )
        )
    return x, ones_like(x, dtype=torch.bool)


@register_lowering(aten.bernoulli_, type_promote=False)
def bernoulli_(x, *args):
    assert (
        config.fallback_random
    ), "this should be handled in decomps unless config.fallback_random"
    x.realize()
    V.graph.realize_users_of(x.get_name())
    ir.InplaceBernoulliFallback(x, *args)
    return x


def make_rand(fn_name):
    def rand_or_randn(
        *size,
        dtype=None,
        layout=0,
        device=None,
        pin_memory=False,
        memory_format=None,
    ):
        log.warning("using triton random, expect difference from eager")
        assert not pin_memory
        assert layout in (0, torch.strided)
        assert memory_format in (None, torch.contiguous_format)
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        size = [sympy.expand(s) for s in size]
        offset = V.graph.increment_randomness_offset(sympy_product(size))

        random_pos = ir.FixedLayout(
            device,
            dtype,
            size,
            ir.FlexibleLayout.contiguous_strides(size),
            offset=offset,
        ).make_indexer()

        seed_buffer = V.graph.random_seed_buffer(device).make_loader()

        def inner_fn(index):
            seed = seed_buffer([])
            # change seed so that we don't collide with philox_rand_like()
            # TODO(jansel): migrate everything to philox_rand_like()
            seed = ops.bitwise_xor(seed, ops.constant(0xFFFF, torch.int32))
            return getattr(ops, fn_name)(
                seed,
                ops.index_expr(random_pos(index), torch.int32),
                dtype,
            )

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=list(size),
        )

    return rand_or_randn


fallback_rand = fallback_handler(aten.rand)
fallback_randn = fallback_handler(aten.randn)
fast_rand = make_rand("rand")
fast_randn = make_rand("randn")


@register_lowering([aten.rand, torch.rand])
def rand(*args, **kwargs):
    if config.fallback_random:
        return fallback_rand(*args, **kwargs)
    else:
        return fast_rand(*args, **kwargs)


@register_lowering([aten.randn, torch.randn])
def randn(*args, **kwargs):
    if config.fallback_random:
        return fallback_randn(*args, **kwargs)
    else:
        return fast_randn(*args, **kwargs)


@register_lowering(overrides.philox_seed_like._overloadpacket)
def philox_seed_like(x):
    log.warning("using triton random, expect difference from eager")
    return V.graph.random_seed_buffer(x.get_device())


@register_lowering(overrides.philox_rand_like._overloadpacket, type_promote=False)
def philox_rand_like(x, seed, offset):
    device = x.get_device()
    dtype = x.get_dtype()
    size = x.get_size()
    random_pos = ir.FixedLayout(
        device,
        dtype,
        size,
        ir.FlexibleLayout.contiguous_strides(size),
        offset=sympy.expand(offset),
    ).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        return ops.rand(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
            dtype,
        )

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )


if has_torchvision_roi_align():
    make_fallback(torch.ops.torchvision.roi_align)

# TODO(jansel): we should implement decomps or lowerings for these
# https://github.com/pytorch/torchdynamo/issues/327
make_fallback(aten._adaptive_avg_pool2d_backward)
make_fallback(aten.as_strided_scatter)
make_fallback(aten.col2im)
make_fallback(aten.convolution_backward)
make_fallback(aten._cudnn_rnn)
make_fallback(aten._cudnn_rnn_backward)
make_fallback(aten.cumsum)
make_fallback(aten._embedding_bag)
make_fallback(aten._embedding_bag_forward_only)
make_fallback(aten._fused_moving_avg_obs_fq_helper)
make_fallback(aten.grid_sampler_2d_backward)
make_fallback(aten.im2col)
make_fallback(aten.native_group_norm_backward)
make_fallback(aten.randperm)
make_fallback(aten.sort)
make_fallback(aten.sort.stable)
make_fallback(aten.topk)
make_fallback(aten.unfold)
make_fallback(aten.unfold_backward)
make_fallback(aten.upsample_bicubic2d)
make_fallback(aten.upsample_bicubic2d_backward)
make_fallback(aten.upsample_bilinear2d_backward)


@register_lowering(aten.convolution)
def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):
    result = TensorBox.create(
        ir.Convolution.create(
            x,
            weight,
            None,  # bias handled below
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    )
    if bias is not None:
        kernel_dims = len(weight.get_size()) - 2
        out_chan = result.get_size()[-1 - kernel_dims]
        bias = view(bias, [out_chan] + kernel_dims * [1])
        result = add(result, bias)
    return result


@register_lowering(aten._convolution)
def _convolution(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32,
):
    return convolution(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )


@register_lowering(aten.clone)
def clone(x, *, memory_format=0):
    # TODO(jansel): memory format
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
    )


if hasattr(aten, "lift_fresh_copy"):
    register_lowering(aten.lift_fresh_copy)(clone)


@register_lowering([torch.arange, aten.arange])
def arange(
    start,
    end=None,
    step=1,
    *,
    dtype=None,
    device=None,
    layout=torch.strided,
    pin_memory=False,
):
    assert layout == torch.strided
    assert not pin_memory
    if end is None:
        end = start
        start = 0

    if isinstance(start, float) and int(start) == start:
        start = int(start)
    if isinstance(end, float) and int(end) == end:
        end = int(end)
    if isinstance(step, float) and int(step) == step:
        step = int(step)

    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    dtype = dtype or torch.int64
    length = (end - start) // step
    start = sympy.Integer(start)
    step = sympy.Integer(step)

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=lambda index: ops.index_expr(step * index[0] + start, dtype),
        ranges=[sympy.Integer(length)],
    )


@register_lowering([torch.linspace, aten.linspace])
def linspace(start, end, steps, *, dtype=None, device=None, pin_memory=False):
    assert not pin_memory
    dtype = dtype or torch.get_default_dtype()

    step_size = (end - start) / (steps - 1)

    def inner_fn(index):
        return ops.add(
            ops.mul(ops.constant(step_size, dtype), ops.index_expr(index[0], dtype)),
            ops.constant(start, dtype),
        )

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[sympy.Integer(steps)],
    )


@register_lowering(aten.triu)
def triu(x, diagonal=0):
    x_loader = x.make_loader()
    dtype = x.get_dtype()

    def inner_fn(index):
        *_, i, j = index
        return ops.where(
            ops.ge(
                ops.index_expr(j - i - diagonal, torch.int32),
                ops.constant(0, torch.int32),
            ),
            x_loader(index),
            ops.constant(0, dtype),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


@register_lowering(aten.select_scatter)
def select_scatter(x, src, dim: int, index: int):
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    src = expand(unsqueeze(src, dim), x.get_size())
    src_loader = src.make_loader()

    def inner_fn(idx):
        return ops.where(
            ops.eq(
                ops.index_expr(idx[dim], torch.int32),
                ops.constant(index, torch.int32),
            ),
            src_loader(idx),
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


@register_lowering(aten.slice_scatter)
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    if start is not None and start < 0:
        start = start + dim_size
    if end is not None and end < 0:
        end = end + dim_size
    if start is None:
        start = 0
    if end is None or V.graph.sizevars.maybe_guard_leq(x.get_size()[dim], end):
        end = dim_size

    src_size = list(x.get_size())
    src_size[dim] = ir.IndexingDiv(sympy.expand(end - start), sympy.expand(step))
    src = expand(src, src_size)
    src_loader = src.make_loader()

    def inner_fn(idx):
        if start == 0 and end == dim_size and step == 1:
            # selecting every element is the same as just src.clone()
            return src_loader(idx)

        idx_dim = ops.index_expr(idx[dim], torch.int32)
        src_idx = list(idx)
        src_idx[dim] = ir.IndexingDiv(idx[dim] - start, step)

        mask = []
        if start != 0:
            mask.append(
                ops.ge(
                    idx_dim,
                    ops.index_expr(sympy.expand(start), torch.int32),
                )
            )
        if end != dim_size:
            mask.append(
                ops.lt(
                    idx_dim,
                    ops.index_expr(sympy.expand(end), torch.int32),
                )
            )
        if step != 1:
            mask.append(
                ops.eq(
                    ops.index_expr(
                        ir.ModularIndexing(idx[dim] - start, 1, step), torch.int32
                    ),
                    ops.constant(0, torch.int32),
                )
            )
        assert mask
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(mask, lambda: src_loader(src_idx), 0.0)
        return ops.where(
            mask,
            src_val,
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


def _unwrap(x):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _unwrap(x[0])
    return x


@register_lowering([torch.tensor, aten.scalar_tensor])
def tensor(data, *, dtype=None, device=None, layout=None):
    assert layout in (None, torch.strided)
    if isinstance(_unwrap(data), int):
        dtype = dtype or torch.int64
    else:
        dtype = dtype or torch.get_default_dtype()

    if isinstance(data, (float, int)):
        ranges = []

        def inner_fn(index):
            return ops.constant(data, dtype)

    elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
        # inline small tensors
        ranges = [sympy.Integer(len(data))]

        def inner_fn(index):
            def binary_search(start, end):
                assert start < end
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(
                    ops.lt(
                        ops.index_expr(index[0], torch.int64),
                        ops.constant(mid, torch.int64),
                    ),
                    binary_search(start, mid),
                    binary_search(mid, end),
                )

            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))

    else:
        return V.graph.add_tensor_constant(
            torch.tensor(data, dtype=dtype, device=device)
        )

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=ranges,
    )


@register_lowering(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    if isinstance(data, TensorBox):
        if dtype is not None:
            data = to(data, dtype)
        if device is not None:
            data = to(data, device)
        return data
    return tensor(data, dtype=dtype, device=device)


@register_lowering(torch.LongTensor)
def long_tensor(data):
    return tensor(data, dtype=torch.int64)


@register_lowering(aten._local_scalar_dense)
def _local_scalar_dense(data):
    return ir.DynamicScalar()


def _full(fill_value, device, dtype, size):
    value = fill_value
    if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
        value = value.value
    if isinstance(value, (int, float)):

        def inner_fn(index):
            return ops.constant(value, dtype)

    else:
        assert len(value.get_size()) == 0
        value_loader = value.make_loader()

        def inner_fn(index):
            return value_loader([])

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )


@register_lowering(aten.full_like)
def full_like(x, fill_value, **kwargs):
    return create_tensor_like(tensor_constructor(fill_value))(x, **kwargs)


def tensor_constructor(fill_value):
    # torch.zeros, torch.ones, etc
    def inner(*size, dtype=None, device=None, layout=0, pin_memory=False):
        assert not pin_memory
        assert layout in (0, torch.strided)
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        size = [sympy.expand(s) for s in size]
        return _full(fill_value, device, dtype, size)

    return inner


empty = register_lowering([torch.empty, aten.empty])(tensor_constructor(0))
zeros = register_lowering([torch.zeros, aten.zeros])(tensor_constructor(0))
ones = register_lowering([torch.ones, aten.ones])(tensor_constructor(1))


def create_tensor_like(creation_fn):
    """
    Shim to convert X_like(...) into X(...).  For example zeros_like() into zeros().
    """

    def _constant_like(
        x, *, dtype=None, device=None, layout=0, pin_memory=False, memory_format=None
    ):
        assert not pin_memory
        assert layout in (0, torch.strided)

        if dtype is None:
            dtype = x.get_dtype()
        else:
            dtype = decode_dtype(dtype)
        device = device or x.get_device()
        size = list(x.get_size())
        return creation_fn(
            size, dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
        )

    return _constant_like


def constant_like(fill_value):
    return create_tensor_like(tensor_constructor(fill_value))


empty_like = register_lowering(aten.empty_like)(create_tensor_like(empty))
zeros_like = register_lowering(aten.zeros_like)(create_tensor_like(zeros))
ones_like = register_lowering(aten.ones_like)(create_tensor_like(ones))
if not config.fallback_random:
    rand_like = register_lowering(aten.rand_like)(create_tensor_like(rand))


def new_constant(fill_value):
    def _new_constant(
        x, size, *, dtype=None, layout=None, device=None, pin_memory=None
    ):
        assert isinstance(size, (list, type))
        assert not pin_memory
        assert not layout or layout == torch.strided
        dtype = decode_dtype(dtype) or x.get_dtype()
        device = device or x.get_device()
        size = [sympy.Integer(s) for s in size]
        return _full(fill_value, device, dtype, size)

    return _new_constant


register_lowering(aten.new_empty)(new_constant(0))
register_lowering(aten.new_zeros)(new_constant(0))
register_lowering(aten.new_ones)(new_constant(1))


@register_lowering(aten.empty_strided)
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    assert isinstance(size, (list, type))
    assert isinstance(stride, (list, type))
    assert not pin_memory
    assert not layout or layout == torch.strided
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    device = device or torch.tensor(0.0).device
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    if tuple(ir.FlexibleLayout.contiguous_strides(size)) == tuple(stride):
        # fast path, no need to realize it
        return pointwise
    pointwise.realize()
    buffer = pointwise.data.data
    assert isinstance(buffer, ir.ComputedBuffer)
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=[sympy.expand(s) for s in size],
        stride=[sympy.expand(s) for s in stride],
    )
    return pointwise


@register_lowering(aten.new_empty_strided)
def new_empty_strided(
    x, size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(
        size, stride, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_lowering([torch.full, aten.full])
def full(size, fill_value, **kwargs):
    return tensor_constructor(fill_value)(size, **kwargs)


@register_lowering(aten.gather, type_promote=False)
def gather(x, dim, index):
    assert isinstance(x, TensorBox)
    assert isinstance(dim, int)
    assert "int" in str(index.get_dtype())
    assert 0 <= dim < len(x.get_size())

    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        idx = list(idx)
        idx[dim] = ops.indirect_indexing(index_loader(idx))
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )


@register_lowering(aten.embedding, type_promote=False)
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    assert not sparse
    assert isinstance(weight, TensorBox)
    assert isinstance(indices, TensorBox)
    assert "int" in str(indices.get_dtype())

    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    new_size = [*indices.get_size(), *weight.get_size()[1:]]

    def fn(idx):
        assert len(idx) == len(new_size), f"{idx} != {new_size}"
        var_index = indices_loader(idx[:indices_ndim])
        weight_idx = [ops.indirect_indexing(var_index)] + [*idx[indices_ndim:]]
        return weight_loader(weight_idx)

    return Pointwise.create(
        device=weight.get_device(),
        dtype=weight.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )


def check_and_broadcast_indices(indices):
    assert all(
        [
            i.get_dtype() in (torch.int64, torch.bool, torch.uint8)
            for i in indices
            if i is not None
        ]
    ), "indices must be int64, byte or bool"
    assert all(
        [i.get_dtype() == torch.int64 for i in indices if i is not None]
    ), "bool indices are not supported yet"
    valid_idxs = [i for i, x in enumerate(indices) if isinstance(x, TensorBox)]
    assert len(valid_idxs) > 0, "requires at least 1 non-None index"
    new_indices = [None] * len(indices)
    for i, x in zip(valid_idxs, broadcast_tensors(*[indices[i] for i in valid_idxs])):
        new_indices[i] = x
        output_dim = len(x.get_size())
    start_offset = 0
    # only support None at start or end for now
    tmp = list(new_indices)
    while tmp and tmp[-1] is None:
        tmp.pop()
    while tmp and tmp[0] is None:
        tmp.pop(0)
        start_offset += 1
    assert all((i is not None) for i in tmp)
    end_offset = output_dim + start_offset

    return new_indices, start_offset, end_offset


@register_lowering(aten.index, type_promote=False)
def index(x, indices):
    assert isinstance(indices, (list, tuple))
    x_loader = x.make_loader()
    indices, start_offset, end_offset = check_and_broadcast_indices(indices)
    indices_sizes = [i.get_size() for i in indices if i is not None]
    indices_loaders = [i.make_loader() for i in indices if i is not None]
    # no guards on output size, all the guards are set in broadcast_tensors
    output_size = list(indices_sizes[0])

    x_size = x.get_size()
    output_size = [
        *x_size[:start_offset],
        *output_size,
        *x_size[start_offset + len(indices_loaders) :],
    ]

    def fn(idx):
        assert len(idx) == len(output_size)
        new_index = [
            ops.indirect_indexing(loader(idx[start_offset:end_offset]))
            for loader in indices_loaders
        ]
        new_index = [*idx[:start_offset], *new_index, *idx[end_offset:]]
        return x_loader(new_index)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=output_size,
    )


@register_lowering(aten.index_put_, type_promote=False)
def index_put_(self, indices, values, accumulate=False):
    values = to_dtype(values, self.get_dtype())
    indices, start_offset, end_offset = check_and_broadcast_indices(indices)
    indices_sizes = [i.get_size() for i in indices if i is not None]
    indices_loaders = [i.make_loader() for i in indices if i is not None]

    assert isinstance(self, TensorBox)
    self.realize()
    V.graph.realize_users_of(self.get_name())

    x_size = self.get_size()
    output_size = list(indices_sizes[0])
    expected_vals_size = [
        *x_size[:start_offset],
        *output_size,
        *x_size[start_offset + len(indices_sizes) :],
    ]

    values = expand(values, expected_vals_size)
    # all guards are set above during broadcast_tensors and expand

    def output_indexer(index):
        assert len(index) == len(expected_vals_size)
        new_index = [
            ops.indirect_indexing(loader(index[start_offset:end_offset]))
            for loader in indices_loaders
        ]
        new_index = [*index[:start_offset], *new_index, *index[end_offset:]]
        return new_index

    scatter = ir.Scatter(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=output_indexer,
        scatter_mode="atomic_add" if accumulate else None,
    )
    buffer = ir.ComputedBuffer(
        None,
        ir.MutationLayout(self),
        scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    return self


@register_lowering(aten.index_select, type_promote=False)
def index_select(x, dim, indices):
    x_loader = x.make_loader()
    index_loader = indices.make_loader()
    dim = _validate_dim(x, dim, 0)

    sizes = list(x.get_size())
    (sizes[dim],) = indices.get_size()

    def fn(idx):
        assert len(idx) == len(sizes)
        idx = list(idx)
        idx[dim] = ops.indirect_indexing(index_loader([idx[dim]]))
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=sizes,
    )


@register_lowering(aten.scatter_, type_promote=False)
def scatter_(self, dim: int, index, src, *, reduce: str = None):
    if reduce == "add":
        reduce = "sum"
    elif reduce == "multiply":
        assert False, "TODO: multiply not supported"
        reduce = "prod"
    else:
        assert reduce is None
    return scatter_reduce_(self, dim, index, src, reduce)


@register_lowering(aten.scatter_reduce_, type_promote=False)
def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool = True):
    # TODO: Need to support more reduction type
    assert reduce is None or reduce in {"sum"}
    assert isinstance(self, TensorBox)
    assert "int" in str(index.get_dtype())
    assert -len(self.get_size()) <= dim < len(self.get_size())

    self.realize()
    V.graph.realize_users_of(self.get_name())
    index_loader = index.make_loader()
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx):
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(index_loader(idx))
        return indirect_idx

    def fn(idx):
        if src_loader:
            return src_loader(idx)
        else:
            # src is a scalar
            return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce):
        if reduce == "sum":
            return "atomic_add"
        else:
            # TODO: Need to support more reduction type
            assert reduce is None
            return None

    if not include_self:
        # zero out the corresponding elements first
        zero_out = ir.Scatter(
            device=self.get_device(),
            dtype=self.get_dtype(),
            inner_fn=lambda index: ops.constant(0, self.get_dtype()),
            ranges=index.get_size(),
            output_indexer=output_indexer,
            scatter_mode=None,
        )
        buffer = ir.ComputedBuffer(
            None,
            ir.MutationLayout(self),
            zero_out,
        )
        buffer.name = V.graph.register_buffer(buffer)

    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    # self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
    scatter = ir.Scatter(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
        output_indexer=output_indexer,
        scatter_mode=backend_reduce_str(reduce),
    )
    buffer = ir.ComputedBuffer(
        None,
        ir.MutationLayout(self),
        scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    return self


@register_lowering(aten.upsample_nearest2d)
def upsample_nearest2d(x, output_size=None, scale_factors=None):
    x.realize_hint()  # elements are reused
    x_loader = x.make_loader()

    *batch, ih, iw = x.get_size()
    ih = V.graph.sizevars.guard_static_shape(ih)
    iw = V.graph.sizevars.guard_static_shape(iw)

    if scale_factors:
        assert not output_size
        sh, sw = scale_factors
        oh = int(ih * sh)
        ow = int(iw * sw)
    else:
        oh, ow = output_size

    scale_h = ih / oh
    scale_w = iw / ow

    def scale(x, scale):
        x = ops.index_expr(x, torch.float32)
        x = ops.mul(x, ops.constant(scale, torch.float32))
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x)

    def fn(idx):
        *b, x, y = idx
        return x_loader([*b, scale(x, scale_h), scale(y, scale_w)])

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=[*batch, sympy.Integer(oh), sympy.Integer(ow)],
    )


@register_lowering([aten.grid_sampler_2d])
def grid_sampler_2d(
    image, optical, interpolation_mode=0, padding_mode=0, align_corners=False
):
    image.realize_hint()  # reuse
    optical.realize_hint()
    image_loader = image.make_loader()
    optical_loader = optical.make_loader()

    assert interpolation_mode == 0
    assert padding_mode in (0, 1)

    N, C, IH, IW = image.get_size()
    _, H, W, _ = optical.get_size()

    def clamp(v, min, max):
        if isinstance(min, (int, sympy.Expr)):
            min = ops.index_expr(min, torch.float32)
        if isinstance(max, (int, sympy.Expr)):
            max = ops.index_expr(max, torch.float32)
        return ops.maximum(min, ops.minimum(max, v))

    def findex(v):
        # indirect indexing via a float value
        return ops.indirect_indexing(ops.to_dtype(v, torch.int64))

    def fn(index):
        n, c, y, x = index
        ix = optical_loader([n, y, x, sympy.Integer(0)])
        iy = optical_loader([n, y, x, sympy.Integer(1)])
        zero = ops.constant(0.0, torch.float32)
        one = ops.constant(1.0, torch.float32)
        two = ops.constant(2.0, torch.float32)

        def grid_sampler_compute_source_index(coord, size):
            size = ops.index_expr(size, torch.float32)
            if align_corners:
                # unnormalize coord from [-1, 1] to [0, size - 1]
                coord = ops.mul(ops.div(ops.add(coord, one), two), ops.sub(size, one))
            else:
                # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
                coord = ops.div(ops.sub(ops.mul(ops.add(coord, one), size), one), two)

            if padding_mode == 0:  # GridSamplerPadding::Zeros
                return coord
            elif padding_mode == 1:  # GridSamplerPadding::Border
                return clamp(coord, zero, ops.sub(size, one))
            else:
                raise NotImplementedError("reflection padding")

        ix = grid_sampler_compute_source_index(ix, IW)
        iy = grid_sampler_compute_source_index(iy, IH)

        ix_nw = ops.floor(ix)
        iy_nw = ops.floor(iy)

        ix_ne = ops.add(ix_nw, one)
        iy_ne = iy_nw

        ix_sw = ix_nw
        iy_sw = ops.add(iy_nw, one)

        ix_se = ops.add(ix_nw, one)
        iy_se = ops.add(iy_nw, one)

        nw = ops.mul(ops.sub(ix_se, ix), ops.sub(iy_se, iy))
        ne = ops.mul(ops.sub(ix, ix_sw), ops.sub(iy_sw, iy))
        sw = ops.mul(ops.sub(ix_ne, ix), ops.sub(iy, iy_ne))
        se = ops.mul(ops.sub(ix, ix_nw), ops.sub(iy, iy_nw))

        # TODO(jansel): here we are missing the mask required
        # for GridSamplerPadding::Zeros and instead always doing
        # GridSamplerPadding::Border.  It does not seem to matter for
        # correctness in the one model using this: SuperSlowMo.

        ix_nw = clamp(ix_nw, 0, IW - 1)
        iy_nw = clamp(iy_nw, 0, IH - 1)
        ix_ne = clamp(ix_ne, 0, IW - 1)
        iy_ne = clamp(iy_ne, 0, IH - 1)
        ix_sw = clamp(ix_sw, 0, IW - 1)
        iy_sw = clamp(iy_sw, 0, IH - 1)
        ix_se = clamp(ix_se, 0, IW - 1)
        iy_se = clamp(iy_se, 0, IH - 1)

        nw_val = image_loader([n, c, findex(iy_nw), findex(ix_nw)])
        ne_val = image_loader([n, c, findex(iy_ne), findex(ix_ne)])
        sw_val = image_loader([n, c, findex(iy_sw), findex(ix_sw)])
        se_val = image_loader([n, c, findex(iy_se), findex(ix_se)])

        return functools.reduce(
            ops.add,
            [
                ops.mul(nw_val, nw),
                ops.mul(ne_val, ne),
                ops.mul(sw_val, sw),
                ops.mul(se_val, se),
            ],
        )

    return Pointwise.create(
        device=image.get_device(),
        dtype=image.get_dtype(),
        inner_fn=fn,
        ranges=[N, C, H, W],
    )


@register_lowering(aten.reflection_pad2d)
def reflection_pad2d(x, padding):
    assert len(padding) == 4
    left, right, top, bot = padding

    x_loader = x.make_loader()
    *batch, h, w = x.get_size()
    h = V.graph.sizevars.guard_static_shape(h)
    w = V.graph.sizevars.guard_static_shape(w)

    def reflect(x, size, offset):
        size = ops.constant(size - 1, torch.int32)
        x = ops.index_expr(x, torch.int32)
        x = ops.sub(x, ops.constant(offset, torch.int32))
        x = ops.sub(size, ops.abs(ops.sub(size, ops.abs(x))))
        return ops.indirect_indexing(x)

    def fn(idx):
        *b, x, y = idx
        x = reflect(x, h, top)
        y = reflect(y, w, left)
        return x_loader([*b, x, y])

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=[*batch, sympy.Integer(h + top + bot), sympy.Integer(w + left + right)],
    )


@register_lowering(aten.reflection_pad2d_backward)
def reflection_pad2d_backward(grad_output, x, padding):
    assert len(padding) == 4
    left, right, top, bot = padding

    *_, h, w = x.get_size()
    h = V.graph.sizevars.guard_static_shape(h) - 1
    w = V.graph.sizevars.guard_static_shape(w) - 1
    grad_loader = grad_output.make_loader()

    def fn(idx):
        *b, x, y = idx

        def load_from_output(x, y):
            x = ops.indirect_indexing(ops.index_expr(x, torch.int32))
            y = ops.indirect_indexing(ops.index_expr(y, torch.int32))
            return grad_loader([*b, x, y])

        def index_range_condition(index_range):
            i, lb, ub = index_range
            i = ops.index_expr(i, torch.int32)
            return ops.and_(ops.ge(i, lb), ops.le(i, ub))

        def accumulate(out_x, out_y, index_range1, index_range2=None):
            nonlocal grad

            # If the upper bound is less than the lower bound, we can get rid of one accumulation.
            # This happens when the padding size is zero.
            if index_range1[2] < index_range1[1]:
                return
            cond = index_range_condition(index_range1)
            if index_range2 is not None:
                if index_range2[2] < index_range2[1]:
                    return
                cond = ops.and_(cond, index_range_condition(index_range2))
            g = ops.masked(cond, lambda: load_from_output(out_x, out_y), 0.0)
            grad = ops.add(grad, g)

        # Areas after reflection:
        #
        #   top-left    |   top     |   top-right
        # -----------------------------------------
        #   left        |   center  |   right
        # -----------------------------------------
        #   bottom-left |   bottom  |   bottom-right
        #
        # The center area is the orignial matrix. Other areas are reflections.

        center_x, center_y = x + top, y + left
        top_reflect_x, left_reflect_y = top - x, left - y
        bot_reflect_x, right_reflect_y = 2 * h + top - x, 2 * w + left - y

        # Accumulate gradients from different areas
        grad = load_from_output(center_x, center_y)
        accumulate(center_x, left_reflect_y, (y, 1, left))
        accumulate(center_x, right_reflect_y, (y, w - right, w - 1))
        accumulate(top_reflect_x, center_y, (x, 1, top))
        accumulate(bot_reflect_x, center_y, (x, h - bot, h - 1))
        accumulate(top_reflect_x, left_reflect_y, (x, 1, top), (y, 1, left))
        accumulate(top_reflect_x, right_reflect_y, (x, 1, top), (y, w - right, w - 1))
        accumulate(bot_reflect_x, left_reflect_y, (x, h - bot, h - 1), (y, 1, left))
        accumulate(
            bot_reflect_x, right_reflect_y, (x, h - bot, h - 1), (y, w - right, w - 1)
        )

        return grad

    return Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,
        ranges=list(x.get_size()),
    )


@register_lowering(aten.constant_pad_nd, type_promote=False)
def constant_pad_nd(x, padding, fill_value=0):
    assert (len(padding) % 2) == 0
    if all(p == 0 for p in padding):
        return x

    sizes = x.get_size()

    bounds = list(reversed(list(zip(padding[::2], padding[1::2]))))
    n = len(sizes) - len(bounds)

    output_size = list(sizes[:n])
    mask_sizes = []
    for (low, high), size in zip(bounds, sizes[n:]):
        size = V.graph.sizevars.guard_static_shape(size)
        mask_sizes.append(size)
        output_size.append(sympy.expand(size + low + high))
    assert len(output_size) == len(sizes)

    def mask(index):
        mask = []
        for idx, (low, high), length in zip(index[n:], bounds, mask_sizes):
            if low != 0:
                mask.append(range_mask_low(idx))
            if high != 0:
                mask.append(range_mask_high(idx, length))
        mask = functools.reduce(ops.and_, mask)
        return ops.masked(mask, lambda: x_loader(index), fill_value)

    def offset_fn(index):
        new_index = list(index[:n])
        for idx, (low, high) in zip(index[n:], bounds):
            new_index.append(idx - low)
        assert len(new_index) == len(index)
        return mask(new_index)

    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=offset_fn,
        ranges=output_size,
    )


def range_mask_low(i: sympy.Expr):
    return ops.ge(
        ops.index_expr(i, torch.int64),
        ops.index_expr(sympy.Integer(0), torch.int64),
    )


def range_mask_high(i: sympy.Expr, length: sympy.Expr):
    return ops.lt(
        ops.index_expr(i, torch.int64),
        ops.index_expr(length, torch.int64),
    )


def range_mask(i: sympy.Expr, length: sympy.Expr):
    return ops.and_(
        range_mask_low(i),
        range_mask_high(i, length),
    )


def constant_boundary_condition_2d(x, fill_value, padding):
    *_, h, w = x.get_size()
    x_loader = x.make_loader()

    def load(index):
        *prefix, ih, iw = index

        mask = ops.and_(
            range_mask(ih, h),
            range_mask(iw, w),
        )
        return ops.masked(mask, lambda: x_loader([*prefix, ih, iw]), fill_value)

    return load


def pooling_size(x, i, kernel_size, stride, padding, ceil_mode):

    x_out = ir.IndexingDiv(
        x + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1), stride[i]
    )

    if ceil_mode:
        x_alt = ir.IndexingDiv(
            x + 2 * padding[i] - (kernel_size[i] - 1) + 2 * (stride[i] - 1), stride[i]
        )

        if V.graph.sizevars.size_hint(x_out - x_alt) == 0:
            # ceil mode is actually a no-op, lets guard on that
            V.graph.sizevars.guard_equals(x_out, x_alt)
            ceil_mode = False
        else:
            x_out = x_alt
    return x_out, ceil_mode


@register_lowering(aten.max_pool2d_with_indices, type_promote=False)
def max_pool2d_with_indices(
    x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    assert dilation == 1 or all(d == 1 for d in dilation)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert padding == 0 or len(padding) == 2
    assert len(x.get_size()) in (3, 4)
    if padding == 0:
        padding = [0, 0]
    if stride is None:
        stride = kernel_size

    x.realize_hint()
    *batch, h, w = x.get_size()

    h_out, ceil_mode1 = pooling_size(h, 0, kernel_size, stride, padding, ceil_mode)
    w_out, ceil_mode2 = pooling_size(w, 1, kernel_size, stride, padding, ceil_mode)

    if padding[0] or padding[1] or ceil_mode1 or ceil_mode2:
        x_loader = constant_boundary_condition_2d(x, float("-inf"), padding)
    else:
        x_loader = x.make_loader()

    new_size = list(batch) + [h_out, w_out]

    def fn(idx, return_index):
        *prefix, bh, bw = idx
        maxval = None
        maxindex = None
        for ih, iw in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            ih = bh * stride[0] + ih - padding[0]
            iw = bw * stride[1] + iw - padding[1]
            val = x_loader([*prefix, ih, iw])
            index = ops.index_expr(ih * w + iw, torch.int64)
            if maxval is None:
                maxindex = index
                maxval = val
            else:
                maxindex = ops.where(ops.gt(val, maxval), index, maxindex)
                maxval = ops.maximum(val, maxval)
        if return_index:
            return maxindex
        else:
            return maxval

    r1 = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=functools.partial(fn, return_index=False),
        ranges=new_size,
    )
    r2 = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int64,
        inner_fn=functools.partial(fn, return_index=True),
        ranges=new_size,
    )
    # TODO(jansel): should we force these to be realized?
    return r1, r2


@register_lowering(aten.max_pool2d_with_indices_backward, type_promote=False)
def max_pool2d_with_indices_backward(
    grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    assert dilation == 1 or all(d == 1 for d in dilation)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert padding == 0 or len(padding) == 2
    assert len(x.get_size()) in (3, 4)
    if padding == 0:
        padding = [0, 0]
    if stride is None:
        stride = kernel_size

    # we will read this many times, so make sure it is computed
    grad_output.realize_hint()
    indices.realize_hint()

    *batch, height, width = x.get_size()
    *_, pooled_height, pooled_width = grad_output.get_size()

    indices_loader = indices.make_loader()
    grad_loader = grad_output.make_loader()
    new_size = list(x.get_size())

    h_window_size = max(
        [
            h // stride[0] - max(0, (h - kernel_size[0]) // stride[0])
            for h in range(kernel_size[0] * 2)
        ]
    )
    w_window_size = max(
        [
            w // stride[1] - max(0, (w - kernel_size[1]) // stride[1])
            for w in range(kernel_size[1] * 2)
        ]
    )

    def fn(idx):
        *prefix, h, w = idx
        index_test = ops.index_expr(h * width + w, torch.int32)
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            ir.IndexingDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            ir.IndexingDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(ir.IndexingDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(ir.IndexingDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                grad_index = [
                    *prefix,
                    ops.indirect_indexing(
                        ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32)))
                    ),
                    ops.indirect_indexing(
                        ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32)))
                    ),
                ]

                index_actual = indices_loader(grad_index)
                grad_part = grad_loader(grad_index)
                check = ops.eq(index_actual, index_test)

                if gradient is None:
                    # don't need mask for 0, 0
                    gradient = ops.where(
                        check, grad_part, ops.constant(0.0, torch.float32)
                    )
                else:
                    mask = ops.and_(
                        ops.and_(
                            ops.lt(ph, phend),
                            ops.lt(pw, pwend),
                        ),
                        check,
                    )
                    gradient = ops.where(mask, ops.add(gradient, grad_part), gradient)
        return gradient

    return Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )


def pad_adaptive_loader(x):
    *_, h, w = x.get_size()
    x_loader = x.make_loader()

    def load(prefix, increments, start_indices, end_indices):
        ih, iw = increments
        h_start_index, w_start_index = start_indices
        h_end_index, w_end_index = end_indices

        mask = ops.and_(
            ops.lt(
                ops.index_expr(h_start_index + ih, torch.int64),
                ops.index_expr(h_end_index, torch.int64),
            ),
            ops.lt(
                ops.index_expr(w_start_index + iw, torch.int64),
                ops.index_expr(w_end_index, torch.int64),
            ),
        )

        return ops.masked(
            mask,
            lambda: x_loader([*prefix, h_start_index + ih, w_start_index + iw]),
            0.0,
        )

    return load


def _adaptive_pooling_idx_sum(kernel_maxes, start_index_fns, end_index_fns):
    h_start_index_fn, w_start_index_fn = start_index_fns
    h_end_index_fn, w_end_index_fn = end_index_fns

    def fn_sum(idx, loader):
        *prefix, bh, bw = idx

        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        total = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total

    return fn_sum


@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    x.realize_hint()

    *batch, h_in, w_in = x.get_size()

    h_in = V.graph.sizevars.guard_static_shape(h_in)
    w_in = V.graph.sizevars.guard_static_shape(w_in)

    h_out, w_out = output_size

    # no-op if the same input and output
    if h_in == h_out and w_in == w_out:
        return clone(x)

    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return avg_pool2d(x, kernel_size)

    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    def start_index(index, out_dim, inp_dim):
        return ir.IndexingDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        return ir.IndexingDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)

    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)

    fn_sum = _adaptive_pooling_idx_sum(
        [h_kernel_max, w_kernel_max],
        [h_start_index, w_start_index],
        [h_end_index, w_end_index],
    )

    ones_loader = pad_adaptive_loader(ones_like(x))

    def fn(idx):
        return ops.div(fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # TODO: should we force these to be realized?
    return rv


@register_lowering(aten.upsample_nearest2d_backward.vec)
def upsample_nearest2d_backward(
    x, output_size=None, input_size=None, scale_factors=None
):
    x.realize_hint()

    *batch, inp_h, inp_w = x.get_size()
    inp_h = V.graph.sizevars.guard_static_shape(inp_h)
    inp_w = V.graph.sizevars.guard_static_shape(inp_w)

    *batch, out_h, out_w = input_size

    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(x, [inp_h // out_h, inp_w // out_w], divisor_override=1)

    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    def start_index(index, out_dim, inp_dim):
        return ir.CeilDiv(index * inp_dim, out_dim)

    def end_index(index, out_dim, inp_dim):
        return start_index((index + 1), out_dim, inp_dim)

    h_start_index = functools.partial(start_index, out_dim=out_h, inp_dim=inp_h)
    h_end_index = functools.partial(end_index, out_dim=out_h, inp_dim=inp_h)

    w_start_index = functools.partial(start_index, out_dim=out_w, inp_dim=inp_w)
    w_end_index = functools.partial(end_index, out_dim=out_w, inp_dim=inp_w)

    fn_sum = _adaptive_pooling_idx_sum(
        [h_kernel_max, w_kernel_max],
        [h_start_index, w_start_index],
        [h_end_index, w_end_index],
    )

    def fn(idx):
        return fn_sum(idx, pad_adaptive_loader(x))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=list(input_size),
    )

    return rv


@register_lowering(aten.avg_pool2d, type_promote=False)
def avg_pool2d(
    x,
    kernel_size,
    stride=[],
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)

    x.realize_hint()
    *batch, h, w = x.get_size()

    h_out, ceil_mode1 = pooling_size(h, 0, kernel_size, stride, padding, ceil_mode)
    w_out, ceil_mode2 = pooling_size(w, 1, kernel_size, stride, padding, ceil_mode)

    if padding[0] or padding[1] or ceil_mode1 or ceil_mode2:
        x_loader = constant_boundary_condition_2d(x, 0.0, padding)
        had_padding = True
    else:
        x_loader = x.make_loader()
        had_padding = False

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    def fn_sum(idx, loader):
        *prefix, bh, bw = idx
        total = None
        for ih, iw in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            ih = bh * stride[0] + ih - padding[0]
            iw = bw * stride[1] + iw - padding[1]
            val = loader([*prefix, ih, iw])
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total

    if count_include_pad or not had_padding or divisor_override:
        if divisor_override:
            scale = 1 / divisor_override
        else:
            scale = 1.0 / (kernel_size[0] * kernel_size[1])

        def fn(idx):
            return ops.mul(fn_sum(idx, x_loader), ops.constant(scale, dtype))

    else:
        ones_loader = constant_boundary_condition_2d(ones_like(x), 0.0, padding)

        def fn(idx):
            # TODO(jansel): optimize to do `int(x<h)` rather than `x<h?1:0`
            return ops.div(fn_sum(idx, x_loader), fn_sum(idx, ones_loader))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # TODO(jansel): should we force these to be realized?
    return rv


@register_lowering(aten.avg_pool2d_backward, type_promote=False)
def avg_pool2d_backward(
    grad_output,
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):

    assert not divisor_override
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)

    grad_output.realize_hint()  # we will read this many times, so make sure it is computed

    *batch, height, width = x.get_size()

    h_out, ceil_mode1 = pooling_size(height, 0, kernel_size, stride, padding, ceil_mode)
    w_out, ceil_mode2 = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)

    grad_loader = grad_output.make_loader()

    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2

    *_, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()

    h_window_size = max(
        [
            h // stride[0] - max(0, (h - kernel_size[0]) // stride[0])
            for h in range(kernel_size[0] * 2)
        ]
    )
    w_window_size = max(
        [
            w // stride[1] - max(0, (w - kernel_size[1]) // stride[1])
            for w in range(kernel_size[1] * 2)
        ]
    )

    def compute_pool_size_without_padding(ph, pw):
        """
        This computes the scaling factor that we will divide an element
        by when `count_include_pad=False`
        """
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        hend = ops.minimum(
            ops.add(hstart, kernel_h),
            ops.add(ops.index_expr(height, torch.int32), pad_h),
        )
        wend = ops.minimum(
            ops.add(wstart, kernel_w),
            ops.add(ops.index_expr(width, torch.int32), pad_w),
        )
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor

    def fn(idx):
        *prefix, h, w = idx
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            ir.IndexingDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            ir.IndexingDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(ir.IndexingDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(ir.IndexingDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))

                if count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)

                part = ops.truediv(
                    grad_loader(
                        [
                            *prefix,
                            ops.indirect_indexing(
                                ops.minimum(
                                    ph, ops.sub(phend, ops.constant(1, torch.int32))
                                )
                            ),
                            ops.indirect_indexing(
                                ops.minimum(
                                    pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                )
                            ),
                        ]
                    ),
                    scale,
                )

                if gradient is None:
                    # don't need mask for 0, 0
                    gradient = part
                else:
                    mask = ops.and_(
                        ops.lt(ph, phend),
                        ops.lt(pw, pwend),
                    )
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        return gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    return rv


def _validate_reduction_axis(x, axis):
    size = x.get_size()
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size)
        assert 0 <= axis[i] < len(size)
    assert len(set(axis)) == len(axis), "reduction axis not unique"
    return axis


def make_reduction(reduction_type: str, override_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        if reduction_type == "min" and axis is not None:
            return (
                reduce_amin(x, axis, keepdims, dtype=dtype),
                reduce_argmin(x, axis, keepdims),
            )
        if reduction_type == "max" and axis is not None:
            return (
                reduce_amax(x, axis, keepdims, dtype=dtype),
                reduce_argmax(x, axis, keepdims),
            )
        if dtype is not None:
            x = to_dtype(x, dtype)
        if reduction_type == "any":
            x = to_dtype(x, torch.bool)
        size = x.get_size()
        axis = set(_validate_reduction_axis(x, axis))

        kept_sizes = []
        kept_idx = []
        reduced_sizes = []
        reduced_idx = []
        for i in range(len(size)):
            if i in axis:
                reduced_idx.append(i)
                reduced_sizes.append(size[i])
            else:
                kept_idx.append(i)
                kept_sizes.append(size[i])

        def loader(index, reduction_index):
            assert len(reduction_index) == len(reduced_idx)
            if keepdims:
                assert len(index) == len(size)
                assert all(index[i] == 0 for i in reduced_idx)
                index = [index[i] for i in kept_idx]
            assert len(index) == len(kept_idx)
            new_index = [None] * (len(index) + len(reduction_index))
            for idx, var in itertools.chain(
                zip(kept_idx, index), zip(reduced_idx, reduction_index)
            ):
                new_index[idx] = var
            return inner_loader(new_index)

        if keepdims:
            new_size = list(size)
            for i in reduced_idx:
                new_size[i] = sympy.Integer(1)
        else:
            new_size = kept_sizes

        inner_loader = x.make_loader()
        result = Reduction.create(
            device=x.get_device(),
            dst_dtype=override_dtype or x.get_dtype(),
            src_dtype=x.get_dtype(),
            inner_fn=loader,
            ranges=new_size,
            reduction_ranges=reduced_sizes,
            reduction_type={"amax": "max", "amin": "min"}.get(
                reduction_type, reduction_type
            ),
        )
        result.realize()
        return result

    return inner


@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    # compute in higher-precision until end of mean lowering
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    sum_result = sum_(x, axis, keepdim)
    denom = sympy_product(size[i] for i in axis)
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return to_dtype(div(sum_result, denom), output_dtype)


@register_lowering([aten.var, prims.var])
def var_(x, axis, correction=1, keepdim=False):
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    diffs = square(sub(x, mean(x, axis, keepdim=True)))
    sum_result = sum_(diffs, axis, keepdim)

    denom = sympy_product(size[i] for i in axis)
    if correction:
        denom = denom - correction
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return div(sum_result, denom)


@register_lowering(aten.var_mean)
def var_mean(x, dim, unbiased=True, keepdim=False, correction=None):
    if correction is None:
        correction = int(unbiased)
    return [
        var_(x, dim, correction=correction, keepdim=keepdim),
        mean(x, dim, keepdim=keepdim),
    ]


@register_lowering(aten.std)
def std(x, axis, correction=1, keepdim=False):
    return sqrt(var_(x, axis, correction, keepdim=keepdim))


def pow_recursive(x, y, dtype):
    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x

    result = pow_recursive(x, y // 2, dtype)
    result = ops.mul(result, result)
    if (y % 2) == 1:
        result = ops.mul(result, x)
    return result


@make_pointwise
def pow_native(a, b):
    return ops.pow(a, b)


@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    if isinstance(b, float) and b == int(b):
        return pow(a, int(b))
    elif isinstance(b, int) and b == 1:
        return a
    elif isinstance(b, int) and -32 < b < 32:
        # Optimize away small fixed powers
        loader = a.make_loader()

        def fn(idx):
            return pow_recursive(loader(idx), b, a.get_dtype())

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
        )
    else:
        return pow_native(a, b)


def mutate_to(changed, val):
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data

    if not isinstance(val, ir.StorageBox):
        # introduce a copy to handle views
        val = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
        ).data
        assert isinstance(val, ir.StorageBox)

    if isinstance(changed_data, ir.StorageBox):
        # Fast path, just swing the data pointer
        val.realize()
        changed_data.data = val.data
        return changed

    ir.MutationLayout.realize_into(val, changed_data)
    return changed


@register_lowering(aten.fill_)
def fill_(x, fill_value):
    return mutate_to(x, full_like(x, fill_value))


@register_lowering(aten.zero_)
def zero_(x):
    return mutate_to(x, full_like(x, 0))


@register_lowering(aten.copy_, type_promote=False)
def copy_(dst, src, non_blocking=False):
    src = to_device(src, dst.get_device())
    src = to_dtype(src, dst.get_dtype())
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)


@make_pointwise
def floordiv(a, b):
    return ops.floordiv(a, b)


@make_pointwise
def truncdiv(a, b):
    return ops.truncdiv(a, b)


@register_lowering(aten.div.Tensor_mode)
def div_mode(a, b, rounding_mode=None):
    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    # floordiv and truncdiv need special handling for integer tensors on Triton,
    # see the discussion at https://github.com/openai/triton/issues/605
    if rounding_mode == "floor":
        assert not both_boolean, "floordiv operands can not be boolean at the same time"
        return floordiv(a, b) if both_integer else floor(div(a, b))
    if rounding_mode == "trunc":
        assert not both_boolean, "truncdiv operands can not be boolean at the same time"
        return truncdiv(a, b) if both_integer else trunc(div(a, b))
    return div(a, b)


@register_lowering([aten.div, prims.div], broadcast=True)
def div(a, b):
    def fn(*args):
        return ops.div(*args)

    dtype = get_promoted_dtype(a, b)
    # truediv produces a float tensor even if both operands are integer types
    if is_integer_type(a) and is_integer_type(b):
        dtype = torch.get_default_dtype()
    return make_pointwise(fn, override_dtype=dtype)(
        a if isinstance(a, Number) else to_dtype(a, dtype),
        b if isinstance(b, Number) else to_dtype(b, dtype),
    )


@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64
    fn = make_reduction("sum", override_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


register_lowering(aten.max)(make_reduction("max"))
register_lowering(aten.min)(make_reduction("min"))
reduce_amax = register_lowering(aten.amax)(make_reduction("amax"))
reduce_amin = register_lowering(aten.amin)(make_reduction("amin"))
register_lowering(aten.any)(make_reduction("any", override_dtype=torch.bool))
reduce_argmax = register_lowering(aten.argmax)(
    make_reduction("argmax", override_dtype=torch.int64)
)
reduce_argmin = register_lowering(aten.argmin)(
    make_reduction("argmin", override_dtype=torch.int64)
)

add = register_pointwise(aten.add)
exp = register_pointwise(aten.exp)
floor = register_pointwise(aten.floor)
mul = register_pointwise(aten.mul)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise(aten.sigmoid)
sqrt = register_pointwise(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub)
trunc = register_pointwise(aten.trunc)

register_pointwise(aten.cos)
register_pointwise(aten.sin)
register_pointwise(aten.abs)
register_pointwise(aten.bitwise_and)
register_pointwise(aten.bitwise_not, override_bool="logical_not")
register_pointwise(aten.bitwise_or)
register_pointwise(aten.bitwise_xor)
register_pointwise(aten.log)
register_pointwise(aten.logical_not)
register_pointwise(aten.maximum)
register_pointwise(aten.minimum)
register_pointwise(aten.neg)
register_pointwise(aten.reciprocal)
register_pointwise(aten.remainder)
register_pointwise(aten.round)
register_pointwise(aten.sign)
register_pointwise(aten.silu)
register_pointwise(aten.ceil)
register_pointwise(aten.isinf, override_dtype=torch.bool)
register_pointwise(aten.isnan, override_dtype=torch.bool)

register_pointwise(aten.le, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.lt, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.ge, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.gt, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.eq, type_promote=False, override_dtype=torch.bool)
register_pointwise(aten.ne, type_promote=False, override_dtype=torch.bool)
register_lowering(aten.__and__, type_promote=False)(
    register_pointwise(aten.logical_and, type_promote=False, override_dtype=torch.bool)
)
register_lowering(aten.__or__, type_promote=False)(
    register_pointwise(aten.logical_or, type_promote=False, override_dtype=torch.bool)
)


def register_inplace(aten_op, outplace_op):
    @register_lowering(aten_op, type_promote=False)
    def fn(*args):
        result = outplace_op(*args)
        return mutate_to(args[0], result)

    return fn


register_inplace(aten.add_, add)
register_inplace(aten.mul_, mul)
register_inplace(aten.div_, div)
register_inplace(aten.sub_, sub)
register_inplace(aten.relu_, relu)
register_inplace(aten.sigmoid_, sigmoid)


@register_lowering(aten.sym_size)
def sym_size(a, dim):
    return a.get_size()[dim]


@register_lowering(operator.mul)
def op_mul(a, b):
    return a * b
