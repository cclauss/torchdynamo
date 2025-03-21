import contextlib
import copy
import functools
import inspect
import logging
import os
import threading
import types
import warnings
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree

import torchdynamo
from torchdynamo.utils import checkpoint_params
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import same

from . import config
from . import convert_frame
from . import skipfiles
from . import utils
from .exc import ResetRequired
from .mutation_guard import install_generation_tagging_init

log = logging.getLogger(__name__)

try:
    from . import _eval_frame
except (ModuleNotFoundError, ImportError) as e:
    raise RuntimeError("run `python setup.py develop` to compile C extensions") from e

try:
    from torch.fx.experimental import proxy_tensor
except (ModuleNotFoundError, ImportError):
    proxy_tensor = None

set_eval_frame = _eval_frame.set_eval_frame
reset_code = _eval_frame.reset_code
unsupported = _eval_frame.unsupported
skip_code = _eval_frame.skip_code
set_guard_fail_hook = _eval_frame.set_guard_fail_hook
set_guard_error_hook = _eval_frame.set_guard_error_hook
always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext
unset = object()
compile_lock = threading.Lock()
most_recent_backend = None


def remove_from_cache(f):
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from torchdynamo import reset

        reset()
        log.warning("could not determine __code__ for %s", f)


def nothing():
    pass


def innermost_fn(fn):
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    unaltered_fn = fn
    while hasattr(unaltered_fn, "_torchdynamo_orig_callable"):
        unaltered_fn = getattr(unaltered_fn, "_torchdynamo_orig_callable")
        assert callable(unaltered_fn)
    return unaltered_fn


class _TorchDynamoContext:
    def __init__(
        self,
        callback,
        on_enter=nothing,
        backend_ctx_ctor=null_context,
        patch_fn=nothing,
    ):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback = callback
        self.prior = unset
        self.on_enter = on_enter
        self.extra_ctx_ctor = backend_ctx_ctor
        patch_fn()

    def __enter__(self):
        self.on_enter()
        self.prior = set_eval_frame(self.callback)
        self.backend_ctx = self.extra_ctx_ctor()
        self.backend_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_eval_frame(self.prior)
        self.prior = unset
        self.backend_ctx.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, fn):
        fn = innermost_fn(fn)
        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            mod = fn
            optimized_forward = self(mod.forward)

            class TorchDynamoNNModuleWrapper:
                """
                A wrapper that redirects the forward call to the optimized
                forward, while for rest it redirects the calls to the original
                module.
                """

                def __getattr__(self, name):
                    return getattr(mod, name)

                def forward(self, *args, **kwargs):
                    return optimized_forward(*args, **kwargs)

                def __call__(self, *args, **kwargs):
                    return self.forward(*args, **kwargs)

            new_mod = TorchDynamoNNModuleWrapper()
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod
            return new_mod

        assert callable(fn)
        callback = self.callback
        on_enter = self.on_enter
        backend_ctx_ctor = self.extra_ctx_ctor

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            on_enter()
            prior = set_eval_frame(callback)
            backend_ctx = backend_ctx_ctor()
            backend_ctx.__enter__()
            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(prior)
                backend_ctx.__exit__(None, None, None)

        # hooks to properly handle inlining
        if isinstance(self, DisableContext):
            _fn._torchdynamo_disable = True
        else:
            _fn._torchdynamo_inline = fn

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn

        # If the function is called with torchdynamo.optimize decorator, we
        # should prevent any type of skipping.
        if callback not in (None, False):
            always_optimize_code_objects[fn.__code__] = True

        return _fn


class OptimizeContext(_TorchDynamoContext):
    def __init__(self, callback, backend_ctx_ctor):
        def on_enter():
            global most_recent_backend
            if (
                most_recent_backend is not None
                and most_recent_backend is not compiler_fn
            ):
                raise ResetRequired()
            most_recent_backend = compiler_fn
            install_generation_tagging_init()

        compiler_fn = innermost_fn(callback)
        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=False)


class DisableContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def catch_errors_wrapper(callback):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size):
        try:
            if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
                log.debug(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
                return None
            if (
                frame.f_code.co_filename == "<string>"
                and frame.f_code.co_name == "__new__"
            ):
                # nametuple constructor
                return None
            with compile_lock:
                return callback(frame, cache_size)
        except Exception:
            logging.basicConfig()
            logging.exception("Error while processing frame")
            raise

    catch_errors._torchdynamo_orig_callable = callback
    return catch_errors


def _optimize_catch_errors(compile_fn, backend_ctx_ctor=null_context):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn), backend_ctx_ctor=backend_ctx_ctor
    )


class WrapperBackend:
    def __init__(self, backend=None):
        self.backend = backend

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):

        self.restore = checkpoint_params(gm)
        self.original_example_inputs = clone_inputs(example_inputs)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, self.original_example_inputs)

        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.candidate

        # if verify_correctness=True
        try:
            correct = self.gm.forward(*self.example_inputs)
            result = self.candidate(*self.example_inputs)

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.candidate

            raise RuntimeError(f"incorrect results of backend {self}")
            return self.gm.forward

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


def get_compiler_fn(compiler_fn):
    """Expand backend strings to functions"""
    if compiler_fn == "inductor":
        from torchinductor.compile_fx import compile_fx

        compiler_fn = compile_fx
    elif isinstance(compiler_fn, str):
        from .optimizations import BACKENDS

        compiler_fn = BACKENDS[compiler_fn]

    return compiler_fn


class _NullDecorator(contextlib.nullcontext):
    def __call__(self, fn):
        assert callable(fn)
        return fn


def optimize(
    backend="inductor", *, nopython=False, guard_export_fn=None, disable=False
):
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
            - Or, a string backend name in `torchdynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.
        disable: If True, turn this decorator into a no-op

    Example Usage:

        @torchdynamo.optimize()
        def toy_example(a, b):
            ...
    """
    if disable or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1":
        return _NullDecorator()

    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    if nopython:
        return optimize_assert(backend, guard_export_fn=guard_export_fn)
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, guard_export_fn=guard_export_fn),
        backend_ctx_ctor,
    )


@patch("torchdynamo.symbolic_convert.explain", True)
def explain(f, *args, **kwargs):
    # TODO(voz): Do we want a decorator for this?
    torchdynamo.reset()

    out_guards = []
    graphs = []
    ops_per_graph = []
    op_count = 0
    break_reasons = []

    def dynamo_graph_accumulating_compiler(gm: torch.fx.GraphModule, example_inputs):
        nonlocal graphs
        nonlocal op_count
        nonlocal ops_per_graph

        graphs.append(gm)
        ops = []
        for node in gm.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)

        op_count += len(ops)
        ops_per_graph.append(ops)
        if gm.compile_subgraph_reason is not None:
            break_reasons.append(gm.compile_subgraph_reason)
        return gm.forward

    def guard_export_print(guards):
        nonlocal out_guards
        out_guards.append(guards)

    with patch(f"{__name__}.most_recent_backend", None), optimize(
        dynamo_graph_accumulating_compiler,
        nopython=False,
        guard_export_fn=guard_export_print,
    ):
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        f(*args, **kwargs)

    graph_count = len(graphs)

    formatted_list = ""
    for idx in range(0, len(break_reasons)):
        formatted_list += f"{idx + 1}. {break_reasons[idx]} \n"

    explanation = f"Dynamo produced {graph_count} graphs"
    explanation += f"with {graph_count - 1} graph break and {op_count} ops"
    explanation += f"\n Break reasons: \n\n{formatted_list}"

    # TODO(voz): Do we want a decorator for this?
    torchdynamo.reset()
    return explanation, out_guards, graphs, ops_per_graph


def export(f, *args, **kwargs):
    f = innermost_fn(f)

    graph = None
    out_guards = None
    graph_captured_input = None
    graph_captured_result = None

    def produce_matching(source_args, candidate_args):
        matched_elements_positions = []
        dict_of_source_args = dict()
        for i in range(0, len(source_args)):
            element_id = id(source_args[i])
            dict_of_source_args[element_id] = i

        for i in range(0, len(candidate_args)):
            arg = candidate_args[i]
            # 1-element tensor arg can be unspec int/float
            if isinstance(arg, torch.Tensor) and torch.numel(arg) == 1:
                if id(arg) in dict_of_source_args:
                    matched_elements_positions.append(dict_of_source_args[id(arg)])
                elif id(arg.item()) in dict_of_source_args:
                    matched_elements_positions.append(
                        dict_of_source_args[id(arg.item())]
                    )
                else:
                    assert (
                        False
                    ), "Dynamo input/output is not consistent with traced input/output"
            else:
                assert (
                    id(arg) in dict_of_source_args
                ), "Dynamo input and output is a strict subset of traced input/output"
                matched_elements_positions.append(dict_of_source_args[id(arg)])

        return matched_elements_positions

    def guard_export_print(guards):
        nonlocal out_guards
        assert out_guards is None, "whole graph export entails exactly one guard export"
        out_guards = guards

    def dynamo_normalization_capturing_compiler(
        gm: torch.fx.GraphModule, example_inputs
    ):
        nonlocal graph

        assert graph is None, "whole graph export entails exactly one graph"
        graph = gm

        def result_capturing_wrapper(*graph_inputs):
            nonlocal graph_captured_result
            nonlocal graph_captured_input

            graph_captured_input = graph_inputs
            graph_captured_result = graph(*graph_inputs)
            return graph_captured_result

        return result_capturing_wrapper

    # TODO(voz): Handle kwargs properly?
    flat_args, in_spec = pytree.tree_flatten(args)

    remove_from_cache(f)
    with patch(f"{__name__}.most_recent_backend", None), optimize_assert(
        dynamo_normalization_capturing_compiler, guard_export_fn=guard_export_print
    ):
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        result_traced = f(*args, **kwargs)
    remove_from_cache(f)

    assert graph is not None, "whole graph export entails exactly one call"
    assert out_guards is not None, "whole graph export entails exactly one guard export"

    matched_input_elements_positions = produce_matching(flat_args, graph_captured_input)

    flat_results_traced, out_spec_traced = pytree.tree_flatten(result_traced)

    flat_both = list(graph_captured_result) + flat_args
    matched_output_elements_positions = produce_matching(flat_both, flat_results_traced)

    class ChangeInputOutputSignature(torch.fx.interpreter.Transformer):
        def __init__(
            self,
            m,
        ):
            super().__init__(m)
            arg_len = len(flat_args)
            self.new_args = [
                super(ChangeInputOutputSignature, self).placeholder(f"arg{i}", (), {})
                for i in range(0, arg_len)
            ]
            self.old_args_gen = (
                self.new_args[i] for i in matched_input_elements_positions
            )

        def placeholder(self, target, args, kwargs):
            return next(self.old_args_gen)

        def output(self, target, args, kwargs):
            dynamo_result_flat = args[0]
            lookup = [*dynamo_result_flat, *self.new_args]
            new_result_flat = [lookup[i] for i in matched_output_elements_positions]
            new_result = pytree.tree_unflatten(new_result_flat, out_spec_traced)

            return super().output(target, (new_result,), {})

    new_graph = ChangeInputOutputSignature(
        graph,
    ).transform()

    return (new_graph, out_guards)


def optimize_assert(backend, *, guard_export_fn=None):
    """
    The same as `torchdynamo.optimize(backend, nopython=True)`
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(backend, guard_export_fn), backend_ctx_ctor
    )


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None):
    """Decorator and context manager to disable TorchDynamo"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return DisableContext()(fn)
    return DisableContext()


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
    return fn


class TorchPatcher:
    @staticmethod
    @functools.lru_cache(None)
    def patch():
        # Disable TorchDynamo on some torch.* compilers generated frames
        torch.jit.trace = disable(torch.jit.trace)
        torch.jit.trace_module = disable(torch.jit.trace_module)
        torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)

        # symbolic_trace creates new frames. We disable Dynamo on such frames
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace
        )

        torch.onnx.export_to_pretty_string = disable(torch.onnx.export_to_pretty_string)
        torch.distributions.Distribution.set_default_validate_args(False)

        if proxy_tensor is not None:
            proxy_tensor.dispatch_trace = disable(proxy_tensor.dispatch_trace)

        optimizers = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # disable profile hook
        for opt in optimizers:
            opt._cuda_graph_capture_health_check = disable(
                opt._cuda_graph_capture_health_check
            )
            # disable any currently set hooks
            # Note: we only want to disable the profiling hook
            # which is the *last* hook applied, we want to keep the no_grad hook
            hooked = getattr(opt.step, "hooked", False)
            if hooked:
                unwrapped_step = getattr(opt.step, "__wrapped__", None)
                if unwrapped_step:
                    opt.step = unwrapped_step

            # disable future hooking
            setattr(opt.step, "hooked", True)

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            return fn(*args, **kwargs)

        return inner_fn
