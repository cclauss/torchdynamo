import copy
import functools
import getpass
import os
import shutil
import subprocess
import textwrap
import uuid
from collections import Counter

import torch
import torch.fx as fx

import torchdynamo
from torchdynamo import config


def minifier_dir():
    return f"/tmp/minifier_{getpass.getuser()}"


def generate_repro_string(gm, args):
    model_str = textwrap.dedent(
        """
        import torch
        from torch import tensor, device
        import torch.fx as fx
        from torchdynamo.testing import rand_strided
        from math import inf
        from torch.fx.experimental.proxy_tensor import make_fx

        """
    )
    model_str += f"# torch version: {torch.version.__version__}\n"
    model_str += f"# torch cuda version: {torch.version.cuda}\n"
    model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
    if torch.cuda.is_available():
        model_str += "# CUDA Info: \n"
        cuda_version_out = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE)
        cuda_version_lines = cuda_version_out.stdout.decode().split("\n")
        cuda_version_out = "".join(
            [f"# {s} \n" for s in cuda_version_lines if s not in [""]]
        )
        model_str += f"{cuda_version_out}\n"
        gpu_names = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=subprocess.PIPE,
        )
        gpu_names = gpu_names.stdout.decode().split("\n")
        gpu_names = [name for name in gpu_names if name not in ("", "name")]
        gpu_names = Counter(gpu_names)

        model_str += "# GPU Hardware Info: \n"
        for name, count in gpu_names.items():
            model_str += f"# {name} : {count} \n"
        model_str += "\n"
    else:
        model_str += (
            "torch.cuda.is_available() returned False - no GPU info collected \n"
        )

    model_str += "class Repro(torch.nn.Module):\n"
    attrs = dir(gm)
    for attr in attrs:
        if "_tensor_constant" in attr:
            val = getattr(gm, attr)
            model_str += f"    {attr} = {val!r}\n"
    model_str += textwrap.indent(gm.code, "    ")
    model_str += "\n"

    model_str += f"args = {[(tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.device.type) for arg in args]!r}\n"
    model_str += "args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]\n"
    model_str += "mod = make_fx(Repro())(*args)\n"
    return model_str


INDUCTOR_IMPORT = """
from torchinductor.compile_fx import compile_fx_inner
"""

NVFUSER_IMPORT = """
from torch.fx.passes.backends.nvfuser import NvFuserBackend
nvfuser = NvFuserBackend()
"""

COMPILER_REPRO_OPTIONS = {
    "inductor": (INDUCTOR_IMPORT, "compile_fx_inner", "inductor_fails"),
    "nvfuser": (NVFUSER_IMPORT, "nvfuser", "nvfuser_fails"),
}


def dump_state(gm, args, compiler_name):
    subdir = f"{minifier_dir()}/checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    with open(file_name, "w") as fd:
        fd.write(generate_repro_string(gm, args))
        fd.write(COMPILER_REPRO_OPTIONS[compiler_name][0])
        fd.write(
            textwrap.dedent(
                f"""
                compiled = {COMPILER_REPRO_OPTIONS[compiler_name][1]}(mod, args)
                compiled(*args)
                """
            )
        )
    repro_path = os.path.join(torchdynamo.config.base_dir, "repro.py")
    shutil.copyfile(file_name, repro_path)


def isolate_fails(fx_g, args, compiler_name: str, env=None):
    if env is None:
        env = {}
    subdir = f"{minifier_dir()}/isolate"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        fd.write(generate_repro_string(fx_g, args))
        fail_fn = COMPILER_REPRO_OPTIONS[compiler_name][2]
        fd.write(
            textwrap.dedent(
                f"""
                from torchdynamo.debug_utils import {fail_fn}
                """
            )
        )
        fd.write(
            textwrap.dedent(
                f"""
                if {fail_fn}(mod, args):
                    exit(1)
                else:
                    exit(0)
                """
            )
        )
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    p = subprocess.Popen(
        ["python", file_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=new_env,
    )
    out, err = p.communicate()
    if p.returncode != 0:
        print(textwrap.indent(out.decode("utf-8"), prefix=">>  "))
        print(textwrap.indent(err.decode("utf-8"), prefix=">>  "))
        return True
    return False


def inductor_fails(fx_g, args, check_str=None):
    from torchinductor import config
    from torchinductor.compile_fx import compile_fx_inner

    config.triton.autotune = False

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any([isinstance(x, (tuple, list)) for x in result])
    except Exception:
        return False

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def nvfuser_fails(fx_g, args, check_str=None):
    from torch.fx.passes.backends.nvfuser import NvFuserBackend

    nvfuser = NvFuserBackend()

    try:
        compile_mod = nvfuser(fx_g, args)
        compile_mod = compile_mod(*args)
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def dump_to_minify(gm, args, compiler_name: str):
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    with open(
        os.path.join(torchdynamo.config.base_dir, "minifier_launcher.py"), "w"
    ) as fd:
        fd.write(generate_repro_string(gm, args))
        fd.write("\n")
        fd.write(
            textwrap.dedent(
                f"""
                from functools import partial
                from torchdynamo.debug_utils import (
                    isolate_fails,
                    dump_state,
                )
                from functorch.compile import minifier

                env_variables = {{"CUDA_VISIBLE_DEVICES": "{favored_device}"}}

                minifier(
                    mod,
                    args,
                    module_fails=partial(isolate_fails, env=env_variables, compiler_name="{compiler_name}"),
                    dump_state=partial(dump_state, compiler_name="{compiler_name}"),
                )
                """
            )
        )
    print("wrote out to minifier_launcher.py")


def wrap_debug(compiler, compiler_name: str):
    from torchdynamo.testing import reduce_to_scalar_loss

    @functools.wraps(compiler)
    def debug_wrapper(gm, example_inputs, **kwargs):
        orig_graph = copy.deepcopy(gm.graph)
        if config.repro_level == 3:
            dump_to_minify(
                fx.GraphModule(gm, orig_graph), example_inputs, compiler_name
            )

        try:
            compiled_fn = compiler(gm, example_inputs, **kwargs)
            if config.repro_level > 0:
                compiled_fn(*example_inputs)
        except Exception as e:
            if config.repro_level == 1:
                dump_state(
                    fx.GraphModule(gm, orig_graph), example_inputs, compiler_name
                )
            elif config.repro_level == 2:
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), example_inputs, compiler_name
                )
            raise e

        return compiled_fn

    return debug_wrapper


def fwd_bwd(gm, args):
    # TODO - Read the grad state to avoid bwd pass if not needed
    from torchdynamo.testing import reduce_to_scalar_loss

    out = gm(*args)
    reduce_to_scalar_loss(out).backward()
    return out


def generate_dynamo_repro_string(gm, args, compiler_name):

    imports = textwrap.dedent(
        """
        import torch
        import torchdynamo
        from torchdynamo.testing import rand_strided
        from torchdynamo.debug_utils import fwd_bwd

        """
    )

    prep_inputs = textwrap.dedent(
        f"""
        args = {[(tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.device.type, arg.requires_grad) for arg in args]}
        args = [rand_strided(shape, stride, dtype, device).requires_grad_(requires_grad) for (shape, stride, dtype, device, requires_grad) in args]

        """
    )

    setup_module = textwrap.dedent(
        f"""
        import module
        mod = module.ReproModule().cuda()
        opt_mod = torchdynamo.optimize("{compiler_name}")(mod)

        """
    )

    # TODO - Figure out the amp state
    run_module = textwrap.dedent(
        """
        with torch.cuda.amp.autocast():
            ref = fwd_bwd(mod, args)
            res = fwd_bwd(opt_mod, args)
        """
    )

    return imports + prep_inputs + setup_module + run_module


def dump_dynamo_state(gm, args, compiler_name):
    import tarfile
    subdir = f"{minifier_dir()}/checkpoints"
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    tmp_dir = os.path.join(subdir, f"{len(gm.graph.nodes)}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    file_name = os.path.join(tmp_dir, "repro.py")
    gm_dir = os.path.join(tmp_dir, "module")
    if not os.path.exists(gm_dir):
        os.makedirs(gm_dir, exist_ok=True)
    print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    for node in gm.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs
    gm.recompile()

    with open(file_name, "w") as fd:
        gm.to_folder(gm_dir, "ReproModule")
        fd.write(generate_dynamo_repro_string(gm, args, compiler_name))
    local_dir = os.path.join(torchdynamo.config.base_dir, "repro")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    shutil.copytree(tmp_dir, local_dir)
    local_tar_file = os.path.join(torchdynamo.config.base_dir, "repro.tar.gz")
    with tarfile.open(local_tar_file, "w:gz") as tar:
        tar.add(local_dir, arcname=os.path.basename(local_dir))


# This one might
def wrap_dynamo_debug(compiler, compiler_name: str):
    from difflib import SequenceMatcher

    @functools.wraps(compiler)
    def debug_wrapper(gm, example_inputs, **kwargs):
        """
        Find the smallest Fx GraphModule that leads to the same error
        """
        compiled_gm = compiler(gm, example_inputs, **kwargs)
        if config.repro_level > 0:
            config.raise_on_backend_error = True
            try:
                fwd_bwd(compiled_gm, example_inputs)
            except Exception as exc:
                print("Original model failed with the following error", exc)
                dump_state = functools.partial(
                    dump_dynamo_state, compiler_name=compiler_name
                )
                if config.repro_level == 1:
                    dump_state(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                    )
                else:
                    # We cannot dump to minifier because minifier expects an fx
                    # graph module. The one above works because make_fx is used
                    # to generate Fx graph module. We cannot do this here
                    # because it alters the behavior.

                    orig_failure = str(exc)

                    def fails(mod, inps):
                        # TODO - Add accuracy chec
                        try:
                            fwd_bwd(compiler(mod, example_inputs, **kwargs), inps)
                            return False
                        except Exception as e:
                            new_failure = str(e)
                            if (
                                SequenceMatcher(None, orig_failure, new_failure).ratio()
                                > 0.5
                            ):
                                return True
                            return False

                    from functorch.compile import minifier

                    minifier(gm, example_inputs, fails, dump_state=dump_state)
                    raise exc

        return compiled_gm

    return debug_wrapper
