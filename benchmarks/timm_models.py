#!/usr/bin/env python
import importlib
import logging
import os
import re
import subprocess
import sys
import warnings

import torch
from common import BenchmarkRunner
from common import main
from torch._subclasses import FakeTensor

from torchdynamo.testing import collect_results
from torchdynamo.utils import clone_inputs


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    importlib.import_module("timm")
except ModuleNotFoundError:
    print("Installing Pytorch Image Models...")
    pip_install("git+https://github.com/rwightman/pytorch-image-models")
finally:
    from timm.data import resolve_data_config
    from timm.models import create_model

TIMM_MODELS = dict()
filename = "timm_models_list.txt"
if os.path.exists("benchmarks"):
    filename = "benchmarks/" + filename
with open(filename, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(" ")
        TIMM_MODELS[model_name] = int(batch_size)


# TODO - Figure out the reason of cold start memory spike
BATCH_SIZE_DIVISORS = {
    "beit_base_patch16_224": 2,
    "cait_m36_384": 4,
    "convit_base": 4,
    "convmixer_768_32": 2,
    "convnext_base": 4,
    "crossvit_9_240": 2,
    "cspdarknet53": 2,
    "deit_base_distilled_patch16_224": 2,
    "densenet121": 2,
    "dla102": 2,
    "dpn107": 2,
    "ecaresnet101d": 2,
    "gluon_senet154": 2,
    "gluon_xception65": 2,
    "gmixer_24_224": 2,
    "gmlp_s16_224": 2,
    "hrnet_w18": 64,
    "jx_nest_base": 2,
    "legacy_senet154": 2,
    "mixer_b16_224": 2,
    "mixnet_l": 2,
    "mobilevit_s": 2,
    "nasnetalarge": 2,
    "pit_b_224": 2,
    "pnasnet5large": 2,
    "poolformer_m36": 2,
    "res2net101_26w_4s": 2,
    "res2net50_14w_8s": 64,
    "res2next50": 64,
    "resnest101e": 4,
    "sebotnet33ts_256": 2,
    "swin_base_patch4_window7_224": 2,
    "swsl_resnext101_32x16d": 1,
    "tf_mixnet_l": 2,
    "tnt_s_patch16_224": 2,
    "twins_pcpvt_base": 4,
    "vit_base_patch16_224": 2,
    "volo_d1_224": 2,
    "xcit_large_24_p8_224": 4,
}

REQUIRE_HIGHER_TOLERANCE = set()

SKIP = {
    # Unusual training setup
    "levit_128",
}


def refresh_model_names():
    import glob

    from timm.models import list_models

    def read_models_from_docs():
        models = set()
        # TODO - set the path to pytorch-image-models repo
        for fn in glob.glob("../pytorch-image-models/docs/models/*.md"):
            with open(fn, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.startswith("model = timm.create_model("):
                        continue

                    model = line.split("'")[1]
                    # print(model)
                    models.add(model)
        return models

    def get_family_name(name):
        known_families = [
            "darknet",
            "densenet",
            "dla",
            "dpn",
            "ecaresnet",
            "halo",
            "regnet",
            "efficientnet",
            "deit",
            "mobilevit",
            "mnasnet",
            "convnext",
            "resnet",
            "resnest",
            "resnext",
            "selecsls",
            "vgg",
            "xception",
        ]

        for known_family in known_families:
            if known_family in name:
                return known_family

        if name.startswith("gluon_"):
            return "gluon_" + name.split("_")[1]
        return name.split("_")[0]

    def populate_family(models):
        family = dict()
        for model_name in models:
            family_name = get_family_name(model_name)
            if family_name not in family:
                family[family_name] = []
            family[family_name].append(model_name)
        return family

    docs_models = read_models_from_docs()
    all_models = list_models(pretrained=True, exclude_filters=["*in21k"])

    all_models_family = populate_family(all_models)
    docs_models_family = populate_family(docs_models)

    # print(docs_models_family.keys())
    for key in docs_models_family:
        del all_models_family[key]

    chosen_models = set()
    for value in docs_models_family.values():
        chosen_models.add(value[0])

    for key, value in all_models_family.items():
        chosen_models.add(value[0])

    filename = "timm_models_list.txt"
    if os.path.exists("benchmarks"):
        filename = "benchmarks/" + filename
    with open(filename, "w") as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + "\n")


class TimmRunnner(BenchmarkRunner):
    def __init__(self):
        super(TimmRunnner, self).__init__()

    def load_model(
        self,
        device,
        model_name,
        is_training,
        use_eval_mode,
        batch_size=None,
        dynamic_shapes=False,
    ):

        _, model_dtype, data_dtype = self.resolve_precision()
        channels_last = self._args.channels_last

        model = create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            # global_pool=kwargs.pop('gp', 'fast'),
            # num_classes=kwargs.pop('num_classes', None),
            # drop_rate=kwargs.pop('drop', 0.),
            # drop_path_rate=kwargs.pop('drop_path', None),
            # drop_block_rate=kwargs.pop('drop_block', None),
        )
        model.to(
            device=device,
            dtype=model_dtype,
            memory_format=torch.channels_last if channels_last else None,
        )

        self.num_classes = model.num_classes

        data_config = resolve_data_config(
            self._args, model=model, use_test_size=not is_training
        )
        input_size = data_config["input_size"]
        recorded_batch_size = TIMM_MODELS[model_name]
        recorded_batch_size = max(
            int(recorded_batch_size / BATCH_SIZE_DIVISORS.get(model_name, 1)), 1
        )
        batch_size = batch_size or recorded_batch_size

        # example_inputs = torch.randn(
        #     (batch_size,) + input_size, device=device, dtype=data_dtype
        # )
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=device
        ).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev

        if channels_last:
            example_inputs = example_inputs.contiguous(
                memory_format=torch.channels_last
            )
        example_inputs = [
            example_inputs,
        ]
        self.target = self._gen_target(batch_size, device)

        self.loss = torch.nn.CrossEntropyLoss().to(device)
        if is_training and not use_eval_mode:
            model.train()
        else:
            model.eval()

        return device, model_name, model, example_inputs, batch_size

    def iter_models(self, args):
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device,
                        model_name,
                        args.training,
                        args.use_eval_mode,
                        args.batch_size,
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def iter_model_names(self, args):
        # for model_name in list_models(pretrained=True, exclude_filters=["*in21k"]):
        model_names = sorted(TIMM_MODELS.keys())
        start, end = self.get_benchmark_indices(len(model_names))
        for index, model_name in enumerate(model_names):
            if index < start or index >= end:
                continue
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in self.skip_models
            ):
                continue

            yield model_name

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        tolerance = 1e-3
        if is_training:
            if REQUIRE_HIGHER_TOLERANCE:
                tolerance = 2 * 1e-2
            else:
                tolerance = 1e-2
        return tolerance, cosine

    def _gen_target(self, batch_size, device):
        # return torch.ones((batch_size,) + (), device=device, dtype=torch.long)
        return torch.empty((batch_size,) + (), device=device, dtype=torch.long).random_(
            self.num_classes
        )

    def compute_loss(self, pred):
        if isinstance(pred, FakeTensor) and not isinstance(self.target, FakeTensor):
            return self.loss(pred, torch.ops.aten.lift_fresh_copy(self.target))
        else:
            return self.loss(pred, self.target)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        mod.zero_grad(True)
        with self.autocast():
            pred = mod(*cloned_inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TimmRunnner())
