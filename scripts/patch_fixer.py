import collections
import gc
import os.path
import traceback
from collections import namedtuple

import ldm.modules.attention
import torch
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch import einsum

from modules import shared, devices, script_callbacks, sd_models
from modules.paths import models_path
from modules.sd_hijack_inpainting import do_inpainting_hijack, should_hijack_inpainting
from modules.sd_models import select_checkpoint, load_model_weights

print("Fixing all the things that could have just been a pull request!")
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(models_path, model_dir))

CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])
checkpoints_list = {}
checkpoints_loaded = collections.OrderedDict()

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def fixed_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def get_config(checkpoint_info):
    path = checkpoint_info[0]
    model_config = checkpoint_info.config
    checkpoint_dir = os.path.join(shared.script_path, "extensions", "sd_auto_fix", "configs")
    is_v21 = False
    if model_config == shared.cmd_opts.config:
        try:
            checkpoint = torch.load(path)
            c_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            v2_key = "cond_stage_model.model.ln_final.weight"
            if v2_key in c_dict:
                if "global_step" in checkpoint and checkpoint_info.config == shared.cmd_opts.config:
                    if checkpoint["global_step"] == 875000 or checkpoint["global_step"] == 220000:
                        model_config = os.path.join(checkpoint_dir, "v2-inference.yaml")
                    else:
                        model_config = os.path.join(checkpoint_dir, "v2-inference-v.yaml")
                print(f"V2 Model detected, selecting model config: {model_config}")
            v21_keys = ["callbacks", "lr_schedulers", "native_amp_scaling_state"]
            for v22key in v21_keys:
                if v22key in checkpoint:
                    is_v21 = True
                    break

            del checkpoint
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
            pass
    return model_config, is_v21


def load_model(checkpoint_info=None):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()
    model_config, is_v21 = get_config(checkpoint_info)

    if model_config != shared.cmd_opts.config:
        print(f"Loading config from: {model_config}")

    if shared.sd_model:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        shared.sd_model = None
        gc.collect()
        devices.torch_gc()

    sd_config = OmegaConf.load(model_config)

    if should_hijack_inpainting(checkpoint_info):
        # Hardcoded config for now...
        sd_config.model.target = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
        sd_config.model.params.use_ema = False
        sd_config.model.params.conditioning_key = "hybrid"
        sd_config.model.params.unet_config.params.in_channels = 9

        # Create a "fake" config with a different name so that we know to unload it when switching models.
        checkpoint_info = checkpoint_info._replace(config=checkpoint_info.config.replace(".yaml", "-inpainting.yaml"))

    do_inpainting_hijack()

    sd_model = instantiate_from_config(sd_config.model)
    load_model_weights(sd_model, checkpoint_info)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        sd_model.to(shared.device)

    sd_hijack.model_hijack.hijack(sd_model)

    if is_v21 and not shared.cmd_opts.xformers and not shared.cmd_opts.force_enable_xformers and not shared.cmd_opts.no_half:
        print("Fixing attention for v21 model.")
        ldm.modules.attention.CrossAttention.forward = fixed_forward

    sd_model.eval()
    shared.sd_model = sd_model

    script_callbacks.model_loaded_callback(sd_model)

    print(f"Model loaded.")
    return sd_model


def reload_model_weights(sd_model=None, info=None):
    from modules import lowvram, devices, sd_hijack
    checkpoint_info = info or select_checkpoint()

    if not sd_model:
        sd_model = shared.sd_model

    if sd_model.sd_model_checkpoint == checkpoint_info.filename:
        return

    model_config, is_v21 = get_config(checkpoint_info)
    checkpoint_info = checkpoint_info._replace(config=model_config)
    if sd_model.sd_checkpoint_info.config != model_config or should_hijack_inpainting(
            checkpoint_info) != should_hijack_inpainting(sd_model.sd_checkpoint_info):
        del sd_model
        checkpoints_loaded.clear()
        load_model(checkpoint_info)
        return shared.sd_model

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_model_weights(sd_model, checkpoint_info)

    sd_hijack.model_hijack.hijack(sd_model)

    if is_v21 and not shared.cmd_opts.xformers and not shared.cmd_opts.force_enable_xformers and not shared.cmd_opts.no_half:
        print("Fixing attention for v21 model.")
        ldm.modules.attention.CrossAttention.forward = fixed_forward

    script_callbacks.model_loaded_callback(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print(f"Weights loaded.")
    return sd_model


# This is so effing ridiculous that we have to do this $hit.

sd_models.load_model = load_model
sd_models.reload_model_weights = reload_model_weights
