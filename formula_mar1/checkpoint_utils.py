import pickle
from typing import Iterable, Tuple

import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .env import TEAM_OBS_DIM


def _is_legacy_shared_trunk(params) -> bool:
    keys = set(unfreeze(params).keys())
    return "input_norm" in keys and "actor_input_norm" not in keys


def _upgrade_legacy_shared_to_split(params):
    """Best-effort remap old shared-trunk checkpoint keys into split actor/critic keys."""
    p = unfreeze(params)
    out = unfreeze(params)
    if "input_norm" in p:
        out["actor_input_norm"] = p["input_norm"]
        out["critic_input_norm"] = p["input_norm"]
    if "LayerNorm_0" in p:
        out["actor_norm1"] = p["LayerNorm_0"]
        out["critic_norm1"] = p["LayerNorm_0"]
    if "Dense_0" in p:
        out["actor_dense1"] = p["Dense_0"]
        out["critic_dense1"] = p["Dense_0"]
    if "Dense_1" in p:
        out["actor_res0"] = p["Dense_1"]
        out["critic_res0"] = p["Dense_1"]
    if "Dense_2" in p:
        out["actor_res1"] = p["Dense_2"]
        out["critic_res1"] = p["Dense_2"]
    if "Dense_3" in p:
        out["actor_dense_out"] = p["Dense_3"]
        out["critic_dense_out"] = p["Dense_3"]
    return freeze(out)


def _is_compatible(model, params) -> Tuple[bool, str]:
    try:
        model.apply({"params": params}, jnp.zeros((1, TEAM_OBS_DIM), dtype=jnp.float32))
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def load_compatible_params(model, weight_candidates: Iterable[str]):
    """Load first checkpoint that is compatible with current network architecture."""
    errors = []
    for path in weight_candidates:
        try:
            with open(path, "rb") as f:
                params = pickle.load(f)
        except FileNotFoundError:
            continue

        ok, err = _is_compatible(model, params)
        if ok:
            return params, path

        if _is_legacy_shared_trunk(params):
            upgraded = _upgrade_legacy_shared_to_split(params)
            ok2, err2 = _is_compatible(model, upgraded)
            if ok2:
                return upgraded, path
            errors.append((path, "legacy remap failed: " + err2))
        else:
            errors.append((path, err))

    if not errors:
        raise FileNotFoundError("No checkpoint files found in provided candidates.")
    msg = "\n".join(["- {}: {}".format(p, e.splitlines()[0]) for p, e in errors])
    raise RuntimeError(
        "Found checkpoints but none match current network architecture.\n"
        "Train fresh weights with current `networks.py` or use matching script/model version.\n"
        + msg
    )
