"""Declared configuration oracles; no learner, market fit or default lookup."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, make_dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

import pytest

DECLARATION = dict(
    n_steps=128,
    batch_size=32,
    n_epochs=10,
    gamma=1.0,
    gae_lambda=0.95,
    learning_rate=0.002,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    pi_layers=(32, 32),
    vf_layers=(32, 32),
    adam_betas=(0.9, 0.999),
    adam_eps=1e-5,
    adam_weight_decay=0.0,
    adam_amsgrad=False,
)
# Literal public contract, independent of producer constants and SB3 defaults.
EXPECTED = {
    "schema": "allocation_ppo_training_protocol_v1",
    "backend": {
        "algorithm": "stable_baselines3.PPO",
        "stable_baselines3": "2.3.2",
        "torch": "2.4.1",
        "device": "cpu",
        "n_envs": 1,
        "num_threads": 1,
        "parameter_dtype": "float32",
    },
    "ppo": {
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "learning_rate": 0.002,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "learning_rate_schedule": "constant_v1",
        "clip_range_schedule": "constant_v1",
        "clip_range_vf_schedule": "none_or_constant_v1",
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": "stable_baselines3.common.buffers.RolloutBuffer",
        "rollout_buffer_kwargs": {},
        "stats_window_size": 100,
        "tensorboard_log": None,
        "verbose": 0,
        "_init_setup_model": True,
    },
    "policy": {
        "class": "MlpPolicy",
        "net_arch": {"pi": [32, 32], "vf": [32, 32]},
        "activation_fn": "torch.nn.Tanh",
        "ortho_init": True,
        "features_extractor_class": "stable_baselines3.common.torch_layers.FlattenExtractor",
        "features_extractor_kwargs": {},
        "share_features_extractor": True,
        "normalize_images": True,
        "log_std_init": 0.0,
        "full_std": True,
        "use_expln": False,
        "squash_output": False,
    },
    "optimizer": {
        "class": "torch.optim.Adam",
        "learning_rate_source": "ppo.learning_rate",
        "betas": [0.9, 0.999],
        "eps": 1e-5,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    },
    "fit": {
        "initialization": "fresh_model_v1",
        "reset_num_timesteps": True,
        "progress_bar": False,
        "log_interval": 1,
        "tb_log_name": "PPO",
    },
}


def owner():
    name = "trade_rl.strategies.rl.allocation_training_protocol"
    try:
        return import_module(name).AllocationPPOTrainingProtocol
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.fail(
                "the explicit immutable allocation training protocol is missing"
            )
        raise


def protocol(**changes):
    return owner()(**(DECLARATION | changes))


def test_literal_resolved_contract_and_independent_digest():
    value = protocol()
    assert value.payload() == EXPECTED
    raw = json.dumps(EXPECTED, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert value.digest == sha256(raw.encode("utf-8")).hexdigest()
    assert owner().from_payload(json.loads(raw)) == value


def test_frozen_tuples_and_detached_payload_reader():
    incoming = deepcopy(EXPECTED)
    value = owner().from_payload(incoming)
    original = value.digest
    incoming["policy"]["net_arch"]["pi"][0] = 999
    incoming["optimizer"]["betas"].clear()
    outgoing = value.payload()
    outgoing["ppo"]["rollout_buffer_kwargs"]["callback"] = "unsafe"
    outgoing["policy"]["net_arch"]["vf"].clear()
    outgoing["backend"].clear()
    assert value.payload() == EXPECTED and value.digest == original
    assert value.pi_layers == (32, 32) and value.adam_betas == (0.9, 0.999)
    with pytest.raises(FrozenInstanceError):
        value.adam_eps = 1e-8
    with pytest.raises(TypeError):
        value.pi_layers[0] = 64


ALTERNATIVES = {
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 1,
    "gae_lambda": 1.0,
    "learning_rate": 0.0003,
    "clip_range": 0.1,
    "clip_range_vf": 0.1,
    "normalize_advantage": False,
    "ent_coef": 0.01,
    "vf_coef": 0.0,
    "max_grad_norm": 1.0,
    "target_kl": 0.02,
    "pi_layers": (16,),
    "vf_layers": (16, 32),
    "adam_betas": (0.8, 0.99),
    "adam_eps": 1e-8,
    "adam_weight_decay": 0.01,
    "adam_amsgrad": True,
}


@pytest.mark.parametrize("field,value", ALTERNATIVES.items())
def test_every_variable_declaration_is_digest_bound(field, value):
    changed = protocol(**{field: value})
    assert changed.digest != protocol().digest
    assert owner().from_payload(changed.payload()) == changed


@pytest.mark.parametrize("field", DECLARATION)
def test_no_required_declaration_inherits_a_library_default(field):
    values = DECLARATION.copy()
    del values[field]
    with pytest.raises(TypeError):
        owner()(**values)


def test_equivalent_scalar_reporting_and_zero_policy_auxiliary_losses():
    equivalent = protocol(gamma=1, learning_rate=Fraction(1, 500), ent_coef=-0.0)
    assert equivalent.digest == protocol().digest
    assert protocol(ent_coef=0, vf_coef=0).vf_coef == 0.0


INVALID = (
    [
        (name, value)
        for name in ("n_steps", "batch_size", "n_epochs")
        for value in (True, 0, -1, 2.0)
    ]
    + [("n_steps", 1), ("batch_size", 1), ("batch_size", 48)]
    + [
        (name, value)
        for name in (
            "gamma",
            "gae_lambda",
            "learning_rate",
            "clip_range",
            "clip_range_vf",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "adam_eps",
            "adam_weight_decay",
        )
        for value in (
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
            "0.2",
            lambda _: 0.2,
        )
    ]
    + [("gamma", 0.99), ("gae_lambda", -0.01), ("gae_lambda", 1.01)]
    + [
        (name, value)
        for name in (
            "learning_rate",
            "clip_range",
            "clip_range_vf",
            "max_grad_norm",
            "target_kl",
            "adam_eps",
        )
        for value in (0, -1, 10**400)
    ]
    + [(name, -0.1) for name in ("ent_coef", "vf_coef", "adam_weight_decay")]
    + [
        (name, value)
        for name in ("normalize_advantage", "adam_amsgrad")
        for value in (0, 1, None, "true")
    ]
    + [
        (name, value)
        for name in ("pi_layers", "vf_layers")
        for value in ([], [32], (), (0,), (True,), (32.0,), "32")
    ]
    + [
        ("adam_betas", value)
        for value in (
            [0.9, 0.999],
            (),
            (0.9,),
            (True, 0.9),
            (-0.1, 0.9),
            (0.9, 1.0),
            (float("nan"), 0.9),
        )
    ]
)


@pytest.mark.parametrize("field,value", INVALID)
def test_constructor_rejects_invalid_numeric_schedule_and_architecture_inputs(
    field, value
):
    with pytest.raises(ValueError):
        protocol(**{field: value})


GROUPS = (
    (),
    ("backend",),
    ("ppo",),
    ("policy",),
    ("policy", "net_arch"),
    ("optimizer",),
    ("fit",),
)


@pytest.mark.parametrize("path", GROUPS)
@pytest.mark.parametrize("mode", ("missing", "extra", "not_mapping"))
def test_reader_closes_every_nested_field_set(path, mode):
    payload = deepcopy(EXPECTED)
    group = payload
    for key in path:
        group = group[key]
    if mode == "missing":
        del group[next(iter(group))]
    elif mode == "extra":
        group["callback"] = "forbidden"
    elif path:
        parent = payload
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = []
    else:
        payload = []
    with pytest.raises(ValueError):
        owner().from_payload(payload)


FIXED_REPLACEMENTS = (
    [("schema", None, "allocation_ppo_training_protocol_v2")]
    + [
        ("backend", name, value)
        for name, value in (
            ("algorithm", "eval"),
            ("stable_baselines3", "2.4.0"),
            ("torch", "2.4.1+cpu"),
            ("device", "cuda"),
            ("n_envs", True),
            ("num_threads", 2),
            ("parameter_dtype", "float64"),
        )
    ]
    + [
        ("ppo", name, value)
        for name, value in (
            ("use_sde", True),
            ("sde_sample_freq", 0),
            ("rollout_buffer_class", "ReplayBuffer"),
            ("rollout_buffer_kwargs", {"gamma": 0.99}),
            ("learning_rate_schedule", "linear"),
            ("clip_range_schedule", lambda _: 0.2),
            ("clip_range_vf_schedule", "constant"),
            ("_init_setup_model", False),
            ("verbose", False),
        )
    ]
    + [
        ("policy", name, value)
        for name, value in (
            ("class", "CnnPolicy"),
            ("activation_fn", "torch.nn.ReLU"),
            ("ortho_init", False),
            ("features_extractor_kwargs", {"width": 8}),
            ("share_features_extractor", False),
            ("normalize_images", False),
            ("full_std", 1),
            ("log_std_init", -0.0),
            ("use_expln", True),
            ("squash_output", True),
        )
    ]
    + [
        ("optimizer", name, value)
        for name, value in (
            ("class", "torch.optim.AdamW"),
            ("learning_rate_source", "other"),
            ("foreach", None),
            ("fused", True),
            ("maximize", True),
            ("capturable", True),
            ("differentiable", True),
        )
    ]
    + [
        ("fit", name, value)
        for name, value in (
            ("initialization", "resume"),
            ("reset_num_timesteps", False),
            ("progress_bar", True),
            ("log_interval", True),
            ("tb_log_name", "other"),
        )
    ]
)


@pytest.mark.parametrize("group,key,value", FIXED_REPLACEMENTS)
def test_reader_cannot_change_pinned_or_dormant_constructor_semantics(
    group, key, value
):
    payload = deepcopy(EXPECTED)
    if key is None:
        payload[group] = value
    else:
        payload[group][key] = value
    with pytest.raises(ValueError):
        owner().from_payload(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("ppo", "gamma"), True),
        (("ppo", "normalize_advantage"), 1),
        (("optimizer", "eps"), True),
        (("optimizer", "betas"), [0.9, True]),
        (("policy", "net_arch"), {"pi": [32], "vf": [True]}),
    ],
)
def test_reader_validates_variable_fields_before_accepting_digestable_json(path, value):
    payload = deepcopy(EXPECTED)
    payload[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        owner().from_payload(payload)


class MutableTuple(tuple):
    def __new__(cls, values):
        result = super().__new__(cls, values)
        result.current = list(values)
        return result

    def __iter__(self):
        return iter(self.current)


@pytest.mark.parametrize("field", ("pi_layers", "vf_layers", "adam_betas"))
def test_constructor_rejects_tuple_subclasses_with_mutable_iteration(field):
    with pytest.raises(ValueError):
        protocol(**{field: MutableTuple(DECLARATION[field])})


@pytest.mark.parametrize(
    "kind", ("path", "enum", "dataclass", "proxy", "tuple", "cycle")
)
def test_reader_rejects_non_native_json_even_if_canonical_helper_converts_it(kind):
    payload = deepcopy(EXPECTED)
    if kind == "path":
        payload["policy"]["class"] = Path("MlpPolicy")
    elif kind == "enum":
        payload["policy"]["class"] = Enum("Policy", {"MLP": "MlpPolicy"}).MLP
    elif kind == "dataclass":
        backend = payload["backend"]
        payload["backend"] = make_dataclass("Backend", [(k, object) for k in backend])(
            **backend
        )
    elif kind == "proxy":
        payload["policy"]["features_extractor_kwargs"] = MappingProxyType({})
    elif kind == "tuple":
        payload["optimizer"]["betas"] = (0.9, 0.999)
    else:
        payload["backend"]["cycle"] = payload
    with pytest.raises(ValueError):
        owner().from_payload(payload)
