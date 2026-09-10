import toml
import pytest

from toml_training import (
    build_training_command,
    inspect_saved_toml,
    normalize_saved_toml,
    qualify_optimizer_type,
    resolve_training_python,
    run_training,
)


def test_normalize_saved_anima_lokr_preset_applies_ui_transformations(tmp_path):
    source = tmp_path / "anima_lokr.toml"
    source.write_text(
        """
[[subsets]]
name = "presentation label"
image_dir = "/workspace/dataset/Star☆Pokémon/panels"
caption_extension = ".txt"
num_repeats = 1
random_crop_padding_percent = 0.05

[train_mode]
train_mode = "lora"

[general_args.args]
mixed_precision = "bf16"
max_train_steps = 2400

[general_args.dataset_args]
resolution = 1024
batch_size = 7

[network_args.args]
network_dim = 1024
network_alpha = 1024.0

[network_args.args.network_args]
conv_dim = 1024
conv_alpha = 1024.0
algo = "lokr"
factor = "4"

[optimizer_args.args]
optimizer_type = "SinkSGD_adv"
lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
lr_scheduler = "rex_annealing_warm_restarts_(RAWR)"
learning_rate = 0.001
warmup_ratio = 0.05

[optimizer_args.args.lr_scheduler_args]
min_lr = 1e-6
gamma = 0.9
d = 0.89

[optimizer_args.args.optimizer_args]
weight_decay = "0.01"
spectral_normalization = "True"
cautious_wd = "True"

[anima_args.args]
pretrained_model_name_or_path = "/workspace/anima.safetensors"
qwen3 = "/workspace/qwen3-06.safetensors"
vae = "/workspace/vae.safetensors"

[[anima_args.args.resolution_schedule]]
resolution = 512
percent = 40
batch_size = 32

[[anima_args.args.resolution_schedule]]
resolution = 1024
batch_size = 10
""",
        encoding="utf-8",
    )

    config, dataset = normalize_saved_toml(source)

    assert dataset["datasets"][0]["subsets"] == [
        {
            "image_dir": "/workspace/dataset/Star☆Pokémon/panels",
            "caption_extension": ".txt",
            "num_repeats": 1,
            "random_crop_padding_percent": 0.05,
        }
    ]
    assert config["network_module"] == "lycoris.kohya"
    assert config["network_args"] == [
        "conv_dim=1024",
        "conv_alpha=1024.0",
        "algo=lokr",
        "factor=4",
    ]
    assert config["optimizer_type"] == "adv_optm.optim.SinkSGD_adv.SinkSGD_adv"
    assert config["optimizer_args"] == [
        "weight_decay=0.01",
        "spectral_normalization=True",
        "cautious_wd=True",
    ]
    assert config["lr_scheduler_args"] == [
        "min_lr=1e-06",
        "gamma=0.9",
        "d=0.89",
        "warmup_steps=120",
    ]
    assert config["resolution_schedule"] == [
        {"resolution": 512, "percent": 40, "batch_size": 32},
        {"resolution": 1024, "batch_size": 10},
    ]


def test_normalize_applies_scheduler_and_logging_derivations(tmp_path):
    source = tmp_path / "derived.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "/workspace/dataset"
flip_aug = false

[general_args.args]
max_train_steps = 2400
unused_optional = false

[optimizer_args.args]
warmup_ratio = 0.05
lr_scheduler_num_cycles = 2
lr_scheduler_type = "custom.Scheduler"

[optimizer_args.args.lr_scheduler_args]
gamma = 0.9

[saving_args.args]
output_name = "run"

[logging_args.args]
log_prefix_mode = "output_name"
run_name_mode = "output_name"
""",
        encoding="utf-8",
    )

    config, dataset = normalize_saved_toml(source)

    assert "warmup_ratio" not in config
    assert config["lr_scheduler_args"] == [
        "gamma=0.9",
        "warmup_steps=60",
        "first_cycle_max_steps=1200",
    ]
    assert config["log_prefix"] == "run_"
    assert config["wandb_run_name"] == "run"
    assert "log_prefix_mode" not in config
    assert "run_name_mode" not in config
    assert "unused_optional" not in config
    assert "flip_aug" not in dataset["datasets"][0]["subsets"][0]


@pytest.mark.parametrize(
    ("section_args", "nested_args", "expected"),
    [
        ("fa = true", "", "networks.lora_fa"),
        ("", "unit = 4", "networks.dylora"),
        ("guidance_scale = 1.0", "", "networks.lora_flux"),
        ("", "", "networks.lora"),
    ],
)
def test_normalize_derives_network_module(tmp_path, section_args, nested_args, expected):
    source = tmp_path / "network.toml"
    source.write_text(
        f"""
[[subsets]]
image_dir = "/workspace/dataset"

[network_args.args]
{section_args}

[network_args.args.network_args]
{nested_args}
""",
        encoding="utf-8",
    )

    config, _ = normalize_saved_toml(source)

    assert config["network_module"] == expected
    assert "fa" not in config.get("network_args", [])


def test_normalize_preserves_qualified_and_builtin_optimizer_names(tmp_path):
    def normalized_optimizer(value):
        source = tmp_path / f"{value.rsplit('.', 1)[-1]}.toml"
        source.write_text(
            f'''[[subsets]]\nimage_dir = "/workspace/dataset"\n\n[optimizer_args.args]\noptimizer_type = "{value}"\n''',
            encoding="utf-8",
        )
        return normalize_saved_toml(source)[0]["optimizer_type"]

    assert normalized_optimizer("AdamW") == "AdamW"
    assert normalized_optimizer("bitsandbytes.optim.AdamW8bit") == "bitsandbytes.optim.AdamW8bit"


@pytest.mark.parametrize(
    ("optimizer", "qualified"),
    [
        ("ADOPT", "LoraEasyCustomOptimizer.adopt.ADOPT"),
        ("CAME", "LoraEasyCustomOptimizer.came.CAME"),
        ("SinkSGD_adv", "adv_optm.optim.SinkSGD_adv.SinkSGD_adv"),
    ],
)
def test_qualify_optimizer_type_uses_the_backend_registry(optimizer, qualified):
    assert qualify_optimizer_type(optimizer) == qualified


def test_empty_later_section_does_not_overwrite_a_valid_argument(tmp_path):
    source = tmp_path / "empty_override.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "/workspace/dataset"

[general_args.args]
pretrained_model_name_or_path = "/workspace/model.safetensors"

[later_args.args]
pretrained_model_name_or_path = ""
""",
        encoding="utf-8",
    )

    config, _ = normalize_saved_toml(source)

    assert config["pretrained_model_name_or_path"] == "/workspace/model.safetensors"


def test_unknown_subset_field_is_rejected_before_training(tmp_path):
    source = tmp_path / "unknown_subset.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "/workspace/dataset"
unsupported_ui_value = 1

[general_args.args]
mixed_precision = "bf16"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"subsets\[0\].*unsupported_ui_value"):
        normalize_saved_toml(source)


@pytest.mark.parametrize(
    ("toml_body", "family", "train_mode"),
    [
        ("[anima_args.args]\nqwen3 = 'model'", "anima", "lora"),
        ("[flux_args.args]\nguidance_scale = 1.0", "flux", "lora"),
        ("[general_args.args]\nsdxl = true", "sdxl", "lora"),
        ("[general_args.args]\nmixed_precision = 'bf16'", "standard", "lora"),
        (
            "[general_args.args]\nmixed_precision = 'bf16'\n[train_mode]\ntrain_mode = 'textual_inversion'",
            "standard",
            "textual_inversion",
        ),
    ],
)
def test_inspect_saved_toml_detects_launcher_settings(tmp_path, toml_body, family, train_mode):
    source = tmp_path / "mode.toml"
    source.write_text(f"[[subsets]]\nimage_dir = 'dataset'\n{toml_body}\n", encoding="utf-8")

    assert inspect_saved_toml(source) == (family, train_mode)


def test_resolve_training_python_prefers_backend_environment(tmp_path):
    backend_python = tmp_path / "backend" / "sd_scripts" / "venv" / "bin" / "python"
    backend_python.parent.mkdir(parents=True)
    backend_python.touch()

    assert resolve_training_python(tmp_path, "current-python") == str(backend_python.resolve())


def test_resolve_training_python_falls_back_to_current_interpreter(tmp_path):
    assert resolve_training_python(tmp_path, "current-python") == "current-python"


def test_normalize_removes_ui_only_sdxl_launcher_flag(tmp_path):
    source = tmp_path / "sdxl.toml"
    source.write_text(
        "[[subsets]]\nimage_dir = 'dataset'\n[general_args.args]\nsdxl = true\n",
        encoding="utf-8",
    )

    config, _ = normalize_saved_toml(source)

    assert "sdxl" not in config


def test_epoch_based_derived_scheduler_settings_fail_instead_of_being_silently_lost(tmp_path):
    source = tmp_path / "epochs.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "dataset"
num_repeats = 1

[general_args.args]
max_train_epochs = 10

[general_args.dataset_args]
resolution = 1024
batch_size = 1

[optimizer_args.args]
warmup_ratio = 0.05
lr_scheduler_num_cycles = 1
lr_scheduler_type = "custom.Scheduler"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_train_steps"):
        normalize_saved_toml(source)


def test_malformed_argument_group_returns_a_clear_error(tmp_path):
    source = tmp_path / "malformed_group.toml"
    source.write_text(
        "subsets = []\n[general_args]\nargs = 'not a table'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"general_args\.args must be a table"):
        normalize_saved_toml(source)


def test_malformed_nested_argument_mapping_returns_a_clear_error(tmp_path):
    source = tmp_path / "malformed_nested.toml"
    source.write_text(
        "subsets = []\n[network_args.args]\nnetwork_args = 'not a table or array'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="network_args must be a table or array"):
        normalize_saved_toml(source)


def test_normalize_saved_anima_toml(tmp_path):
    source = tmp_path / "anima.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "C:/dataset"
num_repeats = 2

[train_mode]
train_mode = "lora"

[general_args.args]
mixed_precision = "bf16"

[anima_args.args]
pretrained_model_name_or_path = "C:/anima.safetensors"
qwen3 = "C:/qwen3"

[general_args.dataset_args]
resolution = 1024
batch_size = 1
""",
        encoding="utf-8",
    )

    config, dataset = normalize_saved_toml(source)

    assert config["mixed_precision"] == "bf16"
    assert config["pretrained_model_name_or_path"] == "C:/anima.safetensors"
    assert config["qwen3"] == "C:/qwen3"
    assert dataset == {
        "general": {"resolution": 1024, "batch_size": 1},
        "datasets": [{"subsets": [{"image_dir": "C:/dataset", "num_repeats": 2}]}],
    }


def test_build_anima_command_uses_generated_tomls(tmp_path):
    command = build_training_command(
        python="python",
        project_root=tmp_path,
        config_path=tmp_path / "config.toml",
        dataset_path=tmp_path / "dataset.toml",
        anima=True,
    )

    assert command == [
        "python",
        str(tmp_path / "backend" / "sd_scripts" / "anima_train_network.py"),
        f"--config_file={(tmp_path / 'config.toml').resolve()}",
        f"--dataset_config={(tmp_path / 'dataset.toml').resolve()}",
    ]


def test_build_accelerate_command(tmp_path):
    command = build_training_command(
        python="python",
        project_root=tmp_path,
        config_path=tmp_path / "config.toml",
        dataset_path=tmp_path / "dataset.toml",
        anima=True,
        accelerate=True,
        num_processes=2,
        main_process_port=29501,
    )

    assert command[:6] == [
        "python",
        "-m",
        "accelerate.commands.launch",
        "--num_processes=2",
        "--main_process_port=29501",
        str(tmp_path / "backend" / "sd_scripts" / "anima_train_network.py"),
    ]


def test_invalid_saved_toml_is_rejected(tmp_path):
    source = tmp_path / "invalid.toml"
    source.write_text("[unknown]\nvalue = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="saved UI training TOML"):
        normalize_saved_toml(source)


def test_run_training_uses_sd_scripts_working_directory(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))

    run_training(["python", "train.py"], project_root=tmp_path, runner=runner)

    assert calls == [
        (["python", "train.py"], {"cwd": tmp_path / "backend" / "sd_scripts"})
    ]
