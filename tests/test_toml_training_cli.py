import toml
import pytest

from toml_training import build_training_command, normalize_saved_toml, run_training


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
