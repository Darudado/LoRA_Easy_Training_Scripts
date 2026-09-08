"""Load saved UI TOMLs and build sd-scripts training commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import toml


def normalize_saved_toml(source: Path) -> tuple[dict, dict]:
    """Convert the UI's combined TOML into config and dataset mappings."""
    data = toml.load(source)
    if "subsets" not in data or not any(
        isinstance(value, dict) and ("args" in value or "dataset_args" in value)
        for key, value in data.items()
        if key not in {"subsets", "train_mode"}
    ):
        raise ValueError("expected a saved UI training TOML with subsets and argument sections")

    config: dict = {}
    dataset_general: dict = {}
    for section, values in data.items():
        if section in {"subsets", "train_mode"}:
            continue
        if not isinstance(values, dict):
            continue
        config.update(values.get("args", {}))
        dataset_general.update(values.get("dataset_args", {}))

    dataset = {
        "general": dataset_general,
        "datasets": [{"subsets": data["subsets"]}],
    }
    return config, dataset


def write_training_tomls(source: Path, output_dir: Path) -> tuple[Path, Path]:
    config_values, dataset_values = normalize_saved_toml(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.toml"
    dataset_path = output_dir / "dataset.toml"
    config_path.write_text(toml.dumps(config_values), encoding="utf-8")
    dataset_path.write_text(toml.dumps(dataset_values), encoding="utf-8")
    return config_path, dataset_path


def select_train_script(*, anima: bool, sdxl: bool, flux: bool, train_mode: str) -> str:
    choices = {
        ("lora", False, False, False): "train_network.py",
        ("lora", True, False, False): "sdxl_train_network.py",
        ("lora", False, True, False): "flux_train_network.py",
        ("lora", False, False, True): "anima_train_network.py",
        ("textual_inversion", False, False, False): "train_textual_inversion.py",
        ("textual_inversion", True, False, False): "sdxl_train_textual_inversion.py",
    }
    try:
        return choices[(train_mode, sdxl, flux, anima)]
    except KeyError as exc:
        raise ValueError("unsupported training combination") from exc


def build_training_command(
    *,
    python: str,
    project_root: Path,
    config_path: Path,
    dataset_path: Path,
    anima: bool = False,
    sdxl: bool = False,
    flux: bool = False,
    train_mode: str = "lora",
    accelerate: bool = False,
    num_processes: int = 2,
    main_process_port: int = 29500,
) -> list[str]:
    script = project_root / "backend" / "sd_scripts" / select_train_script(
        anima=anima, sdxl=sdxl, flux=flux, train_mode=train_mode
    )
    if accelerate:
        command = [
            python,
            "-m",
            "accelerate.commands.launch",
            f"--num_processes={num_processes}",
            f"--main_process_port={main_process_port}",
        ]
    else:
        command = [python]
    command.extend(
        [
            str(script.resolve()),
            f"--config_file={config_path.resolve()}",
            f"--dataset_config={dataset_path.resolve()}",
        ]
    )
    return command


def run_training(
    command: list[str],
    *,
    project_root: Path,
    runner: Callable[..., subprocess.Popen] = subprocess.Popen,
):
    return runner(command, cwd=project_root / "backend" / "sd_scripts")
