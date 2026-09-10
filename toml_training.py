"""Load saved UI TOMLs and build sd-scripts training commands."""

from __future__ import annotations

import ast
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable

import toml


SUPPORTED_SUBSET_KEYS = {
    "addift_alpha_mask",
    "addift_mask_data_dir",
    "alpha_mask",
    "batch_size",
    "cache_info",
    "caption_dropout_every_n_epochs",
    "caption_dropout_rate",
    "caption_extension",
    "caption_prefix",
    "caption_separator",
    "caption_suffix",
    "caption_tag_dropout_rate",
    "class_tokens",
    "color_aug",
    "conditioning_data_dir",
    "custom_attributes",
    "enable_wildcard",
    "face_crop_aug_range",
    "flip_aug",
    "gamma_aug",
    "gamma_aug_range",
    "gamma_aug_rate",
    "image_dir",
    "is_reg",
    "is_val",
    "keep_tokens",
    "keep_tokens_separator",
    "max_bucket_reso",
    "metadata_file",
    "min_bucket_reso",
    "num_repeats",
    "protected_tags_file",
    "random_crop",
    "random_crop_padding_percent",
    "resize_interpolation",
    "resolution",
    "resolution_jitter_batch_sizes",
    "resolution_jitter_resolutions",
    "resolution_jitter_weights",
    "secondary_separator",
    "shuffle_caption",
    "token_warmup_min",
    "token_warmup_step",
}


def inspect_saved_toml(source: Path) -> tuple[str, str]:
    """Return the model family and training mode encoded by a saved UI TOML."""
    data = toml.load(source)
    families = []
    if data.get("anima_args", {}).get("args"):
        families.append("anima")
    if data.get("flux_args", {}).get("args"):
        families.append("flux")
    if data.get("general_args", {}).get("args", {}).get("sdxl"):
        families.append("sdxl")
    if len(families) > 1:
        raise ValueError(f"saved TOML selects conflicting model families: {', '.join(families)}")

    train_mode = data.get("train_mode", {}).get("train_mode", "lora")
    if train_mode not in {"lora", "textual_inversion"}:
        raise ValueError(f"unsupported training mode in saved TOML: {train_mode}")
    return (families[0] if families else "standard", train_mode)


def resolve_training_python(project_root: Path, current_python: str) -> str:
    """Prefer the backend training environment created by the installers."""
    venv = project_root / "backend" / "sd_scripts" / "venv"
    for relative_path in (("bin", "python"), ("Scripts", "python.exe")):
        candidate = venv.joinpath(*relative_path)
        if candidate.is_file():
            return str(candidate.resolve())
    return current_python


@lru_cache(maxsize=1)
def _custom_optimizer_registry() -> dict[str, str]:
    registry_source = (
        Path(__file__).resolve().parent
        / "backend"
        / "custom_scheduler"
        / "LoraEasyCustomOptimizer"
        / "__init__.py"
    )
    if not registry_source.is_file():
        return {}

    tree = ast.parse(registry_source.read_text(encoding="utf-8"), filename=str(registry_source))
    imports: dict[str, str] = {}
    registered_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                local_name = imported.asname or imported.name
                module = node.module
                if module == "adv_optm.optim":
                    module = f"{module}.{imported.name}"
                imports[local_name] = f"{module}.{imported.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "OPTIMIZER_LIST" for target in targets):
                continue
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                registered_names.update(
                    item.id for item in value.elts if isinstance(item, ast.Name)
                )

    return {
        name.lower(): imports[name]
        for name in registered_names
        if name in imports
    }


def qualify_optimizer_type(optimizer_type: str) -> str:
    """Resolve UI optimizer aliases without importing heavyweight optimizer modules."""
    if "." in optimizer_type:
        return optimizer_type
    return _custom_optimizer_registry().get(optimizer_type.lower(), optimizer_type)


def _format_nested_arg(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _flatten_nested_args(config: dict) -> None:
    for key in ("network_args", "optimizer_args", "lr_scheduler_args"):
        value = config.get(key)
        if isinstance(value, dict):
            config[key] = [f"{name}={_format_nested_arg(item)}" for name, item in value.items()]
        elif value is not None and not isinstance(value, list):
            raise ValueError(f"{key} must be a table or array")


def _set_derived_training_args(config: dict) -> None:
    config.pop("sdxl", None)
    network_args = config.get("network_args", [])
    if config.pop("fa", False):
        config["network_module"] = "networks.lora_fa"
    elif any(item.startswith("algo=") for item in network_args):
        config["network_module"] = "lycoris.kohya"
    elif any(item.startswith("unit=") for item in network_args):
        config["network_module"] = "networks.dylora"
    elif "guidance_scale" in config:
        config["network_module"] = "networks.lora_flux"
    elif "network_module" not in config:
        config["network_module"] = "networks.lora"

    optimizer_type = config.get("optimizer_type")
    if isinstance(optimizer_type, str):
        config["optimizer_type"] = qualify_optimizer_type(optimizer_type)


def _is_omitted_optional(value: object) -> bool:
    return value is None or value is False or (isinstance(value, str) and not value.strip())


def _remove_omitted_optionals(values: dict) -> None:
    for key in [key for key, value in values.items() if _is_omitted_optional(value)]:
        del values[key]


def _apply_logging_modes(config: dict) -> None:
    log_prefix_mode = config.pop("log_prefix_mode", None)
    if log_prefix_mode == "output_name" and config.get("output_name"):
        config["log_prefix"] = f'{config["output_name"]}_'
    elif log_prefix_mode == "disabled":
        config.pop("log_prefix", None)

    run_name_mode = config.pop("run_name_mode", None)
    if run_name_mode == "output_name" and config.get("output_name"):
        config["wandb_run_name"] = config["output_name"]
    elif run_name_mode == "manual" and config.get("run_name"):
        config["wandb_run_name"] = config["run_name"]
    elif run_name_mode == "default":
        config.pop("run_name", None)


def _apply_step_derivations(config: dict) -> None:
    warmup_ratio = config.get("warmup_ratio")
    cycles = config.get("lr_scheduler_num_cycles", 1)
    max_train_steps = config.get("max_train_steps")
    needs_total_steps = warmup_ratio is not None or (
        "lr_scheduler_num_cycles" in config and "lr_scheduler_type" in config
    )
    if needs_total_steps and max_train_steps is None:
        raise ValueError(
            "CLI conversion of warmup/restart ratios requires max_train_steps; "
            "epoch-based step calculation must be saved by the UI backend first"
        )

    config.pop("warmup_ratio", None)
    scheduler_args = config.setdefault("lr_scheduler_args", [])

    if warmup_ratio is not None and max_train_steps is not None:
        warmup_steps = round(max_train_steps * warmup_ratio)
        if "lr_scheduler_type" in config:
            scheduler_args.append(f"warmup_steps={warmup_steps // cycles}")
        else:
            config["lr_warmup_steps"] = warmup_steps

    if "lr_scheduler_num_cycles" in config and "lr_scheduler_type" in config and max_train_steps is not None:
        scheduler_args.append(f"first_cycle_max_steps={max_train_steps // cycles}")

    if not scheduler_args:
        config.pop("lr_scheduler_args", None)


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
        args_values = values.get("args", {})
        dataset_values = values.get("dataset_args", {})
        if not isinstance(args_values, dict):
            raise ValueError(f"{section}.args must be a table")
        if not isinstance(dataset_values, dict):
            raise ValueError(f"{section}.dataset_args must be a table")
        config.update(
            (key, value)
            for key, value in args_values.items()
            if not _is_omitted_optional(value)
        )
        dataset_general.update(
            (key, value)
            for key, value in dataset_values.items()
            if not _is_omitted_optional(value)
        )

    _flatten_nested_args(config)
    _set_derived_training_args(config)
    _apply_logging_modes(config)
    _apply_step_derivations(config)
    _remove_omitted_optionals(config)
    normalized_subsets = []
    for index, subset in enumerate(data["subsets"]):
        if not isinstance(subset, dict):
            raise ValueError(f"subsets[{index}] must be a table")
        unknown_keys = set(subset) - SUPPORTED_SUBSET_KEYS - {"name"}
        if unknown_keys:
            fields = ", ".join(sorted(unknown_keys))
            raise ValueError(f"subsets[{index}] contains unsupported field(s): {fields}")
        normalized_subsets.append(
            {
                key: value
                for key, value in subset.items()
                if key != "name" and not _is_omitted_optional(value)
            }
        )

    dataset = {
        "general": dataset_general,
        "datasets": [{"subsets": normalized_subsets}],
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
