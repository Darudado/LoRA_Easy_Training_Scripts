from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from toml_training import (
    build_training_command,
    inspect_saved_toml,
    resolve_training_python,
    run_training,
    write_training_tomls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start LoRA training from a saved UI TOML")
    parser.add_argument("--toml", required=True, type=Path, help="saved UI TOML file")
    family = parser.add_mutually_exclusive_group()
    family.add_argument("--anima", action="store_true", help="use Anima training")
    family.add_argument("--sdxl", action="store_true")
    family.add_argument("--flux", action="store_true")
    parser.add_argument("--train-mode", choices=("lora", "textual_inversion"))
    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--main-process-port", type=int, default=29500)
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_store"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.toml.resolve()
    if not source.is_file():
        print(f"TOML file not found: {source}", file=sys.stderr)
        return 2
    try:
        if args.accelerate and args.num_processes <= 0:
            raise ValueError("num-processes must be greater than zero")

        saved_family, saved_train_mode = inspect_saved_toml(source)
        requested_family = next(
            (family for family in ("anima", "sdxl", "flux") if getattr(args, family)),
            None,
        )
        if requested_family and saved_family != "standard" and requested_family != saved_family:
            raise ValueError(
                f"requested model family '{requested_family}' conflicts with saved TOML model family '{saved_family}'"
            )
        family = requested_family or saved_family
        train_mode = args.train_mode or saved_train_mode

        config_path, dataset_path = write_training_tomls(source, args.output_dir)
        project_root = Path(__file__).resolve().parent
        command = build_training_command(
            python=resolve_training_python(project_root, sys.executable),
            project_root=project_root,
            config_path=config_path,
            dataset_path=dataset_path,
            anima=family == "anima",
            sdxl=family == "sdxl",
            flux=family == "flux",
            train_mode=train_mode,
            accelerate=args.accelerate,
            num_processes=args.num_processes,
            main_process_port=args.main_process_port,
        )
    except (OSError, ValueError) as exc:
        print(f"Unable to prepare training: {exc}", file=sys.stderr)
        return 2

    print(f"Generated {config_path} and {dataset_path}")
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return run_training(command, project_root=project_root).wait()


if __name__ == "__main__":
    raise SystemExit(main())
