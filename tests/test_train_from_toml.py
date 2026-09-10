import subprocess
import sys


def test_dry_run_generates_files_without_starting_training(tmp_path):
    source = tmp_path / "saved.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "C:/dataset"

[general_args.args]
mixed_precision = "bf16"

[anima_args.args]
qwen3 = "C:/qwen3"
""",
        encoding="utf-8",
    )
    output = tmp_path / "generated"

    result = subprocess.run(
        [sys.executable, "train_from_toml.py", "--toml", str(source), "--anima", "--output-dir", str(output), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "anima_train_network.py" in result.stdout
    assert (output / "config.toml").is_file()
    assert (output / "dataset.toml").is_file()


def test_missing_toml_returns_error():
    result = subprocess.run(
        [sys.executable, "train_from_toml.py", "--toml", "missing.toml", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "TOML file not found" in result.stderr


def test_cli_infers_anima_from_saved_toml(tmp_path):
    source = tmp_path / "anima.toml"
    source.write_text(
        """
[[subsets]]
image_dir = "/workspace/dataset"

[anima_args.args]
qwen3 = "/workspace/qwen3.safetensors"
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "train_from_toml.py",
            "--toml",
            str(source),
            "--output-dir",
            str(tmp_path / "generated"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "anima_train_network.py" in result.stdout


def test_cli_rejects_model_family_that_conflicts_with_saved_toml(tmp_path):
    source = tmp_path / "anima.toml"
    source.write_text(
        "[[subsets]]\nimage_dir = 'dataset'\n[anima_args.args]\nqwen3 = 'model'\n",
        encoding="utf-8",
    )
    output = tmp_path / "generated"

    result = subprocess.run(
        [
            sys.executable,
            "train_from_toml.py",
            "--toml",
            str(source),
            "--flux",
            "--output-dir",
            str(output),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "conflicts with saved TOML model family 'anima'" in result.stderr
    assert not output.exists()


def test_cli_rejects_non_positive_accelerate_process_count(tmp_path):
    source = tmp_path / "saved.toml"
    source.write_text(
        "[[subsets]]\nimage_dir = 'dataset'\n[general_args.args]\nmixed_precision = 'bf16'\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "train_from_toml.py",
            "--toml",
            str(source),
            "--accelerate",
            "--num-processes",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "num-processes must be greater than zero" in result.stderr
