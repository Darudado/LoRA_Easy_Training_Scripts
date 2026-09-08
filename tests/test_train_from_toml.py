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
