from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from modules.DragDropLineEdit import DragDropLineEdit
from ui_files.AnimaUI import Ui_anima_ui
from modules.BaseWidget import BaseWidget
from pathlib import Path


class AnimaWidget(BaseWidget):
    Toggled = Signal(bool)  # send to general args
    ResolutionScheduleToggled = Signal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Anima Args")
        self.widget = Ui_anima_ui()

        self.name = "anima_args"

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        def setup_file(elem: DragDropLineEdit, selector: QPushButton, folder=False):
            selector_icon = QIcon(str(Path("icons/more-horizontal.svg")))
            if folder:
                elem.setMode("folder")
            else:
                elem.setMode("file", [".ckpt", ".pt", ".safetensors", ".sft", ".pth"])
            elem.highlight = True
            selector.setIcon(selector_icon)

        setup_file(self.widget.dit_model_input, self.widget.dit_model_selector)
        setup_file(self.widget.qwen3_model_input, self.widget.qwen3_model_selector)
        setup_file(self.widget.vae_model_input, self.widget.vae_model_selector)
        setup_file(self.widget.t5_tokenizer_input, self.widget.t5_tokenizer_selector, folder=True)
        self._setup_resolution_schedule_editor()

    def _setup_resolution_schedule_editor(self) -> None:
        self._schedule_rows = []
        self.schedule_group = QGroupBox("Resolution Schedule", self.widget.anima_training_box)
        layout = QVBoxLayout(self.schedule_group)
        description = QLabel(
            "Run resolution stages sequentially. Each stage preserves aspect ratio; large images downscale, "
            "small images never upscale. Percentages use optimizer steps."
        )
        description.setWordWrap(True)
        description.setToolTip("The model, optimizer, and learning-rate schedule remain live between stages.")
        layout.addWidget(description)

        self.schedule_enabled = QCheckBox("Enable resolution schedule")
        self.schedule_enabled.setToolTip("Per-stage batch sizes replace General Args batch size while enabled.")
        layout.addWidget(self.schedule_enabled)

        header = QHBoxLayout()
        for text in ("Resolution", "Steps %", "Batch size", ""):
            label = QLabel(text)
            label.setMinimumWidth(90 if text else 24)
            header.addWidget(label)
        layout.addLayout(header)
        self.schedule_rows_layout = QVBoxLayout()
        layout.addLayout(self.schedule_rows_layout)

        self.add_schedule_button = QPushButton("Add stage")
        self.add_schedule_button.setToolTip("Add another sequential stage. The final stage always receives remaining steps.")
        layout.addWidget(self.add_schedule_button)
        self.schedule_total_label = QLabel()
        layout.addWidget(self.schedule_total_label)
        self.widget.gridLayout_2.addWidget(self.schedule_group, 1, 0, 1, 1)

        self._append_schedule_row(1024, 1)
        self.schedule_enabled.toggled.connect(self._on_schedule_toggled)
        self.add_schedule_button.clicked.connect(self.add_schedule_row)
        self._sync_resolution_schedule()

    def _append_schedule_row(self, resolution: int, batch_size: int, percent: float = 100.0) -> None:
        row_widget = QWidget(self.schedule_group)
        layout = QHBoxLayout(row_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        resolution_input = QSpinBox()
        resolution_input.setRange(64, 16384)
        resolution_input.setSingleStep(64)
        resolution_input.setValue(resolution)
        resolution_input.setToolTip("Maximum stage area, expressed as a square side. Buckets preserve image aspect ratio.")
        percent_input = QDoubleSpinBox()
        percent_input.setRange(0.1, 100.0)
        percent_input.setDecimals(2)
        percent_input.setValue(percent)
        batch_input = QSpinBox()
        batch_input.setRange(1, 1024)
        batch_input.setValue(batch_size)
        remove_button = QPushButton("−")
        remove_button.setToolTip("Remove this resolution stage")
        for control in (resolution_input, percent_input, batch_input, remove_button):
            layout.addWidget(control)
        row = type("ResolutionScheduleRow", (), {})()
        row.widget = row_widget
        row.resolution = resolution_input
        row.percent = percent_input
        row.batch_size = batch_input
        row.remove = remove_button
        self._schedule_rows.append(row)
        self.schedule_rows_layout.addWidget(row_widget)
        resolution_input.valueChanged.connect(self._sync_resolution_schedule)
        percent_input.valueChanged.connect(self._sync_resolution_schedule)
        batch_input.valueChanged.connect(self._sync_resolution_schedule)
        remove_button.clicked.connect(lambda: self._remove_schedule_row(row))

    def add_schedule_row(self) -> None:
        if self._schedule_rows:
            last = self._schedule_rows[-1]
            self._append_schedule_row(last.resolution.value(), last.batch_size.value())
        else:
            self._append_schedule_row(1024, 1)
        self._sync_resolution_schedule()

    def _remove_schedule_row(self, row) -> None:
        if len(self._schedule_rows) == 1:
            return
        self._schedule_rows.remove(row)
        self.schedule_rows_layout.removeWidget(row.widget)
        row.widget.deleteLater()
        self._sync_resolution_schedule()

    def _on_schedule_toggled(self, enabled: bool) -> None:
        self.ResolutionScheduleToggled.emit(enabled)
        self._sync_resolution_schedule()

    def _sync_resolution_schedule(self, *_args) -> None:
        if not self._schedule_rows:
            return
        requested = sum(row.percent.value() for row in self._schedule_rows[:-1])
        remaining = max(0.0, 100.0 - requested)
        final = self._schedule_rows[-1]
        final.percent.blockSignals(True)
        final.percent.setValue(remaining if remaining > 0 else final.percent.minimum())
        final.percent.blockSignals(False)
        final.percent.setEnabled(False)
        self.schedule_total_label.setText(f"Total: {requested + remaining:.2f}% (final stage automatic)")
        self.schedule_total_label.setStyleSheet("color: #b00020;" if requested >= 100.0 else "")
        self.add_schedule_button.setEnabled(requested < 100.0)
        if not self.schedule_enabled.isChecked():
            self.args.pop("resolution_schedule", None)
            return
        if requested >= 100.0:
            self.args.pop("resolution_schedule", None)
            return
        schedule = []
        for row in self._schedule_rows[:-1]:
            schedule.append({"resolution": row.resolution.value(), "percent": row.percent.value(), "batch_size": row.batch_size.value()})
        schedule.append({"resolution": final.resolution.value(), "batch_size": final.batch_size.value()})
        self.args["resolution_schedule"] = schedule

    def setup_connections(self) -> None:
        self.widget.anima_training_box.clicked.connect(self.enable_disable)
        
        # File inputs — keys match sd_scripts argparse names
        self.widget.dit_model_input.textChanged.connect(lambda x: self.edit_args("pretrained_model_name_or_path", x))
        self.widget.dit_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.dit_model_input, "Anima DiT Model", "Model File")
        )
        
        self.widget.qwen3_model_input.textChanged.connect(lambda x: self.edit_args("qwen3", x))
        self.widget.qwen3_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.qwen3_model_input, "Qwen3 Model", "Model File")
        )
        
        self.widget.vae_model_input.textChanged.connect(lambda x: self.edit_args("vae", x))
        self.widget.vae_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.vae_model_input, "VAE Model", "Model File")
        )
        
        self.widget.t5_tokenizer_input.textChanged.connect(lambda x: self.edit_args("t5_tokenizer_path", x, optional=True))
        self.widget.t5_tokenizer_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.t5_tokenizer_input, "T5 Tokenizer Folder")
        )

        # Token Lengths
        self.widget.qwen3_max_token_input.valueChanged.connect(
            lambda x: self.edit_args("qwen3_max_token_length", x)
        )
        self.widget.t5_max_token_input.valueChanged.connect(
            lambda x: self.edit_args("t5_max_token_length", x)
        )

        # Sampling Params
        self.widget.timestep_sampling_selector.currentIndexChanged.connect(self.change_timestep_sampling_type)
        self.widget.discrete_flow_shift_input.valueChanged.connect(
            lambda x: self.edit_args("discrete_flow_shift", x)
        )
        self.widget.sigmoid_scale_input.valueChanged.connect(
            lambda x: self.edit_args("sigmoid_scale", x)
        )

        # VAE / Memory Options
        self.widget.vae_chunk_size_input.valueChanged.connect(self.change_vae_chunk_size)
        self.widget.vae_disable_cache_enable.clicked.connect(
            lambda x: self.edit_args("vae_disable_cache", x, True)
        )
        self.widget.blocks_to_swap_input.valueChanged.connect(self.change_blocks_to_swap)

        # Misc
        self.widget.flash_attn_enable.clicked.connect(self.change_flash_attn)
        self.widget.split_attn_enable.clicked.connect(
            lambda x: self.edit_args("split_attn", x, True)
        )
        self.widget.unsloth_offload_checkpointing.clicked.connect(
            lambda x: self.edit_args("unsloth_offload_checkpointing", x, True)
        )

    def enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.Toggled.emit(checked)
        if not checked:
            return
            
        self.edit_args("pretrained_model_name_or_path", self.widget.dit_model_input.text())
        self.edit_args("qwen3", self.widget.qwen3_model_input.text())
        self.edit_args("vae", self.widget.vae_model_input.text())
        self.edit_args("t5_tokenizer_path", self.widget.t5_tokenizer_input.text(), optional=True)
        
        self.edit_args("qwen3_max_token_length", self.widget.qwen3_max_token_input.value())
        self.edit_args("t5_max_token_length", self.widget.t5_max_token_input.value())
        
        self.change_timestep_sampling_type(self.widget.timestep_sampling_selector.currentIndex())
        
        self.change_vae_chunk_size(self.widget.vae_chunk_size_input.value())
        self.edit_args("vae_disable_cache", self.widget.vae_disable_cache_enable.isChecked(), True)
        self.change_blocks_to_swap(self.widget.blocks_to_swap_input.value())

        self.change_flash_attn(self.widget.flash_attn_enable.isChecked())
        self.edit_args("split_attn", self.widget.split_attn_enable.isChecked(), True)
        self.edit_args("unsloth_offload_checkpointing", self.widget.unsloth_offload_checkpointing.isChecked(), True)
        self._sync_resolution_schedule()

    def external_enable_disable(self, checked: bool) -> None:
        self.args = {}
        self.widget.anima_training_box.setEnabled(not checked)
        if self.widget.anima_training_box.isEnabled() and self.widget.anima_training_box.isChecked():
            self.enable_disable(True)

    def change_timestep_sampling_type(self, index: int) -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        self.edit_args("timestep_sampling", sampling_type)
        
        # Enable/Disable sigmoid scale based on selection
        # sigmoid, shift, and flux_shift all use sigmoid_scale to control concentration
        uses_sigmoid_scale = sampling_type in ("sigmoid", "shift", "flux_shift")
        self.widget.sigmoid_scale_input.setEnabled(uses_sigmoid_scale)
        if uses_sigmoid_scale:
             self.edit_args("sigmoid_scale", self.widget.sigmoid_scale_input.value())
        elif "sigmoid_scale" in self.args:
            del self.args["sigmoid_scale"]
        
        # Enable/Disable discrete flow shift based on selection
        # sigma uses it via the scheduler's internal shifted sigma table
        # shift applies it directly: sigma' = shift * sigma / (1 + (shift-1) * sigma)
        # sigmoid, flux_shift, and uniform do not use discrete_flow_shift
        uses_flow_shift = sampling_type in ("sigma", "shift")
        self.widget.discrete_flow_shift_input.setEnabled(uses_flow_shift)
        if uses_flow_shift:
            self.edit_args("discrete_flow_shift", self.widget.discrete_flow_shift_input.value())
        elif "discrete_flow_shift" in self.args:
            del self.args["discrete_flow_shift"]

    def change_vae_chunk_size(self, value: int) -> None:
        """VAE chunk size: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("vae_chunk_size", value)
        elif "vae_chunk_size" in self.args:
            del self.args["vae_chunk_size"]

    def change_blocks_to_swap(self, value: int) -> None:
        """Blocks to swap: 0 means disabled (None), any positive value is used as-is."""
        if value > 0:
            self.edit_args("blocks_to_swap", value)
        elif "blocks_to_swap" in self.args:
            del self.args["blocks_to_swap"]

    def change_flash_attn(self, checked: bool) -> None:
        """Flash attention: sets attn_mode to 'flash' when enabled, removes it when disabled."""
        if checked:
            self.edit_args("attn_mode", "flash")
        elif "attn_mode" in self.args:
            del self.args["attn_mode"]

    def update_split_attn_from_general(self, xformers_enabled: bool, sdpa_enabled: bool) -> None:
        """Called from ArgsListUI when attention mode changes in GeneralUI.
        
        When xFormers is enabled with Anima, split_attn should be forcibly enabled.
        When SDPA is enabled, split_attn is automatically enabled internally by sd_scripts.
        """
        if xformers_enabled:
            self.widget.split_attn_enable.setChecked(True)
            self.widget.split_attn_enable.setEnabled(False)
            self.edit_args("split_attn", True, True)
        else:
            self.widget.split_attn_enable.setEnabled(True)

    def load_args(self, args: dict) -> bool:
        args: dict = args.get(self.name, {})

        self.widget.anima_training_box.setChecked(bool(args))
        self.widget.dit_model_input.setText(args.get("pretrained_model_name_or_path", ""))
        self.widget.qwen3_model_input.setText(args.get("qwen3", ""))
        self.widget.vae_model_input.setText(args.get("vae", ""))
        self.widget.t5_tokenizer_input.setText(args.get("t5_tokenizer_path", ""))
        
        self.widget.qwen3_max_token_input.setValue(args.get("qwen3_max_token_length", 512))
        self.widget.t5_max_token_input.setValue(args.get("t5_max_token_length", 512))
        
        self.widget.timestep_sampling_selector.setCurrentText(args.get("timestep_sampling", "sigmoid"))
        self.widget.discrete_flow_shift_input.setValue(args.get("discrete_flow_shift", 3.0))
        self.widget.sigmoid_scale_input.setValue(args.get("sigmoid_scale", 1.0))

        self.widget.vae_chunk_size_input.setValue(args.get("vae_chunk_size", 0))
        self.widget.vae_disable_cache_enable.setChecked(args.get("vae_disable_cache", False))
        self.widget.blocks_to_swap_input.setValue(args.get("blocks_to_swap", 0))
        
        # flash_attn is stored as attn_mode="flash"
        self.widget.flash_attn_enable.setChecked(args.get("attn_mode", "") == "flash")
        self.widget.split_attn_enable.setChecked(args.get("split_attn", False))
        self.widget.unsloth_offload_checkpointing.setChecked(args.get("unsloth_offload_checkpointing", False))

        schedule = args.get("resolution_schedule")
        if schedule:
            while len(self._schedule_rows) > 1:
                self._remove_schedule_row(self._schedule_rows[-1])
            first = schedule[0]
            self._schedule_rows[0].resolution.setValue(first.get("resolution", 1024))
            self._schedule_rows[0].batch_size.setValue(first.get("batch_size", 1))
            self._schedule_rows[0].percent.setValue(first.get("percent", 100))
            for stage in schedule[1:]:
                self._append_schedule_row(stage.get("resolution", 1024), stage.get("batch_size", 1), stage.get("percent", 100))
            self.schedule_enabled.setChecked(True)
        else:
            self.schedule_enabled.setChecked(False)

        self.enable_disable(self.widget.anima_training_box.isChecked())
