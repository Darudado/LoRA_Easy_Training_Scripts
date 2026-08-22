import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSpinBox

from main_ui_files.AnimaUI import AnimaWidget


class AnimaResolutionScheduleUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_schedule_rows_serialize_last_percentage_as_automatic(self):
        widget = AnimaWidget()
        widget.widget.anima_training_box.setChecked(True)
        widget.enable_disable(True)
        widget.schedule_enabled.setChecked(True)
        widget.add_schedule_row()

        first, final = widget._schedule_rows
        first.resolution.setValue(512)
        first.percent.setValue(30)
        first.batch_size.setValue(4)
        final.resolution.setValue(1024)
        final.batch_size.setValue(2)
        widget._sync_resolution_schedule()

        self.assertEqual(
            widget.args["resolution_schedule"],
            [
                {"resolution": 512, "percent": 30, "batch_size": 4},
                {"resolution": 1024, "batch_size": 2},
            ],
        )
        self.assertEqual(final.percent.value(), 70.0)
        self.assertFalse(final.percent.isEnabled())

    def test_multi_resolution_editor_uses_editable_whole_percentages(self):
        widget = AnimaWidget()
        widget.widget.anima_training_box.setChecked(True)
        widget.enable_disable(True)
        widget.schedule_enabled.setChecked(True)
        widget.add_schedule_row()

        first, final = widget._schedule_rows

        self.assertEqual(widget.schedule_group.title(), "Multi Resolution Schedule")
        self.assertIsInstance(first.percent, QSpinBox)
        self.assertEqual((first.percent.minimum(), first.percent.maximum()), (0, 100))
        self.assertTrue(first.resolution.isEnabled())
        self.assertTrue(first.percent.isEnabled())
        self.assertFalse(final.percent.isEnabled())

    def test_loading_schedule_preserves_explicit_percentages(self):
        widget = AnimaWidget()

        widget.load_args(
            {
                "anima_args": {
                    "resolution_schedule": [
                        {"resolution": 512, "percent": 70, "batch_size": 14},
                        {"resolution": 1024, "percent": 20, "batch_size": 7},
                        {"resolution": 1536, "batch_size": 1},
                    ]
                }
            }
        )

        self.assertEqual([row.percent.value() for row in widget._schedule_rows], [70, 20, 10])
        self.assertEqual(
            widget.args["resolution_schedule"],
            [
                {"resolution": 512, "percent": 70, "batch_size": 14},
                {"resolution": 1024, "percent": 20, "batch_size": 7},
                {"resolution": 1536, "batch_size": 1},
            ],
        )


if __name__ == "__main__":
    unittest.main()
