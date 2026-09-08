import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from main_ui_files.ExtraArgsUI import ExtraArgsWidget


class ExtraArgsNamesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_loaded_compile_name_is_trimmed(self):
        widget = ExtraArgsWidget()
        self.addCleanup(widget.deleteLater)
        widget.load_args({"extra_args": {"torch_compile ": True}})
        self.assertEqual(widget.args, {"torch_compile": True})
        self.assertEqual(widget.get_validation_errors(), [])

    def test_duplicate_normalized_names_are_rejected(self):
        widget = ExtraArgsWidget()
        self.addCleanup(widget.deleteLater)
        widget.load_args({"extra_args": {"torch_compile": True, " torch_compile ": False}})
        errors = widget.get_validation_errors()
        self.assertTrue(any("duplicate" in error.lower() and "torch_compile" in error for error in errors))

    def test_whitespace_only_name_with_value_is_rejected(self):
        widget = ExtraArgsWidget()
        self.addCleanup(widget.deleteLater)
        widget.load_args({"extra_args": {"   ": True}})
        self.assertEqual(widget.args, {})
        self.assertTrue(widget.get_validation_errors())


if __name__ == "__main__":
    unittest.main()
