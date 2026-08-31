import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).parents[1]
    / "oneliner-macro"
    / "python"
    / "oneliner_iree"
    / "pytorch_import.py"
)
SPEC = importlib.util.spec_from_file_location("oneliner_pytorch_importer", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


class TensorArgument:
    pass


class ConstantArgument:
    pass


class ExportedProgram:
    def __init__(self, inputs, outputs, range_constraints=None):
        self.graph_signature = SimpleNamespace(
            input_specs=[
                SimpleNamespace(kind=SimpleNamespace(name="USER_INPUT"), arg=value)
                for value in inputs
            ],
            output_specs=[
                SimpleNamespace(kind=SimpleNamespace(name="USER_OUTPUT"), arg=value)
                for value in outputs
            ],
        )
        self.range_constraints = range_constraints or {}


FAKE_TORCH = SimpleNamespace(
    export=SimpleNamespace(ExportedProgram=ExportedProgram),
)


class PytorchImporterTests(unittest.TestCase):
    def test_accepts_one_static_tensor_input_and_output(self):
        program = ExportedProgram([TensorArgument()], [TensorArgument()])

        IMPORTER.validate_exported_program(program, FAKE_TORCH)

    def test_rejects_multiple_user_inputs(self):
        program = ExportedProgram(
            [TensorArgument(), TensorArgument()],
            [TensorArgument()],
        )

        with self.assertRaisesRegex(ValueError, "exactly one PyTorch user input"):
            IMPORTER.validate_exported_program(program, FAKE_TORCH)

    def test_rejects_non_tensor_output(self):
        program = ExportedProgram([TensorArgument()], [ConstantArgument()])

        with self.assertRaisesRegex(TypeError, "requires a tensor output"):
            IMPORTER.validate_exported_program(program, FAKE_TORCH)

    def test_rejects_dynamic_shapes(self):
        program = ExportedProgram(
            [TensorArgument()],
            [TensorArgument()],
            range_constraints={"s0": "VR[1, 8]"},
        )

        with self.assertRaisesRegex(ValueError, "fixed tensor shapes"):
            IMPORTER.validate_exported_program(program, FAKE_TORCH)


if __name__ == "__main__":
    unittest.main()
