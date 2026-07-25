'''
Created on 24 July 2026
start and close the workbench in an isolated off-screen process

@author: Dinghao Luo
'''

#%% imports
from importlib.util import find_spec
import subprocess
import sys
import tempfile
import unittest

from support import PROJECT_ROOT, source_environment


#%% tests
class GUIStartupTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('PyQt5') is not None, 'PyQt5 is not installed')
    def test_workbench_starts_offscreen(self):
        code = (
            'from pathlib import Path; '
            'from PyQt5.QtWidgets import QApplication; '
            'import fibre_sight.__main__; '
            'from fibre_sight.fibre_sight_workbench_gui import FibreSightWorkbench; '
            'app = QApplication.instance() or QApplication([]); '
            'window = FibreSightWorkbench(); '
            'window.image_path = Path(\'examples/demo_test_ref_mat_ch2.npy\'); '
            'window.recname = \'demo_test\'; '
            'assert window.default_roi_path() == '
            'Path.cwd() / \'workspace/output/demo_predicted_ROI_dict.npy\'; '
            'app.processEvents(); '
            'window.close(); '
            'app.processEvents(); '
            'app.quit()'
            )

        with tempfile.TemporaryDirectory(prefix='fibre sight matplotlib ') as temp_dir:
            env = source_environment()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            env['MPLCONFIGDIR'] = temp_dir
            result = subprocess.run(
                [sys.executable, '-c', code],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)


if __name__ == '__main__':
    unittest.main()
