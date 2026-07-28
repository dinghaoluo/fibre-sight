'''
Created on 24 July 2026

check --help for the public command-line modules

@author: Dinghao Luo
'''

#%% imports
import subprocess
import sys
import unittest

from support import PROJECT_ROOT, source_environment


#%% constants
CLI_MODULES = (
    'fibre_sight.predict_rois',
    'fibre_sight.build_manifest',
    'fibre_sight.train_unet',
    'fibre_sight.evaluate',
    'fibre_sight.plot_training_data',
    'fibre_sight.plot_model_diagnostics',
    )


#%% tests
class CLIHelpTests(unittest.TestCase):
    def test_module_help(self):
        for module in CLI_MODULES:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, '-m', module, '--help'],
                    cwd=PROJECT_ROOT,
                    env=source_environment(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertIn('usage:', output.lower())


if __name__ == '__main__':
    unittest.main()
