'''
Created on 24 July 2026

check package resources and standalone workspace paths

@author: Dinghao Luo
'''

#%% imports
from importlib.resources import files
from pathlib import Path
from unittest import mock
import os
import tempfile
import unittest

from support import PROJECT_ROOT, add_source_to_path

add_source_to_path()

from fibre_sight._repo import get_workspace_root, package_path
from fibre_sight.api import get_default_checkpoint, get_model_entry


#%% tests
class ResourcePathTests(unittest.TestCase):
    def test_public_resources_are_packaged(self):
        package = files('fibre_sight')
        resources = [
            ('configs', 'model_registry.yaml'),
            ('models', 'fibre_sight_ch2_v1.pt'),
            ('assets', 'fibresight_icon.ico'),
            ('assets', 'fonts', 'mononoki', 'LICENSE'),
            ('assets', 'fonts', 'mononoki', 'mononoki-Regular.ttf'),
            ]

        for parts in resources:
            with self.subTest(resource='/'.join(parts)):
                self.assertTrue(package.joinpath(*parts).is_file())

    def test_registry_resolves_the_public_checkpoint(self):
        entry = get_model_entry('ch2_v1')
        checkpoint = get_default_checkpoint('ch2_v1')

        self.assertEqual(entry['checkpoint'], 'models/fibre_sight_ch2_v1.pt')
        self.assertEqual(entry['threshold'], 0.25)
        self.assertEqual(entry['min_size'], 45)
        self.assertTrue(entry['tta'])
        self.assertEqual(checkpoint.resolve(), package_path('models', 'fibre_sight_ch2_v1.pt').resolve())
        self.assertTrue(checkpoint.is_file())

    def test_workspace_override_accepts_spaces(self):
        with tempfile.TemporaryDirectory(prefix='fibre sight workspace ') as temp_dir:
            workspace = Path(temp_dir) / 'chosen workspace'
            with mock.patch.dict(os.environ, {'FIBRE_SIGHT_WORKSPACE': str(workspace)}):
                self.assertEqual(get_workspace_root(), workspace)

    def test_checkout_root_is_not_used_as_package_data(self):
        checkpoint = get_default_checkpoint('ch2_v1').resolve()
        self.assertTrue(checkpoint.is_relative_to((PROJECT_ROOT / 'src' / 'fibre_sight').resolve()))


if __name__ == '__main__':
    unittest.main()
