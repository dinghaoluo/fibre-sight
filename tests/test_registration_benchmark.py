'''
Created on 14 August 2026
Modified on 16 August 2026
Modified on 17 August 2026 to follow the benchmark directory
Modified on 18 August 2026

check synthetic motion truth and benchmark errors

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.registration_benchmark import (
    SOURCE_ROOT,
    EXAMPLE_REFERENCES,
    MOTION_RECIPES,
    _align_reference,
    _fit_reference,
    _gradient_ncc,
    _normalise,
    _patchwarp_grid,
    _piecewise_movement,
    _pyflowreg_grid,
    _write_patchwarp_tiff,
    _warp,
    displacement_at,
    make_planes,
    measure_reference_convergence,
    motion_truth,
    registration_errors,
    )
from benchmarking.registration_benchmark_suite import (
    RECIPES,
    _check_sweep,
    _episode_lengths,
    _focal_counts,
    _focal_row,
    _local_movement,
    _paired_values,
    _source_values,
    _supports_provisional_status,
    benchmark_cases,
    bleaching_cases,
    focal_cases,
    intensity_movies,
    photon_cases,
    )


#%% tests
class RegistrationBenchmarkTests(unittest.TestCase):
    def test_intensity_sweep_requires_every_heldout_method(self):
        cases = [{'case': 'a'}, {'case': 'b'}]
        methods = ('first', 'second')
        rows = [
            {'case': case['case'], 'method': method, 'group': 'heldout'}
            for case in cases for method in methods
            ]

        _check_sweep(rows, cases, methods)
        with self.assertRaisesRegex(ValueError, '3 of 4'):
            _check_sweep(rows[:-1], cases, methods)

    def test_source_pairing_keeps_complete_shared_sources(self):
        rows = []
        for source, methods in (
                ('complete', ('fibresight_rigid', 'suite2p_rigid')),
                ('missing_competitor', ('fibresight_rigid',)),
                ):
            for method in methods:
                for recipe_i, recipe in enumerate(RECIPES):
                    rows.append({
                        'case': f'{source}/{recipe}',
                        'source': source,
                        'recipe': recipe,
                        'method': method,
                        'error': str(recipe_i + (method == 'suite2p_rigid')),
                        })

        fibresight = _source_values(rows, 'fibresight_rigid', 'error')
        suite2p = _source_values(rows, 'suite2p_rigid', 'error')
        fibresight_values, suite2p_values = _paired_values(fibresight, suite2p)

        np.testing.assert_allclose(fibresight_values, [1.5])
        np.testing.assert_allclose(suite2p_values, [2.5])

    def test_intensity_sweeps_reuse_motion_truth(self):
        photons = photon_cases()
        bleaching = bleaching_cases()
        self.assertEqual((len(photons), len(bleaching), len(intensity_movies())), (36, 9, 42))
        self.assertEqual(
            len({case['root'] for case in photons} & {case['root'] for case in bleaching}), 3)

        families = {(case['source'], case['recipe']) for case in photons}
        for family in families:
            seeds = {
                case['seed'] for case in photons
                if (case['source'], case['recipe']) == family
                }
            self.assertEqual(len(seeds), 1)

    def test_suite_uses_each_reference_with_each_motion_recipe(self):
        cases = benchmark_cases()
        combinations = {(case['source'], case['recipe']) for case in cases}

        self.assertEqual(len(MOTION_RECIPES), 4)
        self.assertEqual(len(combinations), 40)
        self.assertEqual(len({case['seed'] for case in cases}), 40)
        self.assertEqual(cases, benchmark_cases())

    def test_motion_recipes_cover_the_intended_recording_conditions(self):
        truths = {
            (case['source'], case['recipe']): motion_truth(
                2000,
                (64, 80),
                seed=np.random.SeedSequence(case['seed']).spawn(4)[0],
                recipe=case['recipe'],
                )
            for case in benchmark_cases()
            }
        half = 1000

        for source in EXAMPLE_REFERENCES:
            ordinary = truths[source, 'ordinary_motion']
            large = truths[source, 'large_motion']
            local = truths[source, 'local_deformation']
            focal = truths[source, 'focal_change']

            ordinary_range = np.percentile(
                np.hypot(ordinary['shift_y'], ordinary['shift_x']), 95)
            large_range = np.percentile(
                np.hypot(large['shift_y'], large['shift_x']), 95)
            self.assertGreater(large_range, ordinary_range + 2)
            self.assertFalse((ordinary['nonrigid'] | ordinary['focal']).any())
            self.assertEqual(set(ordinary['scenario']), {'rigid'})
            self.assertFalse(local['focal'].any())
            self.assertLess(_local_movement(local).max(), 3)
            self.assertTrue(np.all((_episode_lengths(focal['focal']) >= 8)))
            self.assertTrue(np.all((_episode_lengths(focal['focal']) <= 15)))

            for flag in (
                    local['nonrigid'], focal['nonrigid'],
                    focal['focal'], focal['ambiguous'],
                    ):
                self.assertTrue(flag[:half].any() and flag[half:].any())

    def test_optical_defocus_keeps_the_competitor_suite_unchanged(self):
        truth = motion_truth(2000, (64, 80), recipe='optical_defocus')
        half = len(truth['focal']) // 2

        self.assertNotIn('optical_defocus', MOTION_RECIPES)
        self.assertEqual(len(focal_cases()), len(EXAMPLE_REFERENCES))
        self.assertEqual(len(_episode_lengths(truth['focal'])), 8)
        np.testing.assert_array_equal(
            _episode_lengths(truth['focal']), [9, 12, 15, 15] * 2)
        self.assertEqual(truth['defocus_sigma_px'][truth['focal']].min(), 2)
        self.assertEqual(truth['defocus_sigma_px'].max(), 8)
        self.assertAlmostEqual(
            float(truth['contrast_loss_fraction'][truth['focal']].min()), 0.10)
        self.assertAlmostEqual(float(truth['contrast_loss_fraction'].max()), 0.30)
        self.assertEqual((truth['blank_frame'].sum(), truth['saturated_frame'].sum()), (4, 4))
        self.assertFalse((truth['blank_frame'] & truth['focal']).any())
        self.assertFalse((truth['saturated_frame'] & truth['focal']).any())
        self.assertFalse(truth['nonrigid'].any())
        for field in ('focal', 'blank_frame', 'saturated_frame'):
            self.assertTrue(truth[field][:half].any() and truth[field][half:].any())

    def test_focal_status_requires_specific_episode_detection(self):
        focal = np.zeros(30, dtype=bool)
        focal[3:6] = True
        focal[18:21] = True
        blank = np.zeros(30, dtype=bool)
        saturated = np.zeros(30, dtype=bool)
        blank[0] = True
        saturated[1] = True
        truth = {
            'focal': focal,
            'blank_frame': blank,
            'saturated_frame': saturated,
            }
        all_nonartefact = ~(blank | saturated)

        perfect = _focal_row(
            'test', 'heldout', _focal_counts(truth, focal, 0, len(focal)))
        always_focal = _focal_row(
            'test', 'heldout',
            _focal_counts(truth, all_nonartefact, 0, len(focal)))

        self.assertTrue(_supports_provisional_status(perfect))
        self.assertFalse(_supports_provisional_status(always_focal))
        self.assertEqual(always_focal['matched_episodes'], 1)

    def test_each_central_plane_comes_from_its_named_source(self):
        for source in EXAMPLE_REFERENCES:
            control = np.load(SOURCE_ROOT / source)
            outer = [
                np.load(SOURCE_ROOT / name)
                for name in EXAMPLE_REFERENCES if name != source
                ]
            planes = make_planes(control, *outer)
            np.testing.assert_allclose(planes[2], _normalise(control))

    def test_outer_planes_follow_a_smaller_control_shape(self):
        control = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        outer = [
            np.arange(256 * 256, dtype=np.float32).reshape(256, 256),
            np.flipud(np.arange(256 * 256, dtype=np.float32).reshape(256, 256)),
            ]

        planes = make_planes(control, *outer)

        self.assertEqual(planes.shape, (5, 128, 128))
        np.testing.assert_allclose(planes[2], _normalise(control))

    def test_synthetic_movement_enters_and_leaves_softly(self):
        truth = motion_truth(2000, (64, 80))
        rigid_step = np.hypot(np.diff(truth['shift_y']), np.diff(truth['shift_x']))

        self.assertLess(rigid_step.max(), 1.5)
        self.assertGreater(np.mean(rigid_step > 0.1), 0.20)
        self.assertGreater(
            np.percentile(np.hypot(truth['shift_y'], truth['shift_x']), 95), 2.5)

    def test_positive_displacement_moves_image_down_and_right(self):
        image = np.zeros((21, 21), dtype=np.float32)
        image[10, 10] = 1
        shifted = _warp(image, 3, 4)

        self.assertEqual(np.unravel_index(np.argmax(shifted), shifted.shape), (13, 14))

    def test_scored_displacement_inverts_the_synthetic_warp(self):
        from scipy.ndimage import map_coordinates

        y, x = np.mgrid[:64, :72].astype(np.float32)
        truth = {
            'shift_y': np.zeros(1),
            'shift_x': np.zeros(1),
            'basis_y': (y - 32)[None],
            'basis_x': (-2 / 3 * (x - 36))[None],
            'coefficient': np.array([[0.06]], dtype=np.float32),
            }
        target_y, target_x = np.meshgrid(
            np.linspace(12, 50, 5), np.linspace(14, 56, 5), indexing='ij')

        movement_y, movement_x = displacement_at(
            truth, np.array([0]), target_y.ravel(), target_x.ravel())
        source_y = target_y.ravel() + movement_y[0]
        source_x = target_x.ravel() + movement_x[0]
        field_y = 0.06 * (y - 32)
        field_x = -0.04 * (x - 36)

        sampled_y = map_coordinates(field_y, (source_y, source_x), order=1)
        sampled_x = map_coordinates(field_x, (source_y, source_x), order=1)
        np.testing.assert_allclose(movement_y[0], sampled_y, atol=1e-6)
        np.testing.assert_allclose(movement_x[0], sampled_x, atol=1e-6)

        latent = (
            np.exp(-((y - 24) ** 2 + (x - 28) ** 2) / 130)
            + 0.6 * np.exp(-((y - 43) ** 2 + (x - 49) ** 2) / 85)
            )
        observed = _warp(latent, field_y, field_x)
        restored = map_coordinates(observed, (source_y, source_x), order=1)
        expected = map_coordinates(
            latent, (target_y.ravel(), target_x.ravel()), order=1)

        np.testing.assert_allclose(restored, expected, atol=2e-3)

    def test_pyflowreg_grid_keeps_row_major_tile_order(self):
        fields = np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9)
        sampled = _pyflowreg_grid(fields, np.array([1, 6]), np.array([2, 7, 8]))
        expected = np.concatenate([
            fields[:, 1, :][:, [2, 7, 8]],
            fields[:, 6, :][:, [2, 7, 8]],
            ], axis=1)
        np.testing.assert_array_equal(sampled, expected)

    def test_patchwarp_grid_averages_overlapping_affine_patches(self):
        warp_cell = np.empty((2, 2, 2), dtype=object)
        shifts = ((1, 10), (3, 30), (5, 50), (7, 70))
        for block_i, (x, y) in enumerate(shifts):
            block_y, block_x = divmod(block_i, 2)
            for frame_i in range(2):
                warp_cell[block_y, block_x, frame_i] = np.array([
                    [1, 0, x + frame_i],
                    [0, 1, y + frame_i],
                    ], dtype=np.float32)
        patch_y = np.empty((2, 2), dtype=object)
        patch_x = np.empty((2, 2), dtype=object)
        for block_y in range(2):
            for block_x in range(2):
                patch_y[block_y, block_x] = np.arange(1 + 2 * block_y, 5 + 2 * block_y)
                patch_x[block_y, block_x] = np.arange(1 + 2 * block_x, 5 + 2 * block_x)

        local_y, local_x = _patchwarp_grid(
            warp_cell,
            patch_y,
            patch_x,
            np.ones(6, dtype=bool),
            np.ones(6, dtype=bool),
            np.array([0, 2]),
            np.array([0, 2]),
            )

        np.testing.assert_allclose(local_y[:, [0, 3]], [[10, 40], [11, 41]])
        np.testing.assert_allclose(local_x[:, [0, 3]], [[1, 4], [2, 5]])

    def test_patchwarp_tiff_keeps_pixels_and_repeats_its_description(self):
        movie = np.arange(4 * 7 * 9, dtype=np.int16).reshape(4, 7, 9) - 80
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'control.tif'
            _write_patchwarp_tiff(movie, path)
            with tifffile.TiffFile(path) as tiff:
                descriptions = [page.description for page in tiff.pages]
                written = tiff.asarray()

        np.testing.assert_array_equal(written, movie)
        self.assertEqual(
            descriptions, ['FibreSight registration benchmark'] * len(movie))

    def test_reference_offset_is_fitted_on_calibration_frames_only(self):
        truth_y = np.array([0, 1, 2, 3], dtype=float)
        truth_x = np.array([0, -1, -2, -3], dtype=float)
        estimate_y = truth_y + np.array([5, 5, 7, 7])
        estimate_x = truth_x + np.array([-2, -2, -1, -1])
        calibration = np.array([True, True, False, False])
        estimable = np.ones(4, dtype=bool)

        error, offset_y, offset_x, valid = registration_errors(
            estimate_y, estimate_x, truth_y, truth_x, calibration, estimable)

        self.assertEqual((offset_y, offset_x), (5, -2))
        np.testing.assert_allclose(error, [0, 0, np.sqrt(5), np.sqrt(5)])
        np.testing.assert_array_equal(valid, estimable)

    def test_piecewise_movie_interpolates_saved_tile_movement(self):
        result = {
            'tile_y': np.array([1, 1, 3, 3]),
            'tile_x': np.array([1, 3, 1, 3]),
            'local_y': np.array([[0, 2, 4, 6]], dtype=np.float32),
            'local_x': np.array([[6, 4, 2, 0]], dtype=np.float32),
            }
        offsets = (0, 0)

        movement_y, movement_x = _piecewise_movement(
            result, 0, (5, 5), offsets)

        np.testing.assert_allclose(movement_y[[0, 2, 4], [0, 2, 4]], [0, 3, 6])
        np.testing.assert_allclose(movement_x[[0, 2, 4], [0, 2, 4]], [6, 3, 0])

    def test_reference_comparison_removes_position_and_intensity(self):
        from scipy import ndimage as ndi

        y, x = np.mgrid[:96, :112]
        latent = (
            np.exp(-((y - 30) ** 2 + (x - 42) ** 2) / 90)
            + 0.7 * np.exp(-((y - 66) ** 2 + (x - 76) ** 2) / 55)
            ).astype(np.float32)
        reference = ndi.shift(
            2.4 * latent + 7,
            (2.35, -1.70),
            order=1,
            mode='constant',
            cval=7,
            prefilter=False,
            )
        aligned, shift = _align_reference(reference, latent)
        valid = np.where(np.isfinite(aligned))
        crop = np.s_[valid[0].min():valid[0].max() + 1, valid[1].min():valid[1].max() + 1]
        fitted, _, _ = _fit_reference(aligned, latent, crop)

        np.testing.assert_allclose(shift, (-2.35, 1.70), atol=0.1)
        self.assertGreater(_gradient_ncc(fitted[crop], latent[crop]), 0.995)
        self.assertLess(np.sqrt(np.mean((fitted[crop] - latent[crop]) ** 2)), 0.01)

        reference[:4] = np.nan
        aligned, _ = _align_reference(reference, latent)
        self.assertFalse(np.isfinite(aligned[4, 20:-20]).any())
        self.assertTrue(np.isfinite(aligned[5, 20:-20]).all())

    def test_reference_convergence_uses_the_known_valid_area(self):
        y, x = np.mgrid[:80, :96]
        latent = (
            4 * np.exp(-((y - 29) ** 2 + (x - 39) ** 2) / 70)
            + 3 * np.exp(-((y - 57) ** 2 + (x - 68) ** 2) / 45)
            ).astype(np.float32)
        rng = np.random.default_rng(42)
        reference = latent.copy()
        border = np.ones(latent.shape, dtype=bool)
        border[8:72, 9:87] = False
        reference[border] = rng.normal(0, 0.5, border.sum())

        truth = {
            'latent_reference': latent,
            'valid_bounds': np.tile([8, 72, 9, 87], (4, 1)),
            }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                'fibresight_reference_convergence.npz',
                'suite2p_reference_convergence.npz',
                'caiman_rigid_reference_convergence.npz',
                'caiman_piecewise_reference_convergence.npz',
                'patchwarp_reference_convergence.npz',
                'pyflowreg_reference_convergence.npz',
                )
            for path in paths:
                np.savez_compressed(
                    root / path,
                    frame_count=np.asarray([2, 4]),
                    reference=np.asarray([reference, reference]),
                    seconds=np.asarray([0.1, 0.2]),
                    )
            rows = measure_reference_convergence(root, truth)

        self.assertEqual(
            {row['method'] for row in rows},
            {
                'fibresight_rigid', 'suite2p_rigid', 'caiman_rigid',
                'caiman_piecewise', 'patchwarp_rigid', 'pyflowreg_piecewise',
                },
            )
        np.testing.assert_allclose(
            [row['gradient_ncc'] for row in rows], 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
