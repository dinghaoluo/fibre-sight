'''
Created on 15 August 2026
Modified on 16 August 2026
Modified on 17 August 2026 to move the benchmark out of the repository root
Modified on 18 August 2026
Modified on 19 August 2026

make and run the multi-image registration benchmark

@author: Dinghao Luo
'''

#%% imports
from argparse import ArgumentParser
from csv import DictReader, DictWriter
from pathlib import Path
import os
import resource
import subprocess
import sys
import time

import numpy as np

from benchmarking import registration_benchmark as benchmark


#%% paths and cases
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / 'examples'
SUITE_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark-suite'
INTENSITY_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark-intensity'
REAL_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark-real'
FOCAL_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark-focal'
AUTO_ROOT = PROJECT_ROOT / 'workspace' / 'registration-auto-validation'
REAL_TIFF = PROJECT_ROOT / 'workspace' / 'dev' / '1000_85%_z3_fov4_00001_00001.tif'
FIBRESIGHT_PYTHON = Path(sys.executable)
SUITE2P_PYTHON = PROJECT_ROOT / 'workspace' / 'dev' / 'envs' / 'suite2p-0.11.1' / 'bin' / 'python'
CAIMAN_PYTHON = PROJECT_ROOT / 'workspace' / 'dev' / 'envs' / 'caiman-1.12.2' / 'bin' / 'python'
PYFLOWREG_PYTHON = PROJECT_ROOT / 'workspace' / 'dev' / 'envs' / 'pyflowreg-0.1.0a9' / 'bin' / 'python'

SOURCES = (
    'lab-fibresight-demo-train-01.npy',
    'lab-fibresight-demo-train-02.npy',
    'lab-fibresight-demo-test.npy',
    'allen-vis-two-photon-01.npy',
    'allen-vis-two-photon-02.npy',
    'dandi-jgcamp8f-soma.npy',
    'dandi-jgcamp8s-soma.npy',
    'dandi-olfactory-bulb.npy',
    'labmate-tdtomato-soma.npy',
    'labmate-gcamp-soma.npy',
    )
SOURCE_ANATOMY = {
    'lab-fibresight-demo-train-01.npy': 'fibre',
    'lab-fibresight-demo-train-02.npy': 'fibre',
    'lab-fibresight-demo-test.npy': 'fibre',
    'allen-vis-two-photon-01.npy': 'somatic',
    'allen-vis-two-photon-02.npy': 'somatic',
    'dandi-jgcamp8f-soma.npy': 'somatic',
    'dandi-jgcamp8s-soma.npy': 'somatic',
    'dandi-olfactory-bulb.npy': 'mesoscale',
    'labmate-tdtomato-soma.npy': 'somatic',
    'labmate-gcamp-soma.npy': 'somatic',
    }
BASE_SOURCES = SOURCES[:3]
AUTO_SOURCES = (
    'lab-fibresight-demo-train-01.npy',
    'lab-fibresight-demo-train-02.npy',
    'lab-fibresight-demo-test.npy',
    )
RECIPES = ('ordinary_motion', 'large_motion', 'local_deformation', 'focal_change')
PHOTON_COUNTS = (30, 65, 150)
BLEACHING_LEVELS = (0, 0.2, 0.5)
METHODS = (
    'fibresight_rigid',
    'suite2p_rigid',
    'suite2p_piecewise',
    'caiman_rigid',
    'caiman_piecewise',
    'patchwarp_rigid',
    'patchwarp_affine',
    'pyflowreg_piecewise',
    )
TILE_METHODS = {
    'fibresight_tiles_64_w0': {'tile_size': 64, 'whitening': 0},
    'fibresight_tiles_64_w0_5': {'tile_size': 64, 'whitening': 0.5},
    'fibresight_tiles_80_w0': {'tile_size': 80, 'whitening': 0},
    'fibresight_tiles_128_w0': {'tile_size': 128, 'whitening': 0},
    'fibresight_tiles_128_to_64_w0': {
        'tile_size': 64, 'coarse_tile_size': 128, 'whitening': 0},
    'fibresight_tiles_80_to_64_w0': {
        'tile_size': 64, 'coarse_tile_size': 80, 'whitening': 0},
    }
FIELD_METHODS = {
    'fibresight_piecewise': {},
    'fibresight_field_80_w0': {
        'tile_size': 80,
        'whitening': 0,
        'field_penalties': (0.1, 1, 10, 100),
        'field_magnitude': 1,
        'refine_penalty': 10,
        'field_residual_limit': 0.28,
        },
    }
WHITENING_METHODS = ('fibresight_whitening_0_5', 'fibresight_whitening_1')
PHOTON_METHODS = (*METHODS, *WHITENING_METHODS)
METHOD_RESOURCES = {
    'fibresight_rigid': ('threads', 4, 1),
    'suite2p_rigid': ('native threads', 1, 4),
    'suite2p_piecewise': ('native threads', 1, 4),
    'caiman_rigid': ('processes', 4, 1),
    'caiman_piecewise': ('processes', 4, 1),
    'patchwarp_rigid': ('processes', 4, 1),
    'patchwarp_affine': ('processes', 4, 1),
    'pyflowreg_piecewise': ('processes', 4, 1),
    'fibresight_whitening_0_5': ('threads', 4, 1),
    'fibresight_whitening_1': ('threads', 4, 1),
    'fibresight_field_80_w0': ('threads', 4, 1),
    'fibresight_piecewise': ('threads', 4, 1),
    }
RIGID_METHODS = (
    'fibresight_rigid', 'suite2p_rigid', 'caiman_rigid', 'patchwarp_rigid')
REFERENCE_METHODS = (
    *RIGID_METHODS, 'caiman_piecewise', 'pyflowreg_piecewise')
REFERENCE_FILES = {
    'fibresight_rigid': 'fibresight_reference_convergence.npz',
    'suite2p_rigid': 'suite2p_reference_convergence.npz',
    'caiman_rigid': 'caiman_rigid_reference_convergence.npz',
    'caiman_piecewise': 'caiman_piecewise_reference_convergence.npz',
    'patchwarp_rigid': 'patchwarp_reference_convergence.npz',
    'pyflowreg_piecewise': 'pyflowreg_reference_convergence.npz',
    }
METHOD_NAMES = {
    'fibresight_rigid': 'FibreSight rigid',
    'suite2p_rigid': 'Suite2p rigid',
    'suite2p_piecewise': 'Suite2p piecewise',
    'caiman_rigid': 'CaImAn rigid',
    'caiman_piecewise': 'CaImAn piecewise',
    'patchwarp_rigid': 'PatchWarp rigid',
    'patchwarp_affine': 'PatchWarp affine (8 to 4 on failure)',
    'pyflowreg_piecewise': 'PyFlowReg piecewise',
    'fibresight_whitening_0_5': 'FibreSight w=0.5',
    'fibresight_whitening_1': 'FibreSight w=1',
    'fibresight_piecewise': 'FibreSight piecewise',
    }
METHOD_COLOURS = {
    # 18 August 2026: the GUI charcoal stays fixed; piecewise uses a lighter shade
    'fibresight_rigid': '#1A1A1C',
    'suite2p_rigid': '#D55E00',
    'suite2p_piecewise': '#E69F00',
    'caiman_rigid': '#009E73',
    'caiman_piecewise': '#CC79A7',
    'patchwarp_rigid': '#882255',
    'patchwarp_affine': '#44AA99',
    'pyflowreg_piecewise': '#0072B2',
    'fibresight_whitening_0_5': '#6A6864',
    'fibresight_whitening_1': '#AAA7A0',
    'fibresight_piecewise': '#66636D',
    }


def _method_tick_label(method):
    if method == 'patchwarp_affine':
        return 'PatchWarp\naffine*'
    return METHOD_NAMES[method].replace(' ', '\n', 1)


def _add_patchwarp_note(figure):
    figure.get_layout_engine().set(rect=(0, 0.08, 1, 1))
    figure.text(
        0.5, 0.012, '* 8x8 grid with 4x4 retry on failure',
        ha='center', fontsize=8,
        )


def benchmark_cases(root=SUITE_ROOT):
    cases = []
    for source_i, source in enumerate(SOURCES):
        for recipe_i, recipe in enumerate(RECIPES):
            source_name = source
            sequence = np.random.SeedSequence([42, source_i, recipe_i])
            cases.append({
                'case': f'{source_name}/{recipe}',
                'source': source,
                'recipe': recipe,
                'seed': int(sequence.generate_state(1, dtype=np.uint32)[0]),
                'root': Path(root) / source_name / recipe,
                'case_order': len(cases) + 1,
                })
    return cases


def focal_cases(root=FOCAL_ROOT):
    cases = []
    for source_i, source in enumerate(BASE_SOURCES):
        source_name = source
        sequence = np.random.SeedSequence([42, source_i, 4])
        cases.append({
            'case': f'{source_name}/optical_defocus',
            'source': source,
            'recipe': 'optical_defocus',
            'seed': int(sequence.generate_state(1, dtype=np.uint32)[0]),
            'root': Path(root) / source_name,
            'case_order': len(cases) + 1,
            })
    return cases


def auto_validation_cases(root=AUTO_ROOT):
    cases = []
    for source_i, source in enumerate(AUTO_SOURCES):
        for recipe_i, recipe in enumerate(RECIPES):
            # 18 August 2026: seed branch 6 gives validation independent movement and noise
            sequence = np.random.SeedSequence([42, 6, source_i, recipe_i])
            cases.append({
                'case': f'{source}/{recipe}',
                'source': source,
                'recipe': recipe,
                'seed': int(sequence.generate_state(1, dtype=np.uint32)[0]),
                'root': Path(root) / source / recipe,
                'case_order': len(cases) + 1,
                })
    return cases


def photon_cases(root=INTENSITY_ROOT):
    cases = []
    for source_i, source in enumerate(BASE_SOURCES):
        source_name = source
        for recipe_i, recipe in enumerate(RECIPES):
            seed = int(np.random.SeedSequence(
                [42, source_i, recipe_i]).generate_state(1, dtype=np.uint32)[0])
            for photons in PHOTON_COUNTS:
                cases.append({
                    'case_order': len(cases) + 1,
                    'case': f'{source_name}/{recipe}/p{photons:03d}_b20',
                    'source': source,
                    'recipe': recipe,
                    'seed': seed,
                    'control_photons': photons,
                    'bleaching': 0.2,
                    'root': Path(root) / source_name / recipe / f'p{photons:03d}_b20',
                    })
    return cases


def bleaching_cases(root=INTENSITY_ROOT):
    cases = []
    for source_i, source in enumerate(BASE_SOURCES):
        source_name = source
        seed = int(np.random.SeedSequence(
            [42, source_i, 0]).generate_state(1, dtype=np.uint32)[0])
        for bleaching in BLEACHING_LEVELS:
            bleach_percent = round(100 * bleaching)
            cases.append({
                'case_order': len(cases) + 1,
                'case': f'{source_name}/ordinary_motion/p065_b{bleach_percent:02d}',
                'source': source,
                'recipe': 'ordinary_motion',
                'seed': seed,
                'control_photons': 65,
                'bleaching': bleaching,
                'root': (
                    Path(root) / source_name / 'ordinary_motion'
                    / f'p065_b{bleach_percent:02d}'
                    ),
                })
    return cases


def intensity_movies(root=INTENSITY_ROOT):
    movies = {}
    for case in (*photon_cases(root), *bleaching_cases(root)):
        movies[case['root']] = case
    return list(movies.values())


def check_suite(root=SUITE_ROOT):
    import tifffile

    for case in benchmark_cases(root):
        case_name = case['case']
        truth_path = case['root'] / 'truth.npz'
        movie_path = case['root'] / 'control.tif'
        if not truth_path.exists() or not movie_path.exists():
            raise FileNotFoundError(f'incomplete benchmark case: {case_name}')
        with np.load(truth_path) as truth:
            matches = (
                str(truth['source']) == 'benchmarking/sources/' + case['source']
                and str(truth['recipe']) == case['recipe']
                and int(truth['seed']) == case['seed']
                and int(truth['n_frames']) == tifffile.memmap(movie_path).shape[0]
                )
        if not matches:
            raise ValueError(f'benchmark truth does not match: {case_name}')


def make_suite(root=SUITE_ROOT, n_frames=2000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = benchmark_cases(root)
    # 15 August 2026: write truth last so an interrupted case remains incomplete
    for case in cases:
        (case['root'] / 'truth.npz').unlink(missing_ok=True)
        for method in METHODS:
            (case['root'] / f'{method}.npz').unlink(missing_ok=True)
        for name in (
                'metrics.csv', 'reference_metrics.csv', 'reference_frc.csv',
                'method_reference_convergence.csv',
                *REFERENCE_FILES.values(),
                ):
            (case['root'] / name).unlink(missing_ok=True)
    for name in (
            'truth_summary.csv', 'motion_truth.png', 'resources.csv',
            'suite_metrics.csv', 'suite_reference_metrics.csv',
            'suite_reference_frc.csv', 'suite_reference_convergence.csv',
            'suite_summary.csv', 'suite_comparisons.csv',
            ):
        (root / name).unlink(missing_ok=True)

    with (root / 'cases.csv').open('w', newline='') as file:
        writer = DictWriter(
            file, fieldnames=['case_order', 'case', 'source', 'recipe', 'seed', 'frames'])
        writer.writeheader()
        for case in cases:
            case_name = case['case']
            print(f'making {case_name}')
            for name in ('fibresight_rigid.npz', 'focal_quality.npz'):
                (case['root'] / name).unlink(missing_ok=True)
            benchmark.make_benchmark(
                root=case['root'],
                n_frames=n_frames,
                seed=case['seed'],
                source=case['source'],
                recipe=case['recipe'],
                save_signal=False,
                )
            writer.writerow({
                **{name: case[name] for name in writer.fieldnames if name != 'frames'},
                'frames': n_frames,
                })
            file.flush()
    summarise_truth(root)
    plot_truth(root)


def make_focal_suite(root=FOCAL_ROOT, n_frames=2000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = focal_cases(root)
    for name in ('cases.csv', 'focal_metrics.csv', 'focal_threshold_calibration.csv'):
        (root / name).unlink(missing_ok=True)

    with (root / 'cases.csv').open('w', newline='') as file:
        writer = DictWriter(
            file, fieldnames=['case_order', 'case', 'source', 'recipe', 'seed', 'frames'])
        writer.writeheader()
        for case in cases:
            case_name = case['case']
            print(f'making {case_name}')
            benchmark.make_benchmark(
                root=case['root'],
                n_frames=n_frames,
                seed=case['seed'],
                source=case['source'],
                recipe=case['recipe'],
                save_signal=False,
                )
            writer.writerow({
                **{name: case[name] for name in writer.fieldnames if name != 'frames'},
                'frames': n_frames,
                })
            file.flush()


def run_focal_suite(root=FOCAL_ROOT):
    for case in focal_cases(root):
        case_name = case['case']
        print(f'running {case_name}')
        benchmark.run_fibresight(case['root'])


#%% truth summaries
def _episode_lengths(mask):
    edges = np.diff(np.pad(np.asarray(mask, dtype=np.int8), 1))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _episode_bounds(mask, merge_gap_frames=0):
    edges = np.diff(np.pad(np.asarray(mask, dtype=np.int8), 1))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    episodes = []
    for start, stop in zip(starts, stops):
        if episodes and start - episodes[-1][1] <= merge_gap_frames:
            episodes[-1][1] = int(stop)
        else:
            episodes.append([int(start), int(stop)])
    return episodes


def _local_movement(truth):
    height, width = truth['basis_y'].shape[1:]
    sample_y = np.linspace(16, height - 17, 9).round().astype(int)
    sample_x = np.linspace(16, width - 17, 9).round().astype(int)
    y, x = np.meshgrid(sample_y, sample_x, indexing='ij')
    local_y = truth['coefficient'] @ truth['basis_y'][:, y.ravel(), x.ravel()]
    local_x = truth['coefficient'] @ truth['basis_x'][:, y.ravel(), x.ravel()]
    return np.max(np.hypot(local_y, local_x), axis=1)


def summarise_truth(root=SUITE_ROOT):
    rows = []
    for case in benchmark_cases(root):
        with np.load(case['root'] / 'truth.npz') as saved:
            truth = {name: saved[name] for name in saved.files}
        rigid = np.hypot(truth['shift_y'], truth['shift_x'])
        step = np.hypot(np.diff(truth['shift_y']), np.diff(truth['shift_x']))
        local = _local_movement(truth)
        focal_lengths = _episode_lengths(truth['focal'])
        rows.append({
            'case_order': case['case_order'],
            'case': case['case'],
            'source': case['source'],
            'recipe': case['recipe'],
            'seed': case['seed'],
            'frames': len(rigid),
            'rigid_p95_px': np.percentile(rigid, 95),
            'rigid_max_px': rigid.max(),
            'largest_frame_step_px': step.max(),
            'local_p95_px': np.percentile(local, 95),
            'local_max_px': local.max(),
            'jacobian_min': truth['jacobian_min'].min(),
            'jacobian_max': truth['jacobian_max'].max(),
            'focal_episodes': len(focal_lengths),
            'focal_min_frames': focal_lengths.min() if len(focal_lengths) else 0,
            'focal_max_frames': focal_lengths.max() if len(focal_lengths) else 0,
            'unestimable_frames': np.sum(~truth['estimable']),
            })

    root = Path(root)
    with (root / 'truth_summary.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_truth(root=SUITE_ROOT):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()

    root = Path(root)
    figure, axes = plt.subplots(
        len(RECIPES), len(SOURCES), figsize=(14, 10), constrained_layout=True)
    for case in benchmark_cases(root):
        source_i = SOURCES.index(case['source'])
        recipe_i = RECIPES.index(case['recipe'])
        axis = axes[recipe_i, source_i]
        with np.load(case['root'] / 'truth.npz') as saved:
            truth = {name: saved[name] for name in saved.files}
        time_s = np.arange(len(truth['shift_y'])) / 30
        local = _local_movement(truth)
        axis.plot(time_s, truth['shift_y'], linewidth=0.8, label='rigid y')
        axis.plot(time_s, truth['shift_x'], linewidth=0.8, label='rigid x')
        axis.plot(time_s, local, linewidth=0.8, label='local magnitude')
        axis.axhline(0, color='0.75', linewidth=0.5)
        for mask, colour, label in (
                (truth['focal'], 'C3', 'focal episode'),
                (truth['ambiguous'], 'C5', 'ambiguous pattern'),
                ):
            starts = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), 1)) == 1)
            for start, length in zip(starts, _episode_lengths(mask)):
                axis.axvspan(
                    start / 30,
                    (start + length) / 30,
                    color=colour,
                    alpha=0.16,
                    label=label if source_i == 0 else None,
                    )
        if case['recipe'] == 'focal_change':
            z_axis = axis.twinx()
            z_axis.plot(time_s, truth['z'], color='C4', linewidth=0.7, label='z plane')
            z_axis.set_ylim(-2.2, 2.2)
            if source_i == len(SOURCES) - 1:
                z_axis.set_ylabel('synthetic z plane')
            if source_i == 0:
                z_axis.legend(frameon=False, fontsize=8, loc='lower right')
        if recipe_i == 0:
            axis.set_title(case['source'])
        if source_i == 0:
            recipe = case['recipe']
            axis.set_ylabel(f'{recipe}\ndisplacement (px)')
        if recipe_i == len(RECIPES) - 1:
            axis.set_xlabel('time (s)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.savefig(root / 'motion_truth.png', dpi=180)
    plt.close(figure)


def _review_centre(truth, recipe):
    if recipe == 'local_deformation':
        return int(np.argmax(_local_movement(truth)))
    if recipe == 'focal_change':
        focal_frames = np.flatnonzero(truth['focal'])
        return int(focal_frames[np.argmax(np.abs(truth['z'][focal_frames]))])
    step = np.hypot(np.gradient(truth['shift_y']), np.gradient(truth['shift_x']))
    return int(np.argmax(step))


def _undo_truth_motion(frame, field_y, field_x):
    from scipy.ndimage import map_coordinates

    y, x = np.mgrid[:frame.shape[0], :frame.shape[1]].astype(np.float32)
    observed_y = y.copy()
    observed_x = x.copy()
    for _ in range(5):
        movement_y = map_coordinates(
            field_y, (observed_y, observed_x), order=1, mode='nearest', prefilter=False)
        movement_x = map_coordinates(
            field_x, (observed_y, observed_x), order=1, mode='nearest', prefilter=False)
        observed_y = y + movement_y
        observed_x = x + movement_x
    return map_coordinates(
        frame,
        (observed_y, observed_x),
        order=1,
        mode='constant',
        cval=float(np.percentile(frame, 2)),
        prefilter=False,
        )


def make_review_movie(
        root=SUITE_ROOT,
        path=EXAMPLE_ROOT / 'registration_benchmark_suite.mp4',
        ):
    import av
    import cv2
    import tifffile

    root = Path(root)
    check_suite(root)
    video = av.open(path, 'w')
    stream = video.add_stream('libx264', rate=30, options={'crf': '23', 'preset': 'medium'})
    shown_size = 192
    margin = 12
    gutter = 18
    title_height = 42
    stream.width = 2 * margin + 3 * shown_size + 2 * gutter
    stream.height = 2 * margin + title_height + shown_size
    stream.pix_fmt = 'yuv420p'

    for case in benchmark_cases(root):
        movie = tifffile.memmap(case['root'] / 'control.tif')
        with np.load(case['root'] / 'truth.npz') as saved:
            truth = {name: saved[name] for name in saved.files}
        centre = _review_centre(truth, case['recipe'])
        start = max(0, min(centre - 30, len(movie) - 60))
        sample = np.asarray(movie[np.linspace(0, len(movie) - 1, 100, dtype=int)])
        low, high = np.percentile(sample, [0.1, 99.9])
        latent = np.clip(
            255 * benchmark._normalise(truth['latent_reference']), 0, 255).astype(np.uint8)

        for frame_i in range(start, start + 60):
            frame = np.asarray(movie[frame_i], dtype=np.float32)
            field_y = truth['shift_y'][frame_i] + np.sum(
                truth['coefficient'][frame_i, :, None, None] * truth['basis_y'], axis=0)
            field_x = truth['shift_x'][frame_i] + np.sum(
                truth['coefficient'][frame_i, :, None, None] * truth['basis_x'], axis=0)
            corrected = _undo_truth_motion(frame, field_y, field_x)
            raw = np.clip(255 * (frame - low) / (high - low), 0, 255).astype(np.uint8)
            corrected = np.clip(
                255 * (corrected - low) / (high - low), 0, 255).astype(np.uint8)

            canvas = np.full((stream.height, stream.width, 3), 24, dtype=np.uint8)
            source = case['source']
            recipe = case['recipe']
            title = f'{source}  {recipe}'
            cv2.putText(
                canvas, title, (margin, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1,
                )
            for panel_i, (panel, label) in enumerate((
                    (latent, 'central plane'),
                    (raw, 'raw'),
                    (corrected, 'truth field corrected'),
                    )):
                panel = cv2.resize(
                    panel, (shown_size, shown_size), interpolation=cv2.INTER_AREA)
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    panel, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3,
                    )
                cv2.putText(
                    panel, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1,
                    )
                x = margin + panel_i * (shown_size + gutter)
                y = margin + title_height
                canvas[y:y + shown_size, x:x + shown_size] = panel
            video_frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
            for packet in stream.encode(video_frame):
                video.mux(packet)

    for packet in stream.encode():
        video.mux(packet)
    video.close()
    return path


def make_rigid_review_movie(
        root=SUITE_ROOT,
        path=EXAMPLE_ROOT / 'rigid_registration_benchmark.mp4',
        ):
    case = benchmark_cases(root)[0]
    return benchmark.make_rigid_comparison_movie(case['root'], path=path)


#%% method processes
def _run_method(method, root):
    root = Path(root)
    if method.startswith('fibresight'):
        whitening = {
            'fibresight_rigid': 0,
            'fibresight_whitening_0_5': 0.5,
            'fibresight_whitening_1': 1,
            }[method]
        benchmark.run_fibresight(root, whitening=whitening, name=method)
    elif method.startswith('suite2p'):
        benchmark.run_suite2p(root, piecewise=method.endswith('piecewise'))
    elif method.startswith('patchwarp'):
        benchmark.run_patchwarp(root, affine=method.endswith('affine'))
    elif method == 'pyflowreg_piecewise':
        benchmark.run_pyflowreg(root)
    else:
        benchmark.run_caiman(root, piecewise=method.endswith('piecewise'))


def _run_tile_method(method, root):
    benchmark.run_fibresight_tile_evidence(
        root,
        name=method,
        **TILE_METHODS[method],
        save_surfaces=method == 'fibresight_tiles_64_w0',
        )


def _run_field_method(method, root):
    if method == 'fibresight_piecewise':
        benchmark.run_fibresight_piecewise(root)
        return
    benchmark.run_fibresight_tile_evidence(
        root,
        name=method,
        **FIELD_METHODS[method],
        )


def _run_reference(method, root):
    if method == 'fibresight_rigid':
        benchmark.run_fibresight_references(root)
    elif method == 'suite2p_rigid':
        benchmark.run_suite2p_references(root)
    elif method == 'patchwarp_rigid':
        benchmark.run_patchwarp_references(root)
    elif method == 'pyflowreg_piecewise':
        benchmark.run_pyflowreg_references(root)
    elif method == 'caiman_piecewise':
        benchmark.run_caiman_references(root, piecewise=True)
    else:
        benchmark.run_caiman_references(root)


def _method_command(method, root):
    if method.startswith('fibresight'):
        python = FIBRESIGHT_PYTHON
    elif method.startswith('suite2p'):
        python = SUITE2P_PYTHON
    elif method.startswith('pyflowreg'):
        python = PYFLOWREG_PYTHON
    elif method.startswith('patchwarp'):
        python = FIBRESIGHT_PYTHON
    else:
        python = CAIMAN_PYTHON
    step = 'field' if method in FIELD_METHODS else 'method'
    return [
        str(python), '-m', 'benchmarking.registration_benchmark_suite',
        step, method, str(root),
        ]


def _reference_command(method, root):
    return [
        *_method_command(method, root)[:3],
        'reference', method, str(root),
        ]


def _method_environment(method, root):
    environment = os.environ.copy()
    _, _, native_threads = METHOD_RESOURCES[method]
    for name in (
            'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS',
            ):
        environment[name] = str(native_threads)
    environment['MPLCONFIGDIR'] = str(Path(root) / '_cache' / 'matplotlib')
    environment['XDG_CACHE_HOME'] = str(Path(root) / '_cache')
    environment['PYTHONHASHSEED'] = '42'
    return environment


def _method_order(case_i):
    # 15 August 2026: rotate method order so each method occupies each cache position
    offsets = tuple(range(len(METHODS)))
    row = case_i % len(METHODS)
    order = [METHODS[(row + offset) % len(METHODS)] for offset in offsets]
    if case_i // len(METHODS) % 2:
        order.reverse()
    return order


def _intensity_method_order(methods, case_i):
    row = case_i % len(methods)
    order = [methods[(row + method_i) % len(methods)] for method_i in range(len(methods))]
    # 15 August 2026: alternate blocks reverse the method order to balance cache drift
    if case_i // len(methods) % 2:
        order.reverse()
    return order


def _keep_finished_cases(path, finished):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline='') as file:
        reader = DictReader(file)
        fields = reader.fieldnames
        rows = [row for row in reader if row['case'] in finished]
    if fields is None:
        return
    with path.open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def warm_up(root, methods):
    imports = {
        'fibresight_rigid': 'import cv2, scipy; import fibre_sight.preprocessing',
        'suite2p_rigid': 'import cv2, matplotlib, numba, suite2p, torch',
        'caiman_rigid': 'import caiman, cv2, matplotlib',
        'pyflowreg_piecewise': 'import pyflowreg',
    }
    # 15 August 2026: import and font caches are warmed outside the scored processes
    for method, code in imports.items():
        if not any(name.startswith(method.split('_')[0]) for name in methods):
            continue
        command = _method_command(method, root)
        subprocess.run(
            [command[0], '-c', code],
            cwd=PROJECT_ROOT,
            env=_method_environment(method, root),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            )

    if not any(method.startswith('patchwarp') for method in methods):
        return

    # 15 August 2026: prime MATLAB's path cache before the first timed case
    matlab_code = (
        f"addpath(genpath('{benchmark.PATCHWARP_ROOT / 'PatchWarp'}')); "
        "which patchwarp_rigid; which patchwarp_affine")
    subprocess.run(
        [str(benchmark.MATLAB_PATH), '-singleCompThread', '-batch', matlab_code],
        cwd=PROJECT_ROOT,
        env=_method_environment('patchwarp_rigid', root),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        )


#%% resource measurements
def _resource_use(process):
    import psutil

    try:
        family = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0, 0
    rss = []
    threads = 0
    for member in family:
        try:
            rss.append(member.memory_info().rss)
            threads += member.num_threads()
        # 15 August 2026: the sampler cannot read several protected MATLAB helpers on macOS
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # 15 August 2026: process-tree RSS counts shared pages twice
    return max(rss, default=0), sum(rss), threads


def measure_method(method, case, suite_root):
    import psutil

    log_root = case['root'] / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(suite_root) / '_cache'
    (cache_root / 'matplotlib').mkdir(parents=True, exist_ok=True)

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_time = time.perf_counter()
    peak_process_rss = 0
    peak_tree_rss = 0
    peak_tree_threads = 0
    with (log_root / f'{method}.txt').open('w') as log:
        child = subprocess.Popen(
            _method_command(method, case['root']),
            cwd=PROJECT_ROOT,
            env=_method_environment(method, suite_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            )
        process = psutil.Process(child.pid)
        while child.poll() is None:
            process_rss, tree_rss, tree_threads = _resource_use(process)
            peak_process_rss = max(peak_process_rss, process_rss)
            peak_tree_rss = max(peak_tree_rss, tree_rss)
            peak_tree_threads = max(peak_tree_threads, tree_threads)
            time.sleep(0.05)
        exit_code = child.wait()

    wall_seconds = time.perf_counter() - start_time
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    user_seconds = after.ru_utime - before.ru_utime
    system_seconds = after.ru_stime - before.ru_stime
    worker_type, workers, native_threads = METHOD_RESOURCES[method]
    version = ''
    patchwarp_grid = ''
    patchwarp_grid_attempts = ''
    if exit_code == 0:
        with np.load(case['root'] / f'{method}.npz') as saved:
            version = str(saved['version'])
            if 'warp_blocksize' in saved:
                patchwarp_grid = str(int(saved['warp_blocksize']))
                patchwarp_grid_attempts = ','.join(
                    str(int(value)) for value in saved['warp_blocksize_attempts'])
    return {
        'wall_seconds': wall_seconds,
        'user_cpu_seconds': user_seconds,
        'system_cpu_seconds': system_seconds,
        'average_cores': (user_seconds + system_seconds) / wall_seconds,
        'peak_process_rss_bytes': peak_process_rss,
        'peak_tree_rss_bytes': peak_tree_rss,
        'peak_tree_threads': peak_tree_threads,
        'exit_code': exit_code,
        'version': version,
        'patchwarp_grid': patchwarp_grid,
        'patchwarp_grid_attempts': patchwarp_grid_attempts,
        'allocated_cores': workers * native_threads,
        'worker_type': worker_type,
        'workers': workers,
        'native_threads_per_worker': native_threads,
        }


def run_suite(root=SUITE_ROOT):
    root = Path(root)
    cases = benchmark_cases(root)
    check_suite(root)
    warm_up(root, METHODS)

    fields = [
        'run_order', 'case_order', 'method_order', 'case', 'source', 'recipe',
        'seed', 'method', 'version', 'patchwarp_grid', 'patchwarp_grid_attempts',
        'allocated_cores', 'worker_type', 'workers',
        'native_threads_per_worker', 'wall_seconds',
        'user_cpu_seconds', 'system_cpu_seconds', 'average_cores',
        'peak_process_rss_bytes', 'peak_tree_rss_bytes', 'peak_tree_threads', 'exit_code',
        ]
    resource_path = root / 'resources.csv'
    if resource_path.exists():
        with resource_path.open(newline='') as file:
            completed = {
                (row['case'], row['method']) for row in DictReader(file)}
    else:
        completed = set()
    run_order = 0
    with resource_path.open('a' if completed else 'w', newline='') as file:
        writer = DictWriter(file, fieldnames=fields)
        if not completed:
            writer.writeheader()
        for case_i, case in enumerate(cases):
            for method_i, method in enumerate(_method_order(case_i), start=1):
                run_order += 1
                if (case['case'], method) in completed:
                    continue
                case_name = case['case']
                print(f'{run_order:02d}/{len(cases) * len(METHODS)}  {case_name}  {method}')
                resources = measure_method(method, case, root)
                writer.writerow({
                    'run_order': run_order,
                    'case_order': case['case_order'],
                    'method_order': method_i,
                    'case': case['case'],
                    'source': case['source'],
                    'recipe': case['recipe'],
                    'seed': case['seed'],
                    'method': method,
                    **resources,
                })
                file.flush()


def run_piecewise_suite(root=SUITE_ROOT):
    root = Path(root)
    cases = benchmark_cases(root)
    method = 'fibresight_field_80_w0'
    resource_path = root / 'piecewise_resources.csv'
    completed = set()
    if resource_path.exists():
        completed = {
            row['case'] for row in _read_rows(resource_path)
            if int(row['exit_code']) == 0}
    fields = [
        'case_order', 'case', 'source', 'recipe', 'seed', 'method',
        'version', 'allocated_cores', 'worker_type', 'workers',
        'native_threads_per_worker', 'wall_seconds', 'user_cpu_seconds',
        'system_cpu_seconds', 'average_cores', 'peak_process_rss_bytes',
        'peak_tree_rss_bytes', 'peak_tree_threads', 'exit_code',
        'patchwarp_grid', 'patchwarp_grid_attempts',
        ]
    with resource_path.open('a' if completed else 'w', newline='') as file:
        writer = DictWriter(file, fieldnames=fields)
        if not completed:
            writer.writeheader()
        for case_i, case in enumerate(cases, start=1):
            if case['case'] in completed:
                continue
            case_name = case['case']
            print(f'{case_i:02d}/{len(cases)}  {case_name}  FibreSight piecewise')
            resources = measure_method(method, case, root)
            writer.writerow({
                **{name: case[name] for name in (
                    'case_order', 'case', 'source', 'recipe', 'seed')},
                'method': method,
                **resources,
                })
            file.flush()
            if resources['exit_code']:
                raise RuntimeError(f'FibreSight piecewise failed for {case_name}')
            benchmark.measure_tile_fields(case['root'], [method])


def measure_piecewise_suite(root=SUITE_ROOT):
    root = Path(root)
    methods = {
        'fibresight_piecewise', 'suite2p_piecewise', 'caiman_piecewise',
        'patchwarp_affine', 'pyflowreg_piecewise',
        }
    rows = []
    for case in benchmark_cases(root):
        benchmark.measure_tile_fields(case['root'], ['fibresight_field_80_w0'])
        for path in (
                case['root'] / 'piecewise_metrics.csv',
                case['root'] / 'metrics.csv',
                ):
            rows.extend({
                'case_order': case['case_order'],
                'case': case['case'],
                'source': case['source'],
                'recipe': case['recipe'],
                **row,
                } for row in _read_rows(path) if row['method'] in methods)
    with (root / 'piecewise_suite_metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def measure_auto_suite(root=SUITE_ROOT, cases=None):
    import tifffile
    from fibre_sight.preprocessing import (
        compare_registration_models,
        estimate_channel_offset,
        measure_quality,
        register_pair,
        )

    root = Path(root)
    rows = []
    cases = benchmark_cases(root) if cases is None else cases
    for case_i, case in enumerate(cases, start=1):
        case_name = case['case']
        print(f'{case_i:02d}/{len(cases)}  {case_name}')
        control_movie = tifffile.memmap(case['root'] / 'control.tif')
        signal_path = case['root'] / 'signal.tif'
        signal_movie = (
            tifffile.memmap(signal_path) if signal_path.exists() else control_movie)
        with np.load(case['root'] / 'fibresight_rigid.npz') as saved:
            rigid = {name: saved[name] for name in saved.files}
        with np.load(case['root'] / 'fibresight_field_80_w0.npz') as saved:
            piecewise = {name: saved[name] for name in saved.files}
        sample = np.linspace(0, len(control_movie) // 2 - 1, 100, dtype=int)
        signal_frames = np.asarray(signal_movie[sample])
        control_frames = np.asarray(control_movie[sample])
        estimates = [{
            'shift_y': -float(rigid['shift_y'][frame_i]),
            'shift_x': -float(rigid['shift_x'][frame_i]),
            'peak_ratio': float(rigid['peak_ratio'][frame_i]),
            'tile_disagreement': float(rigid['tile_disagreement'][frame_i]),
            'out_of_range': bool(rigid['out_of_range'][frame_i]),
            'search_boundary': (
                bool(rigid['search_boundary'][frame_i])
                if 'search_boundary' in rigid else False),
            } for frame_i in sample]
        aligned = [
            register_pair(signal, control, estimate['shift_y'], estimate['shift_x'])
            for signal, control, estimate in zip(signal_frames, control_frames, estimates)
            ]
        channel_offset = estimate_channel_offset(
            [pair['signal'] for pair in aligned],
            [pair['control'] for pair in aligned],
            )
        signal_to_control_offset = (
            channel_offset['shift_y'], channel_offset['shift_x'])

        penalty_i = int(np.flatnonzero(
            piecewise['field_penalties'] == piecewise['refined_penalty'])[0])
        tile_results = []
        fields = []
        for frame_i in sample:
            precision = np.empty((len(piecewise['tile_y']), 2, 2), dtype=np.float32)
            precision[:, 0, 0] = piecewise['field_precision_yy'][frame_i]
            precision[:, 0, 1] = piecewise['field_precision_yx'][frame_i]
            precision[:, 1, 0] = piecewise['field_precision_yx'][frame_i]
            precision[:, 1, 1] = piecewise['field_precision_xx'][frame_i]
            tile_results.append({
                'tile_y': piecewise['tile_y'],
                'tile_x': piecewise['tile_x'],
                'residual_y': -piecewise['local_y'][frame_i],
                'residual_x': -piecewise['local_x'][frame_i],
                'accepted': piecewise['field_accepted'][frame_i],
                'precision': precision,
                })
            fields.append({
                'global_shift_y': -piecewise['field_global_y'][penalty_i, frame_i],
                'global_shift_x': -piecewise['field_global_x'][penalty_i, frame_i],
                'coefficient_y': -piecewise['field_coefficient_y'][penalty_i, frame_i],
                'coefficient_x': -piecewise['field_coefficient_x'][penalty_i, frame_i],
                'control_y': piecewise['field_control_y'],
                'control_x': piecewise['field_control_x'],
                'tile_y': piecewise['tile_y'],
                'tile_x': piecewise['tile_x'],
                'predicted_y': -piecewise['field_local_y'][penalty_i, frame_i],
                'predicted_x': -piecewise['field_local_x'][penalty_i, frame_i],
                'accepted': piecewise['field_accepted'][frame_i],
                'spatial_penalty': piecewise['field_penalties'][penalty_i],
                'magnitude_penalty': piecewise['field_magnitude_penalty'],
                'tile_size': piecewise['tile_size'],
                })

        focus_evidence_available = any(
            estimate['peak_ratio'] >= 1.1
            and estimate['tile_disagreement'] <= 1
            and not estimate['out_of_range']
            for estimate in estimates)
        if focus_evidence_available:
            quality = measure_quality(
                rigid['reference'],
                control_frames,
                estimates,
                30,
                timestamps=np.arange(len(sample), dtype=float) / 30,
                )
            focal_loss = quality['recommended_state'] == 'focal_loss'
        else:
            # 18 August 2026: auto stays rigid when these frames cannot calibrate focus evidence
            focal_loss = np.ones(len(sample), dtype=bool)
        decision = compare_registration_models(
            rigid['reference'],
            signal_frames,
            control_frames,
            estimates,
            tile_results,
            fields,
            signal_to_control_offset=signal_to_control_offset,
            focal_loss=focal_loss,
            )

        with (case['root'] / 'metrics.csv').open() as file:
            metric_rows = list(DictReader(file))
        with (case['root'] / 'piecewise_metrics.csv').open() as file:
            metric_rows.extend(DictReader(file))
        selected = decision['selected_model']
        selected_method = (
            'fibresight_piecewise' if selected == 'piecewise_rigid'
            else 'fibresight_rigid')
        selected_p95 = next(float(row['p95_error_px']) for row in metric_rows
            if row['method'] == selected_method
            and row['group'] == 'heldout')
        rigid_p95 = next(float(row['p95_error_px']) for row in metric_rows
            if row['method'] == 'fibresight_rigid' and row['group'] == 'heldout')
        piecewise_p95 = next(float(row['p95_error_px']) for row in metric_rows
            if row['method'] == 'fibresight_piecewise' and row['group'] == 'heldout')
        rows.append({
            **{name: case[name] for name in (
                'case_order', 'case', 'source', 'recipe')},
            'selected_model': selected,
            'heldout_p95_error_px': selected_p95,
            'rigid_heldout_p95_error_px': rigid_p95,
            'piecewise_heldout_p95_error_px': piecewise_p95,
            'best_heldout_model': (
                'piecewise_rigid' if piecewise_p95 < rigid_p95 else 'rigid'),
            'selection_regret_px': selected_p95 - min(rigid_p95, piecewise_p95),
            'signal_source': 'signal_movie' if signal_path.exists() else 'control_copy',
            'focus_evidence_available': focus_evidence_available,
            **{f'rigid_{name}': value for name, value in decision['rigid'].items()},
            **{f'piecewise_{name}': value for name, value in decision['piecewise'].items()},
            **decision['comparison'],
            **{f'passed_{name}': value for name, value in decision['passed'].items()},
            })
    with (root / 'piecewise_auto.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run_auto_validation(root=AUTO_ROOT, n_frames=1000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = auto_validation_cases(root)
    with (root / 'cases.csv').open('w', newline='') as file:
        writer = DictWriter(
            file,
            fieldnames=['case_order', 'case', 'source', 'recipe', 'seed', 'frames'],
            )
        writer.writeheader()
        for case in cases:
            writer.writerow({
                **{name: case[name] for name in writer.fieldnames if name != 'frames'},
                'frames': n_frames,
                })

    for case_i, case in enumerate(cases, start=1):
        case_name = case['case']
        print(f'{case_i:02d}/{len(cases)}  {case_name}')
        benchmark.make_benchmark(
            root=case['root'],
            n_frames=n_frames,
            seed=case['seed'],
            source=case['source'],
            recipe=case['recipe'],
            save_signal=True,
            )
        benchmark.run_fibresight(case['root'])
        benchmark.run_fibresight_tile_evidence(
            case['root'],
            name='fibresight_field_80_w0',
            **FIELD_METHODS['fibresight_field_80_w0'],
            )
        benchmark.measure(
            case['root'], ['fibresight_rigid'], compare_references=False)
        benchmark.measure_tile_fields(
            case['root'], ['fibresight_field_80_w0'])

    measure_auto_suite(root, cases)
    for case in cases:
        # 18 August 2026: the seed and source reproduce these files; the table is the result
        for path in case['root'].glob('*.tif'):
            path.unlink()
        for path in case['root'].glob('*.npz'):
            path.unlink()


def summarise_piecewise_suite(
        root=SUITE_ROOT,
        intensity_root=INTENSITY_ROOT,
        figure_root=EXAMPLE_ROOT,
        ):
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    root = Path(root)
    intensity_root = Path(intensity_root)
    piecewise_metrics = _read_rows(root / 'piecewise_suite_metrics.csv')
    model_selection = _read_rows(root / 'piecewise_auto.csv')
    methods = (
        'fibresight_piecewise', 'suite2p_piecewise', 'caiman_piecewise',
        'patchwarp_affine', 'pyflowreg_piecewise')
    fibresight = _case_values(
        piecewise_metrics, 'fibresight_piecewise', 'p95_error_px', group='heldout')
    fibresight_sources = _source_values(
        piecewise_metrics, 'fibresight_piecewise', 'p95_error_px', group='heldout')
    comparisons = []
    for method in methods[1:]:
        competitor = _case_values(
            piecewise_metrics, method, 'p95_error_px', group='heldout')
        fibresight_cases, competitor_cases = _paired_values(
            fibresight, competitor)
        advantage = competitor_cases - fibresight_cases
        competitor_sources = _source_values(
            piecewise_metrics, method, 'p95_error_px', group='heldout')
        fibresight_source_values, competitor_source_values = _paired_values(
            fibresight_sources, competitor_sources)
        mean, low, high = _mean_ci(
            competitor_source_values - fibresight_source_values)
        wilcoxon_result = wilcoxon(fibresight_cases, competitor_cases)
        comparisons.append({
            'comparison': method,
            'n_cases': len(advantage),
            'n_sources': len(fibresight_source_values),
            'fibresight_wins': int(np.sum(advantage > 0)),
            'mean_advantage_px': float(mean),
            'ci_low_px': float(low),
            'ci_high_px': float(high),
            'wilcoxon_p_raw': float(wilcoxon_result.pvalue),
            'wilcoxon_p_bonferroni': min(
                1, 4 * float(wilcoxon_result.pvalue)),
            })
    with (root / 'piecewise_comparisons.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)

    benchmark._format_benchmark_plots()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    heldout_errors = {
        method: {
            row['case']: float(row['p95_error_px'])
            for row in piecewise_metrics
            if row['method'] == method and row['group'] == 'heldout'
            }
        for method in methods}
    common_cases = sorted(set.intersection(
        *(set(values) for values in heldout_errors.values())))
    for case in common_cases:
        axes[0].plot(
            range(len(methods)),
            [heldout_errors[method][case] for method in methods],
            color='0.82', linewidth=0.7)
    for method_i, method in enumerate(methods):
        values = list(heldout_errors[method].values())
        axes[0].scatter(
            np.full(len(values), method_i), values, s=18,
            color=METHOD_COLOURS[method], alpha=0.7, zorder=2)
        axes[0].scatter(
            method_i, np.mean(values), marker='D', s=42,
            color=METHOD_COLOURS[method], edgecolor='white', zorder=3)
    axes[0].set_xticks(
        range(len(methods)),
        [_method_tick_label(method) for method in methods])
    axes[0].tick_params(axis='x', pad=7)
    axes[0].set_ylabel('held-out p95 error (px)')
    axes[0].set_title('piecewise registration')
    # 18 August 2026: keep the useful range visible and mark clipped cases at 10 px
    plot_limit = 10
    off_scale_y = plot_limit - 0.15
    axes[0].set_ylim(0, plot_limit)
    short_names = {
        'suite2p_piecewise': 'Suite2p',
        'caiman_piecewise': 'CaImAn',
        'patchwarp_affine': 'PatchWarp*',
        'pyflowreg_piecewise': 'PyFlowReg',
        }
    axes[0].text(
        0.37, 0.96, 'Bonferroni-corrected Wilcoxon p',
        transform=axes[0].transAxes, fontsize=7.5, va='top')
    for comparison_i, row in enumerate(comparisons):
        method = row['comparison']
        p_value = float(row['wilcoxon_p_bonferroni'])
        axes[0].text(
            0.37, 0.90 - 0.055 * comparison_i,
            f'{short_names[method]}: p = {p_value:.2g}',
            color=METHOD_COLOURS[method], fontsize=7.5,
            transform=axes[0].transAxes, va='top')
    for method_i, method in enumerate(methods):
        off_scale_values = sorted(
            value for value in heldout_errors[method].values()
            if value > plot_limit)
        if not off_scale_values:
            continue
        off_scale_x = method_i + np.linspace(
            -0.06, 0.06, len(off_scale_values))
        axes[0].scatter(
            off_scale_x, np.full(len(off_scale_values), off_scale_y),
            marker='^', s=28,
            facecolor='white', edgecolor=METHOD_COLOURS[method], zorder=4,
            )
        axes[0].annotate(
            ', '.join(f'{value:.1f}' for value in off_scale_values) + ' px',
            xy=(method_i, off_scale_y), xytext=(0, -11),
            textcoords='offset points', ha='center', va='top', fontsize=7,
            color=METHOD_COLOURS[method],
            )

    selection_by_case = {row['case']: row for row in model_selection}
    cases = sorted(fibresight)
    selected_models = [selection_by_case[case]['selected_model'] for case in cases]
    heldout_error = [
        float(selection_by_case[case]['heldout_p95_error_px']) for case in cases]
    axes[1].scatter(
        range(len(cases)), heldout_error,
        c=[
            METHOD_COLOURS['fibresight_piecewise']
            if model == 'piecewise_rigid' else METHOD_COLOURS['fibresight_rigid']
            for model in selected_models],
        s=30)
    axes[1].scatter(
        [], [], color=METHOD_COLOURS['fibresight_piecewise'],
        label='piecewise-rigid selected')
    axes[1].scatter(
        [], [], color=METHOD_COLOURS['fibresight_rigid'], label='rigid selected')
    source_abbreviations = {
        'lab-fibresight-demo-train-01.npy': 'FS train 01',
        'lab-fibresight-demo-train-02.npy': 'FS train 02',
        'lab-fibresight-demo-test.npy': 'FS test',
        'allen-vis-two-photon-01.npy': 'Allen 01',
        'allen-vis-two-photon-02.npy': 'Allen 02',
        'dandi-jgcamp8f-soma.npy': 'jGCaMP8f',
        'dandi-jgcamp8s-soma.npy': 'jGCaMP8s',
        'dandi-olfactory-bulb.npy': 'olfactory',
        'labmate-tdtomato-soma.npy': 'tdTomato',
        'labmate-gcamp-soma.npy': 'GCaMP',
        }
    sources = list(dict.fromkeys(case.split('/')[0] for case in cases))
    source_centres = [
        np.mean([case_i for case_i, case in enumerate(cases)
                 if case.startswith(f'{source}/')])
        for source in sources]
    axes[1].set_xticks(
        source_centres,
        [source_abbreviations[source] for source in sources],
        rotation=35, ha='right', va='top', rotation_mode='anchor')
    for source_i in range(1, len(sources)):
        axes[1].axvline(4 * source_i - 0.5, color='0.90', linewidth=0.7)
    axes[1].set_xlabel('source image (four recipes per source)')
    axes[1].set_ylabel('held-out p95 error (px)')
    axes[1].set_title('automatic model selection')
    axes[1].legend(frameon=False, fontsize=8)
    _add_patchwarp_note(figure)
    figure.savefig(
        figure_root / 'registration_benchmark_piecewise.png',
        dpi=180, bbox_inches='tight')
    plt.close(figure)
    plot_registration_comparison(
        _read_rows(root / 'suite_metrics.csv'), piecewise_metrics,
        figure_root / 'registration_benchmark_comparison.png')

    photon_rows = (
        _read_rows(intensity_root / 'piecewise_photon_metrics.csv')
        + _read_rows(intensity_root / 'photon_metrics.csv'))
    figure, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for method in methods:
        medians = []
        for photons in PHOTON_COUNTS:
            values = [
                float(row['p95_error_px']) for row in photon_rows
                if row['method'] == method and row['group'] == 'heldout'
                and int(float(row['control_photons'])) == photons]
            medians.append(np.median(values))
        axis.plot(
            PHOTON_COUNTS, medians, marker='o', linewidth=1.6,
            color=METHOD_COLOURS[method], label=METHOD_NAMES[method])
    axis.set_xscale('log')
    axis.set_xticks(PHOTON_COUNTS, [str(value) for value in PHOTON_COUNTS])
    axis.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    axis.set_xlabel('control-channel photon count')
    axis.set_ylabel('median held-out p95 error (px)')
    axis.set_title('piecewise registration across photon counts')
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(figure_root / 'registration_benchmark_piecewise_noise.png', dpi=180)
    plt.close(figure)
    summarise_source_groups(root, figure_root)


def run_intensity_suite(root=INTENSITY_ROOT, n_frames=2000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    movies = intensity_movies(root)
    photon_lookup = {case['root']: case for case in photon_cases(root)}
    bleaching_lookup = {case['root']: case for case in bleaching_cases(root)}
    warm_up(root, PHOTON_METHODS)

    resource_fields = [
        'run_order', 'movie_order', 'method_order', 'case', 'source', 'recipe',
        'seed', 'control_photons', 'bleaching', 'method', 'version',
        'patchwarp_grid', 'patchwarp_grid_attempts',
        'allocated_cores', 'worker_type', 'workers', 'native_threads_per_worker',
        'wall_seconds', 'user_cpu_seconds', 'system_cpu_seconds', 'average_cores',
        'peak_process_rss_bytes', 'peak_tree_rss_bytes', 'peak_tree_threads', 'exit_code',
        ]
    total = sum(
        len(PHOTON_METHODS if case['root'] in photon_lookup else METHODS)
        for case in movies)
    # 15 August 2026: mark a movie only after its methods and metrics have finished
    finished = {
        case['case'] for case in movies
        if (case['root'] / '.complete').exists()
        }
    paths = (
        root / 'intensity_resources.csv',
        root / 'photon_metrics.csv',
        root / 'bleaching_metrics.csv',
        )
    for path in paths:
        _keep_finished_cases(path, finished)
    existing_resources = (
        _read_rows(paths[0]) if paths[0].exists() else [])
    run_order = len(existing_resources)
    photon_writer = None
    bleaching_writer = None
    with (
            paths[0].open('a', newline='') as resource_file,
            paths[1].open('a', newline='') as photon_file,
            paths[2].open('a', newline='') as bleaching_file,
            ):
        resource_writer = DictWriter(resource_file, fieldnames=resource_fields)
        if resource_file.tell() == 0:
            resource_writer.writeheader()
        for movie_i, case in enumerate(movies):
            case_root = case['root']
            case_name = case['case']
            if case_name in finished:
                continue
            print(f'making {movie_i + 1:02d}/{len(movies)}  {case_name}')
            for path in case_root.glob('*.npz'):
                path.unlink()
            benchmark.make_benchmark(
                root=case_root,
                n_frames=n_frames,
                seed=case['seed'],
                source=case['source'],
                recipe=case['recipe'],
                control_photons=case['control_photons'],
                bleaching=case['bleaching'],
                save_signal=False,
                )

            methods = PHOTON_METHODS if case_root in photon_lookup else METHODS
            for method_i, method in enumerate(
                    _intensity_method_order(methods, movie_i), start=1):
                run_order += 1
                print(f'{run_order:03d}/{total}  {case_name}  {method}')
                resources = measure_method(method, case, root)
                resource_writer.writerow({
                    'run_order': run_order,
                    'movie_order': movie_i + 1,
                    'method_order': method_i,
                    **{name: case[name] for name in (
                        'case', 'source', 'recipe', 'seed',
                        'control_photons', 'bleaching',
                        )},
                    'method': method,
                    **resources,
                })
                resource_file.flush()
                if resources['exit_code']:
                    raise RuntimeError(f'{method} failed; see its log')

            benchmark.measure(case_root, methods, compare_references=False)
            with (case_root / 'metrics.csv').open(newline='') as file:
                metric_rows = list(DictReader(file))
            metric_fields = [
                'case_order', 'case', 'source', 'recipe', 'seed',
                'control_photons', 'bleaching', *metric_rows[0],
                ]
            for lookup, names, file, writer in (
                    (photon_lookup, PHOTON_METHODS, photon_file, photon_writer),
                    (bleaching_lookup, METHODS, bleaching_file, bleaching_writer),
                    ):
                if case_root not in lookup:
                    continue
                sweep_case = lookup[case_root]
                if writer is None:
                    writer = DictWriter(file, fieldnames=metric_fields)
                    if file.tell() == 0:
                        writer.writeheader()
                    if lookup is photon_lookup:
                        photon_writer = writer
                    else:
                        bleaching_writer = writer
                writer.writerows({
                    **{name: sweep_case[name] for name in metric_fields[:7]},
                    **row,
                    } for row in metric_rows if row['method'] in names)
                file.flush()

            # 15 August 2026: keep disk use to one 251 MiB movie and its method outputs
            for path in (
                    case_root / 'control.tif', case_root / 'truth.npz',
                    case_root / 'metrics.csv',
                    *(case_root / f'{method}.npz' for method in methods),
                    ):
                path.unlink(missing_ok=True)
            (case_root / '.complete').touch()


def run_piecewise_intensity_suite(root=INTENSITY_ROOT, n_frames=2000):
    root = Path(root)
    resource_path = root / 'piecewise_photon_resources.csv'
    metric_path = root / 'piecewise_photon_metrics.csv'
    completed = {
        row['case'] for row in _read_rows(resource_path)
        if int(row['field_exit_code']) == 0
        } if resource_path.exists() else set()
    resource_fields = [
        'case_order', 'case', 'source', 'recipe', 'seed', 'control_photons',
        'bleaching', 'rigid_wall_seconds', 'field_wall_seconds',
        'total_wall_seconds', 'peak_process_rss_bytes', 'peak_tree_threads',
        'rigid_exit_code', 'field_exit_code',
        ]
    metric_fields = [
        'case_order', 'case', 'source', 'recipe', 'seed', 'control_photons',
        'bleaching', 'method', 'group', 'n', 'eligible_n', 'valid_fraction',
        'median_error_px', 'p95_error_px', 'over_1px_fraction',
        'offset_y_px', 'offset_x_px', 'seconds',
        ]
    with (
            resource_path.open('a' if completed else 'w', newline='') as resource_file,
            metric_path.open('a' if completed else 'w', newline='') as metric_file,
            ):
        resource_writer = DictWriter(resource_file, fieldnames=resource_fields)
        metric_writer = DictWriter(metric_file, fieldnames=metric_fields)
        if not completed:
            resource_writer.writeheader()
            metric_writer.writeheader()
        for case_i, case in enumerate(photon_cases(root), start=1):
            if case['case'] in completed:
                continue
            case_root = case['root']
            case_name = case['case']
            print(f'{case_i:02d}/36  {case_name}  FibreSight piecewise')
            benchmark.make_benchmark(
                root=case_root,
                n_frames=n_frames,
                seed=case['seed'],
                source=case['source'],
                recipe=case['recipe'],
                control_photons=case['control_photons'],
                bleaching=case['bleaching'],
                save_signal=False,
                )
            rigid = measure_method('fibresight_rigid', case, root)
            field = measure_method('fibresight_piecewise', case, root)
            resource_writer.writerow({
                **{name: case[name] for name in (
                    'case_order', 'case', 'source', 'recipe', 'seed',
                    'control_photons', 'bleaching')},
                'rigid_wall_seconds': rigid['wall_seconds'],
                'field_wall_seconds': field['wall_seconds'],
                'total_wall_seconds': rigid['wall_seconds'] + field['wall_seconds'],
                'peak_process_rss_bytes': max(
                    rigid['peak_process_rss_bytes'], field['peak_process_rss_bytes']),
                'peak_tree_threads': max(
                    rigid['peak_tree_threads'], field['peak_tree_threads']),
                'rigid_exit_code': rigid['exit_code'],
                'field_exit_code': field['exit_code'],
                })
            resource_file.flush()
            if rigid['exit_code'] or field['exit_code']:
                raise RuntimeError(f'FibreSight piecewise failed for {case_name}')
            benchmark.measure(
                case_root, ['fibresight_piecewise'], compare_references=False)
            metric_writer.writerows({
                **{name: case[name] for name in metric_fields[:7]},
                **row,
                } for row in _read_rows(case_root / 'metrics.csv'))
            metric_file.flush()
            for path in (
                    case_root / 'control.tif', case_root / 'truth.npz',
                    case_root / 'fibresight_rigid.npz',
                    case_root / 'fibresight_piecewise.npz',
                    case_root / 'metrics.csv',
                    case_root / 'piecewise_metrics.csv',
                    case_root / 'field_metrics.csv',
                    case_root / 'field_selection.csv',
                    case_root / 'field_confidence.csv',
                    ):
                path.unlink(missing_ok=True)


def run_reference_suite(root=SUITE_ROOT):
    root = Path(root)
    cases = benchmark_cases(root)
    for case_i, case in enumerate(cases, start=1):
        for method in REFERENCE_METHODS:
            if (case['root'] / REFERENCE_FILES[method]).exists():
                continue
            case_name = case['case']
            print(f'{case_i:02d}/{len(cases)}  {case_name}  {method}')
            log_root = case['root'] / 'logs'
            log_root.mkdir(parents=True, exist_ok=True)
            with (log_root / f'{method}_reference_convergence.txt').open('w') as log:
                subprocess.run(
                    _reference_command(method, case['root']),
                    cwd=PROJECT_ROOT,
                    env=_method_environment(method, root),
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    )


#%% suite measurements
def _case_rows(path, case):
    with Path(path).open(newline='') as file:
        return [
            {
                'case_order': case['case_order'],
                'case': case['case'],
                'source': case['source'],
                'recipe': case['recipe'],
                'seed': case['seed'],
                **row,
                }
            for row in DictReader(file)
            ]


def measure_suite(root=SUITE_ROOT):
    root = Path(root)
    check_suite(root)
    outputs = {
        'metrics.csv': 'suite_metrics.csv',
        'reference_metrics.csv': 'suite_reference_metrics.csv',
        'reference_frc.csv': 'suite_reference_frc.csv',
        'method_reference_convergence.csv': 'suite_reference_convergence.csv',
    }
    rows = {name: [] for name in outputs}
    for case in benchmark_cases(root):
        case_name = case['case']
        metrics_path = case['root'] / 'metrics.csv'
        if not metrics_path.exists():
            print(f'measuring {case_name}')
            benchmark.measure(case['root'], METHODS, compare_references=False)
        for name in outputs:
            path = case['root'] / name
            if path.exists():
                rows[name].extend(_case_rows(path, case))

    for name, output in outputs.items():
        if not rows[name]:
            continue
        with (root / output).open('w', newline='') as file:
            writer = DictWriter(file, fieldnames=rows[name][0])
            writer.writeheader()
            writer.writerows(rows[name])


#%% suite statistics
def _read_rows(path):
    with Path(path).open(newline='') as file:
        return list(DictReader(file))


def _case_values(rows, method, metric, **conditions):
    return {
        row['case']: float(row[metric]) for row in rows
        if row['method'] == method
        and all(row[name] == value for name, value in conditions.items())
        }


def _source_values(rows, method, metric, **conditions):
    selected = [
        row for row in rows
        if row['method'] == method
        and all(row[name] == value for name, value in conditions.items())
        ]
    recipes = {conditions['recipe']} if 'recipe' in conditions else set(RECIPES)
    values = {}
    for source in sorted({row['source'] for row in selected}):
        source_rows = [row for row in selected if row['source'] == source]
        if {row['recipe'] for row in source_rows} != recipes:
            continue
        values[source] = np.mean([float(row[metric]) for row in source_rows])
    return values


def _paired_values(first, second):
    names = sorted(first.keys() & second.keys())
    return (
        np.asarray([first[name] for name in names]),
        np.asarray([second[name] for name in names]),
        )


def _mean_ci(values, axis=0):
    from scipy.stats import t

    values = np.asarray(values)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    half_width = (
        t.ppf(0.975, n - 1) * values.std(axis=axis, ddof=1) / np.sqrt(n))
    return mean, mean - half_width, mean + half_width


def _summary_row(section, group, metric, method, cases, sources):
    case_values = np.asarray(list(cases.values()))
    source_values = np.asarray(list(sources.values()))
    mean, low, high = _mean_ci(source_values)
    return {
        'section': section,
        'group': group,
        'metric': metric,
        'method': method,
        'n_cases': len(cases),
        'n_sources': len(sources),
        'degrees_freedom': len(sources) - 1,
        'mean': float(mean),
        'ci_low': float(low),
        'ci_high': float(high),
        'median': float(np.median(case_values)),
        'minimum': float(case_values.min()),
        'maximum': float(case_values.max()),
        }


def _paired_plot(axis, rows, methods, metric, **conditions):
    selected = [
        row for row in rows
        if row['method'] in methods
        and all(row[name] == value for name, value in conditions.items())
        ]
    cases = sorted({int(row['case_order']) for row in selected})
    values = np.full((len(cases), len(methods)), np.nan)
    for row in selected:
        values[cases.index(int(row['case_order'])), methods.index(row['method'])] = float(
            row[metric])
    x = np.arange(len(methods))
    for case in values:
        axis.plot(x, case, color='0.78', linewidth=0.7, zorder=1)
    for method_i, method in enumerate(methods):
        finite = np.isfinite(values[:, method_i])
        axis.scatter(
            np.full(finite.sum(), method_i), values[finite, method_i],
            s=18, color=METHOD_COLOURS[method], alpha=0.65, zorder=2,
            )
        axis.scatter(
            method_i, np.nanmean(values[:, method_i]), marker='D', s=42,
            color=METHOD_COLOURS[method], edgecolor='white', linewidth=0.7, zorder=3,
            )
    labels = [_method_tick_label(method) for method in methods]
    axis.set_xticks(x, labels)
    axis.tick_params(axis='x', pad=7)
    if len(methods) > 5:
        axis.tick_params(axis='x', labelsize=7)
    axis.grid(axis='y', color='0.90', linewidth=0.6)
    axis.spines[['top', 'right']].set_visible(False)


def summarise_source_groups(root=SUITE_ROOT, figure_root=EXAMPLE_ROOT):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import sobel

    root = Path(root)
    figure_root = Path(figure_root)
    rigid_rows = _read_rows(root / 'suite_metrics.csv')
    piecewise_rows = _read_rows(root / 'piecewise_suite_metrics.csv')

    source_info = {}
    for source in SOURCES:
        image = benchmark._normalise(np.load(benchmark.SOURCE_ROOT / source)) ** 2
        gradient = np.hypot(sobel(image, axis=0), sobel(image, axis=1))
        source_info[source] = {
            'anatomy': SOURCE_ANATOMY[source],
            'image_side_px': image.shape[0],
            'latent_gradient_magnitude': float(np.mean(gradient)),
            }
    gradient_split = np.median([
        values['latent_gradient_magnitude'] for values in source_info.values()])
    for values in source_info.values():
        # the synthetic latent image uses this same percentile scaling before photon noise
        # 18 August 2026: the median gives two equal descriptive groups for ten sources
        values['gradient_group'] = (
            'lower five'
            if values['latent_gradient_magnitude'] <= gradient_split
            else 'higher five'
            )

    comparisons = (
        (
            'rigid', rigid_rows, 'fibresight_rigid',
            ('suite2p_rigid', 'caiman_rigid', 'patchwarp_rigid'),
            ),
        (
            'piecewise', piecewise_rows, 'fibresight_piecewise',
            ('suite2p_piecewise', 'caiman_piecewise',
             'patchwarp_affine', 'pyflowreg_piecewise'),
            ),
        )
    source_rows = []
    source_scores = {}
    for model, rows, fibresight_method, methods in comparisons:
        for method in (fibresight_method, *methods):
            scores = _source_values(
                rows, method, 'p95_error_px', group='heldout')
            source_scores[(model, method)] = scores
            for source, score in scores.items():
                source_rows.append({
                    'model': model,
                    'method': method,
                    'source': source,
                    **source_info[source],
                    'gradient_split': float(gradient_split),
                    'source_mean_p95_error_px': float(score),
                    })
    with (root / 'source_group_metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=source_rows[0])
        writer.writeheader()
        writer.writerows(source_rows)

    properties = (
        ('anatomy', ('fibre', 'somatic', 'mesoscale')),
        ('gradient_group', ('lower five', 'higher five')),
        ('image_side_px', (128, 256, 512)),
        )
    group_rows = []
    for model, _, fibresight_method, methods in comparisons:
        fibresight = source_scores[(model, fibresight_method)]
        for method in methods:
            competitor = source_scores[(model, method)]
            for property_name, groups in properties:
                for group in groups:
                    names = [
                        source for source in SOURCES
                        if source_info[source][property_name] == group
                        and source in fibresight and source in competitor
                        ]
                    fibresight_values = np.asarray([
                        fibresight[source] for source in names])
                    competitor_values = np.asarray([
                        competitor[source] for source in names])
                    advantage = competitor_values - fibresight_values
                    group_rows.append({
                        'model': model,
                        'property': property_name,
                        'group': group,
                        'comparison': method,
                        'n_sources': len(names),
                        'sources': ';'.join(names),
                        'fibresight_mean_p95_error_px': float(
                            fibresight_values.mean()) if len(names) else np.nan,
                        'comparison_mean_p95_error_px': float(
                            competitor_values.mean()) if len(names) else np.nan,
                        'mean_advantage_px': float(
                            advantage.mean()) if len(names) else np.nan,
                        'minimum_advantage_px': float(
                            advantage.min()) if len(names) else np.nan,
                        'maximum_advantage_px': float(
                            advantage.max()) if len(names) else np.nan,
                        })
    with (root / 'source_group_comparisons.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=group_rows[0])
        writer.writeheader()
        writer.writerows(group_rows)

    benchmark._format_benchmark_plots()
    figure, axes = plt.subplots(
        2, 3, figsize=(14.5, 8.8), constrained_layout=True)
    panel_names = {
        'anatomy': 'anatomy',
        'gradient_group': 'latent gradient information',
        'image_side_px': 'stored image side (px)',
        }
    # 18 August 2026: fixed ranges keep near-zero comparisons legible; clipped values stay labelled
    row_limits = {'rigid': (-0.25, 1.60), 'piecewise': (-0.25, 5.30)}
    for row_i, (model, _, fibresight_method, methods) in enumerate(comparisons):
        fibresight = source_scores[(model, fibresight_method)]
        lower, upper = row_limits[model]
        offsets = np.linspace(-0.24, 0.24, len(methods))
        for column_i, (property_name, groups) in enumerate(properties):
            axis = axes[row_i, column_i]
            axis.axhline(
                0, color=METHOD_COLOURS[fibresight_method], linewidth=1.0)
            for method_i, method in enumerate(methods):
                competitor = source_scores[(model, method)]
                for group_i, group in enumerate(groups):
                    names = [
                        source for source in SOURCES
                        if source_info[source][property_name] == group
                        and source in fibresight and source in competitor
                        ]
                    if not names:
                        continue
                    advantage = np.asarray([
                        competitor[source] - fibresight[source] for source in names])
                    position = group_i + offsets[method_i]
                    jitter = np.linspace(-0.025, 0.025, len(advantage))
                    visible = np.clip(advantage, lower, upper)
                    axis.scatter(
                        position + jitter, visible, s=14,
                        color=METHOD_COLOURS[method], alpha=0.35)
                    mean = float(advantage.mean())
                    axis.errorbar(
                        position, mean,
                        yerr=[[mean - max(float(advantage.min()), lower)],
                              [min(float(advantage.max()), upper) - mean]],
                        fmt='D', markersize=5, capsize=3,
                        color=METHOD_COLOURS[method], zorder=3,
                        )
                    if advantage.max() > upper:
                        axis.scatter(
                            position, upper - 0.04, marker='^', s=34,
                            facecolor='white', edgecolor=METHOD_COLOURS[method],
                            zorder=4)
                        axis.annotate(
                            f'{advantage.max():.1f}',
                            (position, upper - 0.04), xytext=(0, -11),
                            textcoords='offset points', ha='center', fontsize=6.5,
                            color=METHOD_COLOURS[method])
            axis.set_xticks(range(len(groups)), [str(group) for group in groups])
            axis.set_ylim(lower, upper)
            axis.set_title(f'{model}: {panel_names[property_name]}')
            axis.grid(axis='y', color='0.90', linewidth=0.6)
            axis.spines[['top', 'right']].set_visible(False)
            if column_i == 0:
                axis.set_ylabel('competitor minus FibreSight p95 error (px)')

    legend_methods = (
        'suite2p_rigid', 'caiman_rigid', 'patchwarp_rigid',
        'suite2p_piecewise', 'caiman_piecewise',
        'patchwarp_affine', 'pyflowreg_piecewise',
        )
    handles = [
        Line2D(
            [], [], marker='D', linestyle='none', markersize=5,
            color=METHOD_COLOURS[method],
            label=('PatchWarp affine*' if method == 'patchwarp_affine'
                   else METHOD_NAMES[method]),
            )
        for method in legend_methods]
    figure.legend(
        handles=handles[:3], loc='upper center', ncol=3,
        bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8)
    figure.legend(
        handles=handles[3:], loc='upper center', ncol=4,
        bbox_to_anchor=(0.5, 0.965), frameon=False, fontsize=8)
    _add_patchwarp_note(figure)
    figure.get_layout_engine().set(rect=(0, 0.08, 1, 0.86))
    figure.text(
        0.5, 0.045,
        'positive values mean lower FibreSight error; diamonds and bars show the mean and observed range; open triangles mark clipped values',
        ha='center', fontsize=8)
    figure.savefig(
        figure_root / 'registration_benchmark_source_groups.png',
        dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_accuracy(metrics, comparisons, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    _paired_plot(axes[0], metrics, RIGID_METHODS, 'p95_error_px', group='heldout')
    case_count = len({row['case'] for row in metrics if row['group'] == 'heldout'})
    axes[0].set_title(f'held-out error across {case_count} cases')
    axes[0].set_ylabel('95th-percentile error (px)')

    compared = [
        row for row in comparisons
        if row['section'] == 'accuracy' and row['metric'] == 'p95_error_px'
        ]
    x = np.arange(len(compared))
    advantage = np.array([float(row['fibresight_advantage']) for row in compared])
    low = np.array([float(row['ci_low']) for row in compared])
    high = np.array([float(row['ci_high']) for row in compared])
    for method_i, row in enumerate(compared):
        colour = METHOD_COLOURS[row['comparison']]
        axes[1].errorbar(
            method_i,
            advantage[method_i],
            yerr=[[advantage[method_i] - low[method_i]],
                  [high[method_i] - advantage[method_i]]],
            fmt='o', color=colour, capsize=4,
            )
    axes[1].axhline(0, color='0.45', linewidth=0.8)
    comparison_labels = [
        METHOD_NAMES[row['comparison']].replace(' ', '\n', 1)
        for row in compared
        ]
    axes[1].set_xticks(x, comparison_labels)
    axes[1].set_title('paired source mean advantage (95% t CI)')
    axes[1].set_ylabel('competitor error minus FibreSight error (px)')
    axes[1].text(
        0.48, 0.72, 'Bonferroni-corrected Wilcoxon p',
        transform=axes[1].transAxes, fontsize=8.5,
        )
    for row_i, row in enumerate(compared):
        method = row['comparison']
        corrected_p = float(row['wilcoxon_p_bonferroni'])
        suffix = ' (n.s.)' if corrected_p >= 0.05 else ''
        axes[1].text(
            0.48, 0.66 - 0.06 * row_i,
            f'{METHOD_NAMES[method]}: p = {corrected_p:.3g}{suffix}',
            color=METHOD_COLOURS[method],
            transform=axes[1].transAxes, fontsize=8.5,
            )
    axes[1].grid(axis='y', color='0.90', linewidth=0.6)
    axes[1].spines[['top', 'right']].set_visible(False)

    _paired_plot(
        axes[2], metrics, RIGID_METHODS, 'p95_error_px',
        group='nonrigid', recipe='local_deformation',
        )
    axes[2].set_title('rigid methods on local deformation')
    axes[2].set_ylabel('95th-percentile error (px)')
    axes[2].set_ylim(bottom=0)
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_registration_comparison(rigid_rows, piecewise_rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    panels = (
        (axes[0], rigid_rows, RIGID_METHODS, 'Rigid registration'),
        (axes[1], piecewise_rows, (
            'fibresight_piecewise', 'suite2p_piecewise', 'caiman_piecewise',
            'patchwarp_affine', 'pyflowreg_piecewise'), 'Piecewise registration'),
        )
    plot_limit = 10
    for axis, rows, methods, title in panels:
        _paired_plot(axis, rows, methods, 'p95_error_px', group='heldout')
        axis.set_title(title)
        axis.set_ylabel('held-out p95 error (px)')
        axis.set_ylim(0, plot_limit)
        for method_i, method in enumerate(methods):
            clipped = [
                float(row['p95_error_px']) for row in rows
                if row['method'] == method and row['group'] == 'heldout'
                and float(row['p95_error_px']) > plot_limit
                ]
            if clipped:
                axis.scatter(
                    method_i + np.linspace(-0.04, 0.04, len(clipped)),
                    np.full(len(clipped), plot_limit - 0.15),
                    marker='^', s=28, facecolor='white',
                    edgecolor=METHOD_COLOURS[method], zorder=4,
                    )
    figure.get_layout_engine().set(rect=(0, 0.08, 1, 1))
    figure.text(
        0.5, 0.012,
        'diamonds show means; open triangles mark values above 10 px; * 8x8 grid with 4x4 retry on failure',
        ha='center', fontsize=8,
        )
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_deformation(metrics, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    rigid_then_piecewise = (
        *RIGID_METHODS,
        'suite2p_piecewise', 'caiman_piecewise',
        'patchwarp_affine', 'pyflowreg_piecewise',
        )
    _paired_plot(
        axis, metrics, rigid_then_piecewise, 'p95_error_px',
        group='nonrigid', recipe='local_deformation',
        )
    axis.axvline(3.5, color='0.65', linewidth=0.8)
    axis.text(0.25, 0.98, 'rigid', transform=axis.transAxes, ha='center', va='top')
    axis.text(
        0.75, 0.98, 'piecewise', transform=axis.transAxes,
        ha='center', va='top')
    axis.set_title('motivation for non-rigid registration')
    axis.set_ylabel('95th-percentile error (px)')
    _add_patchwarp_note(figure)
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_references(reference_rows, frc_rows, convergence_rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()

    figure, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True)
    panels = (
        ('gradient_ncc', 'gradient NCC', 'correlation'),
        ('adjusted_rmse', 'intensity-adjusted RMSE', 'normalised error'),
        ('ssim', 'structural similarity', 'SSIM'),
        )
    for axis, (metric, title, ylabel) in zip(axes, panels):
        _paired_plot(axis, reference_rows, REFERENCE_METHODS, metric)
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    frequencies = np.linspace(0.02, 0.48, 100)
    sources = sorted({row['source'] for row in frc_rows})
    # 15 August 2026: average the four recipes within each source before the t interval
    for method in REFERENCE_METHODS:
        source_curves = []
        for source in sources:
            source_rows = [
                row for row in frc_rows
                if row['method'] == method and row['source'] == source
                ]
            if {row['recipe'] for row in source_rows} != set(RECIPES):
                continue
            curves = []
            cases = sorted({
                row['case'] for row in source_rows
                })
            for case in cases:
                rows = [
                    row for row in frc_rows
                    if row['method'] == method and row['case'] == case
                    ]
                x = np.array([
                    float(row['spatial_frequency_cycles_per_px']) for row in rows])
                y = np.array([float(row['frc']) for row in rows])
                curves.append(np.interp(frequencies, x, y))
            source_curves.append(np.mean(curves, axis=0))
        source_curves = np.asarray(source_curves)
        mean, low, high = _mean_ci(source_curves, axis=0)
        axes[3].plot(
            frequencies, mean,
            color=METHOD_COLOURS[method], label=METHOD_NAMES[method],
            )
        axes[3].fill_between(
            frequencies, low, high, color=METHOD_COLOURS[method], alpha=0.12)
    axes[3].set_title('Fourier ring correlation')
    axes[3].set_xlabel('spatial frequency (cycles/px)')
    axes[3].set_ylabel('correlation')
    axes[3].set_ylim(0.35, 1.01)
    axes[3].grid(color='0.90', linewidth=0.6)
    axes[3].spines[['top', 'right']].set_visible(False)
    axes[3].legend(frameon=False, fontsize=8)

    for method in REFERENCE_METHODS:
        counts = sorted({
            int(row['frames']) for row in convergence_rows if row['method'] == method})
        means = []
        for count in counts:
            values = [
                float(row['gradient_ncc']) for row in convergence_rows
                if row['method'] == method and int(row['frames']) == count
                ]
            means.append(np.mean(values))
        axes[4].plot(
            counts, means, marker='o', color=METHOD_COLOURS[method],
            label=METHOD_NAMES[method],
            )
    axes[4].set_title('reference convergence')
    axes[4].set_xlabel('input frames')
    axes[4].set_ylabel('gradient NCC')
    axes[4].set_xscale('log')
    axes[4].set_xticks(
        benchmark.REFERENCE_FRAME_COUNTS,
        [str(count) for count in benchmark.REFERENCE_FRAME_COUNTS],
        )
    axes[4].grid(color='0.90', linewidth=0.6)
    axes[4].spines[['top', 'right']].set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_resources(resource_rows, reference_rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    panels = (
        (resource_rows, 'wall_seconds', 'wall time', 'seconds', METHODS),
        (resource_rows, 'cpu_seconds', 'CPU time', 'seconds', METHODS),
        (resource_rows, 'peak_tree_rss_gib',
         'process-tree RSS (shared pages repeated)', 'summed RSS (GiB)', METHODS),
        (reference_rows, 'reference_seconds', 'reference construction',
         'seconds', REFERENCE_METHODS),
        )
    for axis, (rows, metric, title, ylabel, methods) in zip(axes.ravel(), panels):
        _paired_plot(axis, rows, methods, metric)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if metric in ('wall_seconds', 'cpu_seconds'):
            # 18 August 2026: the 512 px cases span two orders of magnitude
            axis.set_yscale('log')
            axis.set_ylabel('seconds (log scale)')
    _add_patchwarp_note(figure)
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _sweep_values(rows, method, x_name, x_value, **conditions):
    return np.asarray([
        float(row['p95_error_px']) for row in rows
        if row['method'] == method
        and row['group'] == 'heldout'
        and float(row[x_name]) == x_value
        and all(row[name] == value for name, value in conditions.items())
        ])


def _check_sweep(rows, cases, methods):
    observed = [
        (row['case'], row['method']) for row in rows
        if row['group'] == 'heldout'
        ]
    expected = {
        (case['case'], method) for case in cases for method in methods
        }
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(
            f'incomplete sweep: {len(observed)} of {len(expected)} held-out measurements')


def plot_noise_sweep(rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for axis, recipe in zip(axes.ravel()[:4], RECIPES):
        for method in METHODS:
            means = [
                _sweep_values(
                    rows, method, 'control_photons', photons, recipe=recipe).mean()
                for photons in PHOTON_COUNTS
                ]
            _plot_sweep_line(axis, PHOTON_COUNTS, means, method)
        axis.set_title(recipe.replace('_', ' '))
        axis.set_xlabel('control photons')
        axis.set_ylabel('95th-percentile error (px)')
        axis.set_xscale('log')
        axis.set_xticks(PHOTON_COUNTS, [str(value) for value in PHOTON_COUNTS])
        axis.tick_params(axis='x', which='minor', labelbottom=False)
        axis.grid(color='0.90', linewidth=0.6)
        axis.spines[['top', 'right']].set_visible(False)
        if recipe == 'focal_change':
            pyflowreg = [
                _sweep_values(
                    rows, 'pyflowreg_piecewise', 'control_photons', photons,
                    recipe=recipe).mean()
                for photons in PHOTON_COUNTS
                ]
            axis.set_ylim(0, 3)
            for photons, value in zip(PHOTON_COUNTS, pyflowreg):
                if value <= 3:
                    continue
                axis.plot(
                    photons, 2.95, marker='o', linestyle='none',
                    markerfacecolor='none',
                    markeredgecolor=METHOD_COLOURS['pyflowreg_piecewise'],
                    clip_on=False,
                    )
                axis.annotate(
                    f'{value:.1f} px',
                    xy=(photons, 2.95), xytext=(0, -20),
                    textcoords='offset points', ha='center', fontsize=7,
                    color=METHOD_COLOURS['pyflowreg_piecewise'],
                    arrowprops={
                        'arrowstyle': '->',
                        'color': METHOD_COLOURS['pyflowreg_piecewise'],
                        'linewidth': 0.8,
                        },
                    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    for method in METHODS:
        means = [
            _sweep_values(rows, method, 'control_photons', photons).mean()
            for photons in PHOTON_COUNTS
            ]
        _plot_sweep_line(axes[1, 1], PHOTON_COUNTS, means, method)
    axes[1, 1].set_title('all 12 image-motion pairs')
    axes[1, 1].set_xlabel('control photons')
    axes[1, 1].set_ylabel('mean 95th-percentile error (px)')

    axes[1, 2].remove()
    for axis in (axes[1, 1],):
        axis.set_xscale('log')
        axis.set_xticks(PHOTON_COUNTS, [str(value) for value in PHOTON_COUNTS])
        axis.tick_params(axis='x', which='minor', labelbottom=False)
        axis.grid(color='0.90', linewidth=0.6)
        axis.spines[['top', 'right']].set_visible(False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_sweep_line(axis, x, y, method):
    colour = METHOD_COLOURS[method]
    axis.plot(x, y, color=colour, linewidth=1.5, label=METHOD_NAMES[method])
    for x_value, y_value in zip(x, y):
        failed = y_value > 5
        axis.plot(
            x_value, y_value, marker='o', linestyle='none',
            markerfacecolor='none' if failed else colour,
            markeredgecolor=colour,
            )
        if failed:
            axis.annotate(
                '>5 px', (x_value, y_value), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=6, color=colour,
                )


def plot_whitening_sweep(rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()
    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for method in ('fibresight_rigid', *WHITENING_METHODS):
        means = [
            _sweep_values(rows, method, 'control_photons', photons).mean()
            for photons in PHOTON_COUNTS
            ]
        _plot_sweep_line(axis, PHOTON_COUNTS, means, method)
    axis.set_title('FibreSight whitening')
    axis.set_xlabel('control photons')
    axis.set_ylabel('mean 95th-percentile error (px)')
    axis.set_xscale('log')
    axis.set_xticks(PHOTON_COUNTS, [str(value) for value in PHOTON_COUNTS])
    axis.tick_params(axis='x', which='minor', labelbottom=False)
    axis.grid(color='0.90', linewidth=0.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_bleaching_sweep(rows, path):
    import matplotlib.pyplot as plt

    benchmark._format_benchmark_plots()
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for method in METHODS:
        means = [
            _sweep_values(rows, method, 'bleaching', bleaching).mean()
            for bleaching in BLEACHING_LEVELS
            ]
        axis.plot(
            100 * np.asarray(BLEACHING_LEVELS), means, marker='o',
            color=METHOD_COLOURS[method], label=METHOD_NAMES[method],
            )
    axis.set_title('ordinary motion across 3 images')
    axis.set_xlabel('total bleaching (%)')
    axis.set_ylabel('mean 95th-percentile error (px)')
    axis.set_xticks(100 * np.asarray(BLEACHING_LEVELS))
    axis.grid(color='0.90', linewidth=0.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarise_intensity_suite(root=INTENSITY_ROOT, figure_root=EXAMPLE_ROOT):
    root = Path(root)
    figure_root = Path(figure_root)
    photons = _read_rows(root / 'photon_metrics.csv')
    bleaching = _read_rows(root / 'bleaching_metrics.csv')
    _check_sweep(photons, photon_cases(root), PHOTON_METHODS)
    _check_sweep(bleaching, bleaching_cases(root), METHODS)
    figure_root.mkdir(parents=True, exist_ok=True)
    plot_noise_sweep(photons, figure_root / 'registration_benchmark_noise.png')
    plot_whitening_sweep(
        photons, figure_root / 'registration_benchmark_whitening.png')
    plot_bleaching_sweep(
        bleaching, figure_root / 'registration_benchmark_bleaching.png')


#%% real recording
def make_real_data(root=REAL_ROOT, raw_tiff=REAL_TIFF):
    import tifffile
    from fibre_sight.preprocessing import index_tiffs, read_tiffs

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw_tiff = Path(raw_tiff)
    with tifffile.TiffFile(raw_tiff) as tiff:
        n_frames = len(tiff.pages) // 2
        shape = tiff.pages[0].shape
        dtype = tiff.pages[0].dtype
    description = 'FibreSight registration benchmark'
    tiff_tags = [(270, 's', len(description) + 1, description, False)]
    movie = tifffile.memmap(
        root / 'control.tif', shape=(n_frames, *shape), dtype=dtype,
        bigtiff=True, metadata=None, extratags=tiff_tags)
    recording = index_tiffs(
        [raw_tiff], signal_channel=1, control_channel=2,
        sampling_frequency_hz=30, multiplexed=True)
    for pair in read_tiffs(recording):
        movie[pair['frame']] = pair['control']
    movie.flush()
    np.savez(
        root / 'source.npz',
        raw_tiff=str(raw_tiff),
        control_page_offset=np.int64(1),
        n_frames=np.int64(n_frames),
        sampling_frequency_hz=np.float32(30),
        )


def run_real_data(root=REAL_ROOT):
    root = Path(root)
    warm_up(root, RIGID_METHODS)
    fields = [
        'run_order', 'method', 'version', 'patchwarp_grid',
        'patchwarp_grid_attempts', 'allocated_cores', 'worker_type',
        'workers', 'native_threads_per_worker', 'wall_seconds', 'user_cpu_seconds',
        'system_cpu_seconds', 'average_cores', 'peak_process_rss_bytes',
        'peak_tree_rss_bytes', 'peak_tree_threads', 'exit_code',
        ]
    case = {'root': root}
    with (root / 'real_resources.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run_i, method in enumerate(RIGID_METHODS, start=1):
            print(f'{run_i}/{len(RIGID_METHODS)}  {method}')
            resources = measure_method(method, case, root)
            writer.writerow({'run_order': run_i, 'method': method, **resources})
            file.flush()
            if resources['exit_code']:
                raise RuntimeError(f'{method} failed; see its log')


#%% synthetic focal validation
def _focal_counts(truth, detected, start, stop, sampling_frequency_hz=30):
    focal = np.asarray(truth['focal'][start:stop], dtype=bool)
    detected = np.asarray(detected[start:stop], dtype=bool)
    blank = np.asarray(truth['blank_frame'][start:stop], dtype=bool)
    saturated = np.asarray(truth['saturated_frame'][start:stop], dtype=bool)
    truth_episodes = _episode_bounds(focal)
    detected_episodes = _episode_bounds(
        detected, merge_gap_frames=round(0.10 * sampling_frequency_hz))
    onset_errors = []
    onset_delays = []
    matched = 0
    used_detected = set()
    for truth_start, truth_stop in truth_episodes:
        overlaps = [
            (episode_i, detected_start, detected_stop)
            for episode_i, (detected_start, detected_stop) in enumerate(detected_episodes)
            if episode_i not in used_detected
            if detected_start < truth_stop and detected_stop > truth_start
            ]
        if overlaps:
            episode_i, detected_start, _ = overlaps[0]
            used_detected.add(episode_i)
            delay = (detected_start - truth_start) / sampling_frequency_hz
            onset_delays.append(delay)
            onset_errors.append(abs(delay))
            matched += 1

    normal = ~focal & ~blank & ~saturated
    return {
        'truth_focal_frames': int(focal.sum()),
        'detected_focal_frames': int(detected.sum()),
        'true_positive_frames': int(np.sum(focal & detected)),
        'normal_frames': int(normal.sum()),
        'normal_false_positive_frames': int(np.sum(normal & detected)),
        'blank_frames': int(blank.sum()),
        'blank_focal_frames': int(np.sum(blank & detected)),
        'saturated_frames': int(saturated.sum()),
        'saturated_focal_frames': int(np.sum(saturated & detected)),
        'truth_episodes': len(truth_episodes),
        'matched_episodes': matched,
        'detected_episodes': len(detected_episodes),
        'onset_errors_s': onset_errors,
        'onset_delays_s': onset_delays,
        }


def _focal_row(source, split, counts, thresholds=None):
    onset_errors = np.asarray(counts['onset_errors_s'], dtype=float)
    onset_delays = np.asarray(counts['onset_delays_s'], dtype=float)
    row = {
        'source': source,
        'split': split,
        **{name: value for name, value in counts.items()
           if name not in ('onset_errors_s', 'onset_delays_s')},
        'frame_recall': counts['true_positive_frames'] / counts['truth_focal_frames'],
        'episode_recall': counts['matched_episodes'] / counts['truth_episodes'],
        'normal_false_positive_rate': (
            counts['normal_false_positive_frames'] / counts['normal_frames']),
        'pmt_to_focal_fraction': counts['blank_focal_frames'] / counts['blank_frames'],
        'saturated_to_focal_fraction': (
            counts['saturated_focal_frames'] / counts['saturated_frames']),
        'median_onset_error_s': float(np.median(onset_errors)) if len(onset_errors) else np.nan,
        'p95_onset_error_s': float(np.percentile(onset_errors, 95)) if len(onset_errors) else np.nan,
        'median_onset_delay_s': float(np.median(onset_delays)) if len(onset_delays) else np.nan,
        }
    for name in (
            'canonical_focal', 'local_focal', 'high_frequency_focal',
            'control_gain_focal', 'canonical_ambiguous'):
        row[name] = thresholds[name] if thresholds is not None else np.nan
    return row


def _supports_provisional_status(row):
    # 17 August 2026: 0.10 s is three observations at the paired 30 Hz sampling rate
    return (
        row['frame_recall'] >= 0.95
        and row['episode_recall'] >= 0.95
        and row['normal_false_positive_rate'] <= 0.01
        and row['pmt_to_focal_fraction'] <= 0.01
        and row['saturated_to_focal_fraction'] <= 0.01
        and row['p95_onset_error_s'] <= 0.10
        )


def _write_focal_threshold_calibration(results, path):
    from fibre_sight.preprocessing import _focal_candidates, _quality_thresholds

    rows = []
    for n_mad in (0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4):
        counts = []
        for result in results:
            half = len(result['detected']) // 2
            quality = result['quality']
            outside_search = result['outside_search'][:half]
            baseline = quality['threshold_calibration']
            thresholds = _quality_thresholds(quality, baseline, n_mad)
            detected = (
                _focal_candidates(quality, thresholds)[:half]
                & ~quality['detector_artifact'][:half]
                & ~outside_search
                )
            counts.append(_focal_counts(result['truth'], detected, 0, half))
        total = {
            name: sum(count[name] for count in counts)
            for name in counts[0]
            if name not in ('onset_errors_s', 'onset_delays_s')
            }
        total['onset_errors_s'] = sum(
            (count['onset_errors_s'] for count in counts), [])
        total['onset_delays_s'] = sum(
            (count['onset_delays_s'] for count in counts), [])
        row = _focal_row('all', 'calibration', total)
        row = {
            'n_mad': n_mad,
            'frame_recall': row['frame_recall'],
            'episode_recall': row['episode_recall'],
            'normal_false_positive_rate': row['normal_false_positive_rate'],
            'pmt_to_focal_fraction': row['pmt_to_focal_fraction'],
            'saturated_to_focal_fraction': row['saturated_to_focal_fraction'],
            'median_onset_error_s': row['median_onset_error_s'],
            'p95_onset_error_s': row['p95_onset_error_s'],
            }
        rows.append(row)
    accepted = [row for row in rows if _supports_provisional_status(row)]
    selected = max(row['n_mad'] for row in accepted)
    for row in rows:
        row['selected'] = row['n_mad'] == selected
    with path.open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return selected


def _plot_focal_candidates(results, path):
    import matplotlib.pyplot as plt
    from fibre_sight.preprocessing import warp_frame

    benchmark._format_benchmark_plots()
    rows = []
    for result in results:
        truth = result['truth']
        detected = result['detected']
        normal = ~(truth['focal'] | truth['blank_frame'] | truth['saturated_frame'])
        sample_indices = np.flatnonzero(normal)
        sample_indices = sample_indices[
            np.linspace(0, len(sample_indices) - 1, 100, dtype=int)]
        display_limits = np.percentile(
            result['movie'][sample_indices], [0.1, 99.9])
        truth_episodes = _episode_bounds(truth['focal'])
        detected_episodes = _episode_bounds(detected, merge_gap_frames=3)
        used_detected = set()
        for truth_start, truth_stop in truth_episodes:
            overlaps = [
                (episode_i, detected_start, detected_stop)
                for episode_i, (detected_start, detected_stop) in enumerate(detected_episodes)
                if episode_i not in used_detected
                if detected_start < truth_stop and detected_stop > truth_start
                ]
            if overlaps:
                episode_i, detected_start, _ = overlaps[0]
                used_detected.add(episode_i)
                status = 'detected'
                onset_delay = (detected_start - truth_start) / 30
            else:
                status = 'missed'
                onset_delay = np.nan
            rows.append({
                **result,
                'start': truth_start,
                'stop': truth_stop,
                'status': status,
                'onset_delay_s': onset_delay,
                'peak_sigma_px': float(np.max(
                    truth['defocus_sigma_px'][truth_start:truth_stop])),
                'peak_contrast_loss': float(np.max(
                    truth['contrast_loss_fraction'][truth_start:truth_stop])),
                'display_limits': display_limits,
                })
        for episode_i, (detected_start, detected_stop) in enumerate(detected_episodes):
            if episode_i in used_detected:
                continue
            rows.append({
                **result,
                'start': detected_start,
                'stop': detected_stop,
                'status': 'false candidate',
                'onset_delay_s': np.nan,
                'peak_sigma_px': 0,
                'peak_contrast_loss': 0,
                'display_limits': display_limits,
                })

    figure, axes = plt.subplots(
        len(rows), 5, figsize=(12, 1.45 * len(rows) + 1), squeeze=False)
    headings = ('onset - 3', 'onset - 2', 'onset - 1', 'first focal', 'recovery')
    for axis, heading in zip(axes[0], headings):
        axis.set_title(heading, fontsize=8)
    for row_i, row in enumerate(rows):
        start = row['start']
        stop = min(row['stop'], len(row['movie']) - 1)
        frame_indices = [max(0, start - offset) for offset in (3, 2, 1)]
        frame_indices.extend((start, stop))
        low, high = row['display_limits']
        for axis, frame_i in zip(axes[row_i], frame_indices):
            estimate = row['estimates'][frame_i]
            image, _ = warp_frame(
                row['movie'][frame_i], estimate['shift_y'], estimate['shift_x'])
            axis.imshow(image, cmap='gray', vmin=low, vmax=high)
            axis.set_xticks([])
            axis.set_yticks([])
        split = 'calibration' if start < len(row['movie']) // 2 else 'held-out'
        onset_delay = row['onset_delay_s']
        delay = (
            f'{onset_delay:+.2f} s' if np.isfinite(onset_delay) else 'no onset')
        source = row['source']
        peak_sigma = row['peak_sigma_px']
        peak_loss = row['peak_contrast_loss']
        status = row['status']
        axes[row_i, 0].set_ylabel(
            f'{source}\n{split}; peak {peak_sigma:.0f} px, '
            f'{100 * peak_loss:.0f}%\n{status}; {delay}',
            fontsize=5.5, rotation=0, ha='right', va='center', labelpad=8)
    figure.subplots_adjust(
        left=0.23, right=0.995, bottom=0.01, top=0.985, hspace=0.08, wspace=0.02)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def measure_focal_suite(
        root=FOCAL_ROOT,
        figure_path=EXAMPLE_ROOT / 'registration_benchmark_focal_candidates.png',
        ):
    import tifffile
    from fibre_sight.preprocessing import make_local_references, measure_quality

    root = Path(root)
    rows = []
    counts_by_split = {'calibration': [], 'heldout': []}
    results = []
    for case in focal_cases(root):
        with np.load(case['root'] / 'truth.npz') as saved:
            truth = {name: saved[name] for name in saved.files}
        with np.load(case['root'] / 'fibresight_rigid.npz') as saved:
            result = {name: saved[name] for name in saved.files}
        movie = tifffile.memmap(case['root'] / 'control.tif')
        estimates = [
            {
                'shift_y': -float(result['shift_y'][frame_i]),
                'shift_x': -float(result['shift_x'][frame_i]),
                'peak_ratio': float(result['peak_ratio'][frame_i]),
                'tile_disagreement': float(result['tile_disagreement'][frame_i]),
                'out_of_range': bool(result['out_of_range'][frame_i]),
                'search_boundary': bool(result['search_boundary'][frame_i]),
                }
            for frame_i in range(len(movie))
            ]
        half = len(movie) // 2
        timestamps = np.arange(len(movie), dtype=float) / 30
        first_references = make_local_references(
            result['reference'], movie[:half], estimates[:half], 30,
            timestamps=timestamps[:half])
        second_references = make_local_references(
            result['reference'], movie[half:], estimates[half:], 30,
            timestamps=timestamps[half:])
        local_references = {
            'images': np.concatenate((
                first_references['images'], second_references['images'])),
            'reference_index': np.concatenate((
                first_references['reference_index'],
                second_references['reference_index'] + len(first_references['images']),
                )),
            'canonical_fallback': np.concatenate((
                first_references['canonical_fallback'],
                second_references['canonical_fallback'],
                )),
            }
        calibration = np.arange(len(movie)) < half
        quality = measure_quality(
            result['reference'], movie, estimates, 30,
            calibration_mask=calibration,
            local_references=local_references,
            timestamps=timestamps,
            )
        detected = quality['recommended_state'] == 'focal_loss'
        source_name = case['source']
        for split, start, stop in (
                ('calibration', 0, half), ('heldout', half, len(movie))):
            counts = _focal_counts(truth, detected, start, stop)
            counts_by_split[split].append(counts)
            rows.append(_focal_row(
                source_name, split, counts, quality['thresholds']))
        np.savez_compressed(
            case['root'] / 'focal_quality.npz',
            source=case['source'],
            seed=np.uint64(case['seed']),
            sampling_frequency_hz=np.float32(30),
            calibration_mask=calibration,
            recommended_state=np.asarray(quality['recommended_state'], dtype='<U16'),
            reason_code=np.asarray(quality['reason_code'], dtype='<U128'),
            canonical_gradient_ncc=quality['canonical_gradient_ncc'],
            local_gradient_ncc=quality['local_gradient_ncc'],
            high_frequency_fraction=quality['high_frequency_fraction'],
            control_gain=quality['control_gain'],
            detector_artifact=quality['detector_artifact'],
            threshold_calibration=quality['threshold_calibration'],
            analysis_valid=quality['analysis_valid'],
            version=result['version'],
            git_commit=result['git_commit'],
            git_dirty=result['git_dirty'],
            **quality['thresholds'],
            )
        results.append({
            'source': case['source'],
            'movie': movie,
            'truth': truth,
            'estimates': estimates,
            'detected': detected,
            'quality': quality,
            'outside_search': result['out_of_range'] | result['search_boundary'],
            })

    selected_mads = _write_focal_threshold_calibration(
        results, root / 'focal_threshold_calibration.csv')
    evaluated_mads = {
        result['quality']['thresholds']['focal_mads'] for result in results}
    if evaluated_mads != {selected_mads}:
        raise RuntimeError(
            f'calibration selected {selected_mads} MADs; evaluated {evaluated_mads}')

    for split, split_counts in counts_by_split.items():
        total = {
            name: sum(count[name] for count in split_counts)
            for name in split_counts[0]
            if name not in ('onset_errors_s', 'onset_delays_s')
            }
        total['onset_errors_s'] = sum(
            (count['onset_errors_s'] for count in split_counts), [])
        total['onset_delays_s'] = sum(
            (count['onset_delays_s'] for count in split_counts), [])
        rows.append(_focal_row('all', split, total))

    heldout = [
        row for row in rows
        if row['source'] != 'all' and row['split'] == 'heldout']
    status = (
        'provisional'
        if all(_supports_provisional_status(row) for row in heldout)
        else 'experimental'
        )
    for row in rows:
        row['classifier_status'] = status
    with (root / 'focal_metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    _plot_focal_candidates(results, figure_path)
    return rows


#%% correspondence-loss quality record
def measure_quality_suite(root=SUITE_ROOT):
    import tifffile
    from fibre_sight.preprocessing import measure_quality

    root = Path(root)
    rows = []
    for case in benchmark_cases(root):
        if case['recipe'] != 'focal_change':
            continue
        with np.load(case['root'] / 'truth.npz') as saved:
            truth = {name: saved[name] for name in saved.files}
        with np.load(case['root'] / 'fibresight_rigid.npz') as saved:
            result = {name: saved[name] for name in saved.files}
        movie = tifffile.memmap(case['root'] / 'control.tif')
        search_boundary = result['search_boundary']
        estimates = [
            {
                'shift_y': -float(result['shift_y'][frame_i]),
                'shift_x': -float(result['shift_x'][frame_i]),
                'peak_ratio': float(result['peak_ratio'][frame_i]),
                'tile_disagreement': float(result['tile_disagreement'][frame_i]),
                'out_of_range': bool(result['out_of_range'][frame_i]),
                'search_boundary': bool(search_boundary[frame_i]),
                }
            for frame_i in range(len(movie))
            ]
        quality = measure_quality(result['reference'], movie, estimates, 30)
        focal_truth = np.asarray(truth['focal'], dtype=bool)
        focal_detected = quality['recommended_state'] == 'focal_loss'
        overlap = focal_truth & focal_detected
        finite_gain = quality['control_gain'][np.isfinite(quality['control_gain'])]
        finite_axial = quality['axial_similarity'][np.isfinite(quality['axial_similarity'])]
        rows.append({
            'case_order': case['case_order'],
            'case': case['case'],
            'source': case['source'],
            'frames': len(movie),
            'truth_focal_frames': int(focal_truth.sum()),
            'detected_focal_frames': int(focal_detected.sum()),
            'overlap_frames': int(overlap.sum()),
            'ambiguous_frames': int(np.sum(quality['recommended_state'] == 'ambiguous')),
            'out_of_range_frames': int(np.sum(
                quality['recommended_state'] == 'out_of_range')),
            'analysis_valid_frames': int(quality['analysis_valid'].sum()),
            'truth_focal_episodes': len(_episode_lengths(focal_truth)),
            'detected_focal_episodes': len(quality['focal_loss_episodes']),
            'detected_episode_durations_s': ';'.join(
                f'{episode["duration_s"]:.3f}'
                for episode in quality['focal_loss_episodes']),
            'minimum_canonical_gradient_ncc': float(
                np.nanmin(quality['canonical_gradient_ncc'])),
            'minimum_local_gradient_ncc': float(
                np.nanmin(quality['local_gradient_ncc'])),
            'minimum_control_gain': float(np.min(finite_gain)),
            'median_axial_similarity': float(np.median(finite_axial)),
            'local_reference_fallback_blocks': int(
                np.sum(quality['local_references']['canonical_fallback'])),
            })

    output = root / 'quality_metrics.csv'
    with output.open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def measure_real_data(root=REAL_ROOT):
    import matplotlib.pyplot as plt
    import tifffile
    from fibre_sight.preprocessing import warp_frame

    benchmark._format_benchmark_plots()
    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    results = {}
    for method in RIGID_METHODS:
        with np.load(root / f'{method}.npz') as saved:
            results[method] = {name: saved[name] for name in saved.files}

    largest_shift = max(
        np.nanmax(np.abs(results[method][axis]))
        for method in RIGID_METHODS for axis in ('shift_y', 'shift_x'))
    margin = int(np.ceil(largest_shift)) + 5
    crop = np.s_[margin:-margin, margin:-margin]
    scores = np.empty((len(movie), len(RIGID_METHODS)), dtype=np.float32)
    for frame_i, frame in enumerate(movie):
        for method_i, method in enumerate(RIGID_METHODS):
            corrected, _ = warp_frame(
                frame,
                -float(results[method]['shift_y'][frame_i]),
                -float(results[method]['shift_x'][frame_i]),
                )
            scores[frame_i, method_i] = benchmark._gradient_ncc(
                corrected[crop], results[method]['reference'][crop])
    all_methods_finite = np.all(np.isfinite(scores), axis=1)
    disagreement = np.full(len(movie), np.nan, dtype=np.float32)
    disagreement[all_methods_finite] = np.ptp(scores[all_methods_finite], axis=1)
    heldout = all_methods_finite.copy()
    heldout[:len(movie) // 2] = False
    order = np.flatnonzero(heldout)[np.argsort(disagreement[heldout])[::-1]]
    ranks = np.full(len(movie), np.nan)
    ranks[order] = np.arange(1, len(order) + 1)

    rows = []
    for frame_i in range(len(movie)):
        rows.append({
            'frame': frame_i,
            **{
                f'{method}_gradient_ncc': scores[frame_i, method_i]
                for method_i, method in enumerate(RIGID_METHODS)
                },
            'inter_method_range': disagreement[frame_i],
            'disagreement_rank': ranks[frame_i],
            'all_methods_finite': all_methods_finite[frame_i],
            })
    with (root / 'real_gradient_ncc.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    sample = np.asarray(movie[np.linspace(0, len(movie) - 1, 100, dtype=int)])
    low, high = np.percentile(sample, [0.1, 99.9])
    figure, axes = plt.subplots(5, 4, figsize=(20, 7.5), constrained_layout=True)
    figure.suptitle('self-reference gradient NCC; diagnostic only')
    for axis, frame_i in zip(axes.ravel(), order[:20]):
        panels = [np.asarray(movie[frame_i], dtype=np.float32)]
        for method in RIGID_METHODS:
            median_y = np.nanmedian(results[method]['shift_y'][:len(movie) // 2])
            median_x = np.nanmedian(results[method]['shift_x'][:len(movie) // 2])
            corrected, _ = warp_frame(
                movie[frame_i],
                median_y - float(results[method]['shift_y'][frame_i]),
                median_x - float(results[method]['shift_x'][frame_i]),
                )
            panels.append(corrected)
        shown = [
            np.clip(255 * np.nan_to_num((panel - low) / (high - low)), 0, 255)
            for panel in panels
            ]
        # 16 August 2026: 16 px remains visible once the whole sheet is scaled down
        gutter = np.zeros((shown[0].shape[0], 16), dtype=np.float32)
        row = [shown[0]]
        for panel in shown[1:]:
            row.extend((gutter, panel))
        axis.imshow(np.concatenate(row, axis=1), cmap='gray', vmin=0, vmax=255)
        axis.set_title(
            f'frame {frame_i} | NCC range {disagreement[frame_i]:.3f}', fontsize=8)
        labels = ('raw', *(
            f'{METHOD_NAMES[method]}\n{scores[frame_i, method_i]:.3f}'
            for method_i, method in enumerate(RIGID_METHODS)
            ))
        for panel_i, label in enumerate(labels):
            axis.text(
                (panel_i + 0.5) / len(labels), 0.04, label,
                color='white', fontsize=5.5, ha='center', transform=axis.transAxes,
                bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 1, 'edgecolor': 'none'},
                )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(root / 'real_disagreement.png', dpi=180)
    plt.close(figure)


def summarise_suite(root=SUITE_ROOT, figure_root=EXAMPLE_ROOT):
    root = Path(root)
    figure_root = Path(figure_root)
    metrics = _read_rows(root / 'suite_metrics.csv')
    references = _read_rows(root / 'suite_reference_metrics.csv')
    frc = _read_rows(root / 'suite_reference_frc.csv')
    convergence = _read_rows(root / 'suite_reference_convergence.csv')
    base_references = [row for row in references if row['source'] in BASE_SOURCES]
    base_frc = [row for row in frc if row['source'] in BASE_SOURCES]
    base_convergence = [row for row in convergence if row['source'] in BASE_SOURCES]
    resources = _read_rows(root / 'resources.csv')
    resources = [row for row in resources if int(row['exit_code']) == 0]
    for row in resources:
        row['cpu_seconds'] = str(
            float(row['user_cpu_seconds']) + float(row['system_cpu_seconds']))
        row['peak_tree_rss_gib'] = str(float(row['peak_tree_rss_bytes']) / 2 ** 30)

    summaries = []
    groups = (
        ('accuracy', 'heldout', metrics, METHODS,
         ('median_error_px', 'p95_error_px', 'valid_fraction'), {'group': 'heldout'}),
        ('accuracy', 'local_deformation_nonrigid', metrics, METHODS,
         ('median_error_px', 'p95_error_px'),
         {'group': 'nonrigid', 'recipe': 'local_deformation'}),
        ('reference', 'all', base_references, REFERENCE_METHODS,
         ('gradient_ncc', 'adjusted_rmse', 'ssim', 'heldout_p95_error_px'), {}),
        ('resources', 'all', resources, METHODS,
         ('wall_seconds', 'cpu_seconds', 'average_cores', 'peak_tree_rss_gib'), {}),
        )
    for section, group, rows, methods, metric_names, conditions in groups:
        for metric in metric_names:
            for method in methods:
                cases = _case_values(rows, method, metric, **conditions)
                sources = _source_values(rows, method, metric, **conditions)
                summaries.append(_summary_row(
                    section, group, metric, method, cases, sources))

    with (root / 'suite_summary.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)

    comparisons = []
    comparison_groups = (
        ('accuracy', 'heldout', metrics,
         (('median_error_px', False), ('p95_error_px', False)), {'group': 'heldout'}),
        ('reference', 'all', base_references,
         (('gradient_ncc', True), ('adjusted_rmse', False),
          ('ssim', True), ('heldout_p95_error_px', False)), {}),
        )
    for section, group, rows, metric_names, conditions in comparison_groups:
        for metric, higher_is_better in metric_names:
            fibresight_cases = _case_values(
                rows, 'fibresight_rigid', metric, **conditions)
            fibresight = _source_values(
                rows, 'fibresight_rigid', metric, **conditions)
            competitors = (
                RIGID_METHODS[1:] if section == 'accuracy'
                else REFERENCE_METHODS[1:]
                )
            for method in competitors:
                competitor_case_map = _case_values(rows, method, metric, **conditions)
                competitor_source_map = _source_values(rows, method, metric, **conditions)
                fibresight_case_values, competitor_cases = _paired_values(
                    fibresight_cases, competitor_case_map)
                fibresight_values, competitor = _paired_values(
                    fibresight, competitor_source_map)
                advantage = (
                    fibresight_values - competitor if higher_is_better
                    else competitor - fibresight_values
                    )
                case_advantage = (
                    fibresight_case_values - competitor_cases if higher_is_better
                    else competitor_cases - fibresight_case_values
                    )
                mean, low, high = _mean_ci(advantage)
                wilcoxon_statistic = np.nan
                wilcoxon_p_raw = np.nan
                wilcoxon_p_bonferroni = np.nan
                wilcoxon_n = 0
                if section == 'accuracy' and metric == 'p95_error_px':
                    from scipy.stats import wilcoxon

                    test = wilcoxon(fibresight_case_values, competitor_cases)
                    wilcoxon_statistic = float(test.statistic)
                    wilcoxon_p_raw = float(test.pvalue)
                    wilcoxon_p_bonferroni = min(1, 3 * wilcoxon_p_raw)
                    wilcoxon_n = len(case_advantage)
                comparisons.append({
                    'section': section,
                    'group': group,
                    'metric': metric,
                    'comparison': method,
                    'better_value': 'higher' if higher_is_better else 'lower',
                    'n_cases': len(case_advantage),
                    'n_sources': len(advantage),
                    'degrees_freedom': len(advantage) - 1,
                    'fibresight_mean': float(fibresight_values.mean()),
                    'comparison_mean': float(competitor.mean()),
                    'fibresight_advantage': float(mean),
                    'ci_low': float(low),
                    'ci_high': float(high),
                    'fibresight_case_wins': int(np.sum(case_advantage > 0)),
                    'fibresight_source_wins': int(np.sum(advantage > 0)),
                    'wilcoxon_statistic': wilcoxon_statistic,
                    'wilcoxon_p_raw': wilcoxon_p_raw,
                    'wilcoxon_p_bonferroni': wilcoxon_p_bonferroni,
                    'wilcoxon_n': wilcoxon_n,
                    })
    with (root / 'suite_comparisons.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)

    figure_root.mkdir(parents=True, exist_ok=True)
    plot_accuracy(
        metrics, comparisons, figure_root / 'registration_benchmark_accuracy.png')
    plot_deformation(
        metrics, figure_root / 'registration_benchmark_deformation.png')
    plot_references(
        base_references, base_frc, base_convergence,
        figure_root / 'registration_benchmark_references.png')
    plot_resources(
        resources, base_references, figure_root / 'registration_benchmark_resources.png')


#%% command line
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'step', choices=[
            'make', 'run', 'measure', 'summarise', 'review', 'references',
            'run-intensity', 'summarise-intensity',
            'make-real', 'run-real', 'measure-real',
            'make-focal', 'run-focal', 'measure-focal',
            'run-piecewise', 'measure-piecewise',
            'measure-auto',
            'validate-auto',
            'summarise-piecewise',
            'run-piecewise-intensity',
            'quality', 'review-rigid',
            'method', 'reference', 'tile', 'measure-tile', 'field', 'measure-field',
            ])
    parser.add_argument('method', nargs='?')
    parser.add_argument('case_root', nargs='?', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--frames', type=int, default=2000)
    args = parser.parse_args()
    suite_root = args.root or SUITE_ROOT
    intensity_root = args.root or INTENSITY_ROOT
    real_root = args.root or REAL_ROOT
    focal_root = args.root or FOCAL_ROOT

    if args.step == 'make':
        make_suite(suite_root, args.frames)
    elif args.step == 'run':
        run_suite(suite_root)
    elif args.step == 'measure':
        measure_suite(suite_root)
    elif args.step == 'summarise':
        summarise_suite(suite_root)
    elif args.step == 'review':
        make_review_movie(suite_root)
    elif args.step == 'review-rigid':
        make_rigid_review_movie(suite_root)
    elif args.step == 'references':
        run_reference_suite(suite_root)
    elif args.step == 'run-intensity':
        run_intensity_suite(intensity_root, args.frames)
    elif args.step == 'summarise-intensity':
        summarise_intensity_suite(intensity_root)
    elif args.step == 'make-real':
        make_real_data(real_root)
    elif args.step == 'run-real':
        run_real_data(real_root)
    elif args.step == 'measure-real':
        measure_real_data(real_root)
    elif args.step == 'make-focal':
        make_focal_suite(focal_root, args.frames)
    elif args.step == 'run-focal':
        run_focal_suite(focal_root)
    elif args.step == 'measure-focal':
        measure_focal_suite(focal_root)
    elif args.step == 'run-piecewise':
        run_piecewise_suite(suite_root)
    elif args.step == 'measure-piecewise':
        measure_piecewise_suite(suite_root)
    elif args.step == 'measure-auto':
        measure_auto_suite(suite_root)
    elif args.step == 'validate-auto':
        run_auto_validation(args.root or AUTO_ROOT, args.frames)
    elif args.step == 'summarise-piecewise':
        summarise_piecewise_suite(suite_root, INTENSITY_ROOT)
    elif args.step == 'run-piecewise-intensity':
        run_piecewise_intensity_suite(intensity_root, args.frames)
    elif args.step == 'quality':
        measure_quality_suite(suite_root)
    elif args.step == 'method':
        _run_method(args.method, args.case_root)
    elif args.step == 'tile':
        _run_tile_method(args.method, args.case_root)
    elif args.step == 'measure-tile':
        benchmark.measure_tile_evidence(suite_root)
    elif args.step == 'field':
        _run_field_method(args.method, args.case_root)
    elif args.step == 'measure-field':
        benchmark.measure_tile_fields(suite_root)
    else:
        _run_reference(args.method, args.case_root)


if __name__ == '__main__':
    main()
