function patchwarp_benchmark(patchwarp_root, movie_path, output_path, run_affine, warp_blocksize, worker_num)
% Created on 15 August 2026
%
% run the pinned PatchWarp benchmark on one chronological TIFF
%
% @author: Dinghao Luo

addpath(genpath(fullfile(patchwarp_root, 'PatchWarp')))

source_path = fileparts(movie_path);
prewarp_path = fullfile(output_path, 'pre_warp');
target_path = fullfile(prewarp_path, 'target');
mkdir(prewarp_path)
mkdir(target_path)

info = imfinfo(movie_path);
n_reference = floor(length(info) / 2);
block_frames = 50;
% 15 August 2026: the first half is the fixed calibration boundary
[reference, selected, reference_seconds] = make_patchwarp_reference(...
    movie_path, fullfile(output_path, 'reference'), n_reference);
write_tiff(fullfile(target_path, 'template_AVG1.tif'), int16(reference));

delete(gcp('nocreate'))
maxNumCompThreads(1);
% 15 August 2026: keep each pool's job files with its temporary benchmark output
cluster = parcluster('local');
job_path = fullfile(output_path, 'parallel_jobs');
mkdir(job_path)
cluster.JobStorageLocation = job_path;
parpool(cluster, worker_num)
pctRunOnAll maxNumCompThreads(1)
% 16 August 2026: PatchWarp's affine ECC samples gradient points with randi
rng(42, 'twister')

rigid_start = tic;
patchwarp_rigid(...
    source_path, prewarp_path, 1, 1, 1, 'rank', 32, 1, 0.2, 1, 0.8, false, ...
    block_frames, 0, [], 'f');
rigid_seconds = toc(rigid_start);

affine_seconds = 0;
if run_affine
    postwarp_path = fullfile(output_path, 'post_warp');
    affine_start = tic;
    patchwarp_affine(...
        prewarp_path, postwarp_path, 1, 1, 32, 1, 1, warp_blocksize, 0.15, 6, 0, ...
        50, 10, 0.5, 1, 'affine', 1, 50, block_frames, worker_num, 0);
    affine_seconds = toc(affine_start);
end

save(fullfile(output_path, 'benchmark_times.mat'), ...
    'reference_seconds', 'rigid_seconds', 'affine_seconds', ...
    'n_reference', 'selected', 'warp_blocksize', '-v6');
delete(gcp('nocreate'))
end
