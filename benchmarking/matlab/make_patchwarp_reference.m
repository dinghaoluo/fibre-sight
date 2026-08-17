function [reference, selected, seconds] = make_patchwarp_reference(movie_path, work_path, n_reference)
% Created on 15 August 2026
%
% make PatchWarp's two-pass rigid reference from the fixed calibration frames
%
% @author: Dinghao Luo

mkdir(work_path)
reference_start = tic;
info = imfinfo(movie_path);
first_frame = imread(movie_path, 1, 'Info', info);
frames = zeros(size(first_frame, 1), size(first_frame, 2), n_reference, 'like', first_frame);
frames(:, :, 1) = first_frame;
for frame_i = 2:n_reference
    frames(:, :, frame_i) = imread(movie_path, frame_i, 'Info', info);
end

input_path = fullfile(work_path, 'reference_input.tif');
write_tiff(input_path, frames)
initial_reference = make_template_fftdenoise(frames, [], [], 0.2, false);
clear frames

pyramid_registration(...
    input_path, initial_reference, work_path, 1, 1, 0, 0, 1, ...
    'rank', 32, 0.8, 0);
aligned = read_tiff(fullfile(work_path, 'reference_input_corrected.tif'), 1, 1, 0);
[reference, selected] = make_template_fftdenoise(aligned, [], [], 0.2, false);
seconds = toc(reference_start);
end
