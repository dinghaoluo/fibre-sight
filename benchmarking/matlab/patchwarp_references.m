function patchwarp_references(patchwarp_root, movie_path, output_path, counts)
% Created on 15 August 2026
%
% measure PatchWarp reference convergence at the requested frame counts
%
% @author: Dinghao Luo

addpath(genpath(fullfile(patchwarp_root, 'PatchWarp')))
mkdir(output_path)
seconds = zeros(size(counts));
selected = cell(size(counts));
for count_i = 1:length(counts)
    count = counts(count_i);
    work_path = fullfile(output_path, sprintf('n%d', count));
    [reference, selected{count_i}, seconds(count_i)] = make_patchwarp_reference(...
        movie_path, work_path, count);
    write_tiff(fullfile(output_path, sprintf('reference_%d.tif', count)), int16(reference));
end
save(fullfile(output_path, 'reference_times.mat'), 'counts', 'seconds', 'selected', '-v6');
end
