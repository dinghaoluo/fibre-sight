function result = nanmean(values, dimension)
% Created on 15 August 2026
%
% supply the old reduction used by PatchWarp v1.3.3
%
% @author: Dinghao Luo

result = mean(values, dimension, 'omitnan');
end
