function result = nanmedian(values, dimension)
% Created on 15 August 2026
%
% supply the old reduction used by PatchWarp v1.3.3
%
% @author: Dinghao Luo

result = median(values, dimension, 'omitnan');
end
