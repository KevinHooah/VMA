function [fused_result, fused_conf] = winBaseFusion(i_pred, i_conf, v_pred, v_conf)
% ========================%
% This function is to fuse the two window-based prediction by comparing their confidence.
% Input:
%       i_pred, i_conf: imu prediction result and confidence
%       v_pred, v_conf: vibration  prediction result and confidence
% Output:
%       fused prediction result.
% ========================%
fused_result = i_pred;
fused_conf = i_conf;

for idx = 1:length(fused_result)
    if i_conf(idx) < v_conf(idx)
        fused_result(idx) = v_pred(idx);
        fused_conf(idx) = v_conf(idx);
    end
end
end
