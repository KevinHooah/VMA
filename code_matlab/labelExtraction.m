function label = labelExtraction(y, winsSize, num_mins)
% ========================%
% This function is to extract the labels from sliding windows' data with half overlapping.
% Input:
%       y: prediction result
%       winsSize: window length in seconds
%       num_mins: length of minutes of the y, to estimate the sampling rate
% Output:
%       window-based prediction result.
% ========================%
sec_len = ceil(length(y)/(num_mins*60));
if mod(sec_len,2) ~= 0
    sec_len = sec_len + 1;
end

windowSize = winsSize * sec_len;

segLength = floor(windowSize/2); %the length of non-overlapping part
m = floor(length(y)/segLength) - 1;
label = zeros(m,1);
trim = 0;

for i=1:m
    idx_start = 1 + segLength * (i - 1);
    idx_end = idx_start + windowSize - 1;

    if idx_end>length(y)
        trim = 1;
        offset = idx_end-length(y);
        disp(offset)
        idx_end = length(y);
    end

    if length(y(idx_start:idx_end)) == windowSize
        cache_label = y(idx_start:idx_end);
        label(i) = mode(cache_label);
    end

    if trim
        label = label(1:end-offset);
    end

end



