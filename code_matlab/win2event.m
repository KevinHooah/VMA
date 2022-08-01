function [event_predict, event_conf] = win2event(sig, event_start_idx, event_end_idx, win_label, win_conf, num_mins, winSize)
% ========================%
% This function is to convert the window-based prediction back to event-based.
% Input:
%       sig: signal itself
%       event_start_idx: index array of starting points of detected events
%       event_end_idx: index array of ending points of detected events
%       win_label: window-based prediction class
%       win_conf: window-based predicton confidence
%       num_mins: length of minutes of the signal, to estimate the sampling rate
%       winsSize: window length in seconds
% Output:
%       event-based prediction result and prediction confidence.
% ========================%
event_predict = zeros(length(event_start_idx),1);
event_conf = zeros(length(event_start_idx),1);

y = sig(:,1);
sec_len = ceil(length(y)/(num_mins*60));
if mod(sec_len,2) ~= 0
    sec_len = sec_len + 1;
end
windowSize = winSize * sec_len;
segLength = floor(windowSize/2);

sz = size(win_label);
start_index_array = zeros(sz);
end_index_array = zeros(sz);

% hash the window index to an array
for jdx = 1:length(win_label)
    start_index_array(jdx) = 1+(jdx-1)*segLength;
    end_index_array(jdx) = start_index_array(jdx) + windowSize-1; 
end
% start to map the result
for idx = 1:length(event_start_idx)
%     idx

    start_idx = event_start_idx(idx);
    end_idx = event_end_idx(idx);

    [~,win_label_start_idx] = (min(abs(start_index_array - start_idx)));
    [~,win_label_end_idx] = (min(abs(end_index_array - end_idx)));

    win_start_idx = start_index_array(win_label_start_idx);
    win_end_idx = end_index_array(win_label_end_idx);

    
    if abs(win_start_idx - start_idx) > segLength
        % that means over half of the window is not in the event
        if win_start_idx < start_idx
            win_label_start_idx = win_label_start_idx + 1;
        else
            win_label_start_idx = win_label_start_idx - 1;
        end
    end

    if abs(win_end_idx - end_idx) > segLength
        if win_end_idx > end_idx
            win_label_end_idx = win_label_end_idx - 1;
        else
            win_label_end_idx = win_label_end_idx + 1;
        end
    end

    if win_label_end_idx < win_label_start_idx
        % for those super short events
        win_label_end_idx = win_label_start_idx;
    end

    label_cache = win_label(win_label_start_idx:win_label_end_idx);
    conf_cache = win_conf(win_label_start_idx:win_label_end_idx);

    this_event_label = mode(label_cache);
    idx_array_of_label = label_cache == this_event_label;
    conf_array_of_label = conf_cache(idx_array_of_label);
    this_event_conf = max(conf_array_of_label);

    event_predict(idx) = this_event_label;
    event_conf(idx) = this_event_conf;
end