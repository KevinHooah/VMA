function digit_predict = event2digit(sig, event_start_idx, event_end_idx, event_label)
% ========================%
% Converting the event-based prediction to signal digit based prediction.
% Input:
%       sig: signal itself
%       event_start_idx: index array of starting points of detected events
%       event_end_idx: index array of ending points of detected events
%       event_label: the event-based prediction result
% Output:
%       digit-based prediction result.
% ========================%
sz = size(sig(:,1));
digit_predict = zeros(sz);

for idx = 1:length(event_start_idx)
    start_idx = event_start_idx(idx);
    end_idx = event_end_idx(idx);
    label = event_label(idx);
    digit_predict(start_idx:end_idx) = label;
end
end