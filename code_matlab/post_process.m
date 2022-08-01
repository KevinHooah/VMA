% ========================%
% This function is to post process the result after the python-based model prediction result.
% Required files:
%       predicton result: inter_and_baseline.mat or final_result.mat
%       vibration feature: _vib_extr.mat
%       vibration event index: _vib_idx.mat
%       imu_feature: _extr.mat
%       cleaned_signal: _cleaned.mat
% ========================%
clc, clear, close all
addpath('../output/')
addpath('../data_extracted/imu/')
addpath('../data_extracted/vib/')
addpath('../data_extracted/cleaned_data/')
addpath('../data_extracted/vib_index/')

save_file = 1; % whethre to save the result or not
%% set the target
person_id = 3;
location_id = 1;
predict_intermediate = 0;
%%%
% To use this code, if you want to get the intermediate domain prediction,
% load inter_and_baseline.mat, Then set predict_intermediate = 1, and put 
% the person_id and location_id as the intemediate domain. 
% If you want to have the M-A Direct predict baseline, set
% predict_intermediate = 0, and and put the person_id and location_id as
% the final target domain.
% For final VMA prediction result, load final_result.mat, and set
% predict_intermediate = 0. Put the person_id and location_id as
% the final target domain.
%%%
data_name = ['p', num2str(person_id), 'l', num2str(location_id)];
% load('inter_and_baseline.mat')
load('final_result.mat')
load([data_name, '_vib_extr.mat'])
load([data_name,'_extr.mat'])
load([data_name,'_cleaned.mat'])
load([data_name,'_vib_idx.mat'])

if predict_intermediate
    disp('Intermediate Domain Result: ')
    %% Get each modalitie's result first
    % Intermediate Domain
    % Process the vibration data first
    [v_w_conf, v_w_pred] = max(inter_vib_conf_w,[],2);
    [v_k_conf, v_k_pred] = max(inter_vib_conf_k,[],2);
    v_k_pred = v_k_pred + 3; % living area statrs from 4;
    v_k_pred(v_k_pred>6) = v_k_pred(v_k_pred>6) + 1; % vibration doesn't have class 7 event detected;
    
    v_pred = [v_w_pred; v_k_pred];
    cm_v = confusionchart(vib_ext(:,1),v_pred,'Normalization','row-normalized');
    cm_v = cm_v.NormalizedValues;
    close all;
    acc_v_inter = trace(cm_v)/length(cm_v); % Avarage activity rcognition accuracy
    disp(['Intermdeiate vibration accuracy: ', num2str(acc_v_inter)])
    
    % Process the IMU data then
    [i_w_conf, i_w_pred] = max(inter_imu_conf_w,[],2);
    [i_k_conf, i_k_pred] = max(inter_imu_conf_k,[],2);
    i_k_pred = i_k_pred + 3;
    i_pred = [i_w_pred; i_k_pred];
    i_conf = [i_w_conf; i_k_conf];
    cm_i = confusionchart(ext(:,1),i_pred,'Normalization','row-normalized');
    cm_i = cm_i.NormalizedValues;
    close all;
    acc_i_inter = trace(cm_i)/length(cm_i);
    disp(['Intermdeiate IMU accuracy: ', num2str(acc_i_inter)])
    
    %% Fuse the result
    % event to digit
    work_start_idx = [start_type, start_mouse, start_write]'; clearvars start_write start_type start_mouse
    work_end_idx = [end_type, end_mouse, end_write]'; clearvars end_write end_type end_mouse
    digit_pred_v_work = event2digit(work_cleaned, work_start_idx, work_end_idx, v_w_pred);
    digit_conf_v_work = event2digit(work_cleaned, work_start_idx, work_end_idx, v_w_conf);
    
    % this part the index is based on the ground truth for an accuracte evaluation
    % also, you can purly use the event index come from your event detection
    % code and the data length to do the fusion
    % it is not necessary to write with suce tedious steps :-)
    
    cook_start_idx = [start_cut, start_stir]'; clearvars start_cut start_stir
    cook_end_idx = [end_cut, end_stir,]'; clearvars end_cut end_stir
    clean_start_idx = [start_wipe, start_drawer]'; clearvars start_wipe start_drawer
    clean_end_idx = [end_wipe, end_drawer]'; clearvars end_wipe end_drawer
    % for cooking
    cooking_sub_array = find(vib_ext(:,1) == 4|vib_ext(:,1) == 5);
    cooking_sub_array = cooking_sub_array-length(v_w_pred);
    digit_pred_v_cook = event2digit(cook_cleaned, cook_start_idx, cook_end_idx, v_k_pred(cooking_sub_array));
    digit_conf_v_cook = event2digit(cook_cleaned, cook_start_idx, cook_end_idx, v_k_conf(cooking_sub_array));
    % for cleaning
    cleaning_sub_array = find(vib_ext(:,1) == 6|vib_ext(:,1) == 9);
    cleaning_sub_array = cleaning_sub_array-length(v_w_pred);
    digit_pred_v_clean = event2digit(clean_cleaned, clean_start_idx, clean_end_idx, v_k_pred(cleaning_sub_array));
    digit_conf_v_clean = event2digit(clean_cleaned, clean_start_idx, clean_end_idx, v_k_conf(cleaning_sub_array));
    % for vacuuming
    vac_sub_array = find(vib_ext(:,1) == 8);
    vac_sub_array = vac_sub_array-length(v_w_pred);
    digit_pred_v_vac = event2digit(vac_cleaned, start_vac, end_vac, v_k_pred(vac_sub_array));
    digit_conf_v_vac = event2digit(vac_cleaned, start_vac, end_vac, v_k_conf(vac_sub_array));
    
    % digit to slding window
    win_pred_v_work = labelExtraction(digit_pred_v_work, 1.5, 8);
    win_conf_v_work = labelExtraction(digit_conf_v_work, 1.5, 8);
    win_pred_v_cook = labelExtraction(digit_pred_v_cook, 1.5, 8);
    win_conf_v_cook = labelExtraction(digit_conf_v_cook, 1.5, 8);
    win_pred_v_clean = labelExtraction(digit_pred_v_clean, 1.5, 8);
    win_conf_v_clean = labelExtraction(digit_conf_v_clean, 1.5, 8);
    win_pred_v_vac = labelExtraction(digit_pred_v_vac, 1.5, 4);
    win_conf_v_vac = labelExtraction(digit_conf_v_vac, 1.5, 4);
    v_pred_win = [win_pred_v_work;win_pred_v_cook;win_pred_v_clean;win_pred_v_vac];
    v_conf_win = [win_conf_v_work;win_conf_v_cook;win_conf_v_clean;win_conf_v_vac];
    [fused_pred_win, fused_conf_win] = winBaseFusion(i_pred, i_conf, v_pred_win, v_conf_win);
    cm = confusionchart(ext(:,1),fused_pred_win,'Normalization','row-normalized');
    cm = cm.NormalizedValues;
    close all;
    acc_fused = trace(cm)/length(cm);
    disp(['Fused accuracy: ', num2str(acc_fused)])
    % IMU final result
    pseu_i_w = fused_pred_win(1:length(i_w_pred));
    pseu_i_w(pseu_i_w > 3) = 10; % otherwise it is not in the studying area
    pseu_i_k = fused_pred_win(1+length(i_w_pred):end);
    conf_i_w = fused_conf_win(1:length(i_w_pred));
    conf_i_k = fused_conf_win(1+length(i_w_pred):end);
    
    %% Transfer the window-based prediction back to event_based
    win_pseu_v_work = fused_pred_win(1:length(win_pred_v_work));
    win_conf_v_work = fused_conf_win(1:length(win_conf_v_work));
    win_pseu_v_cook = fused_pred_win(1+length(win_pred_v_work):length(win_pred_v_work)+length(win_pred_v_cook));
    win_conf_v_cook = fused_conf_win(1+length(win_conf_v_work):length(win_pred_v_work)+length(win_pred_v_cook));
    win_pseu_v_clean = fused_pred_win(1+length(win_pred_v_work)+length(win_pred_v_cook):length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean));
    win_conf_v_clean = fused_conf_win(1+length(win_pred_v_work)+length(win_pred_v_cook):length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean));
    win_pseu_v_vac = fused_pred_win(1+length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean):length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean)+length(win_pred_v_vac));
    win_conf_v_vac = fused_conf_win(1+length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean):length(win_pred_v_work)+length(win_pred_v_cook)+length(win_pred_v_clean)+length(win_pred_v_vac));
    % transfer them back one-by-one
    [event_pseu_v_work, event_conf_v_work] = win2event(work_cleaned, work_start_idx, work_end_idx, win_pseu_v_work, win_conf_v_work, 8, 1.5);
    [event_pseu_v_cook, event_conf_v_cook] = win2event(cook_cleaned, cook_start_idx, cook_end_idx, win_pseu_v_cook, win_conf_v_cook, 8, 1.5);
    [event_pseu_v_clean, event_conf_v_clean] = win2event(clean_cleaned, clean_start_idx, clean_end_idx, win_pseu_v_clean, win_conf_v_clean, 8, 1.5);
    [event_pseu_v_vac, event_conf_v_vac] = win2event(vac_cleaned, start_vac, end_vac, win_pseu_v_vac, win_conf_v_vac, 4, 1.5);
    event_pseu_v_work(event_pseu_v_work >3) = randi(3); %IMU has the fourth class, idle, we use uniform sampling to replace
    % combine for the vib study final result
    pseu_v_w = event_pseu_v_work;
    conf_v_w = event_conf_v_work;
    
    %start to reindex the living area data since the IMU ends with 9,10,8 and
    %vibration ends with 8,9
    original_class_6_length = length(find(vib_ext(:,1)<7))-length(find(vib_ext(:,1)<6));
    pseu_class_6 = event_pseu_v_clean(1:original_class_6_length);
    conf_class_6 = event_conf_v_clean(1:original_class_6_length);
    pseu_class_9 = event_pseu_v_clean(1+original_class_6_length:end);
    conf_class_9 = event_conf_v_clean(1+original_class_6_length:end);
    % combine for the vib living final result
    pseu_v_k = [event_pseu_v_cook;pseu_class_6;event_pseu_v_vac;pseu_class_9];
    conf_v_k = [event_conf_v_cook;conf_class_6;event_conf_v_vac;conf_class_9];
    
    %% Check the dimention consistency
    disp('Checking the dimention consistency:')
    if all(size(pseu_v_w) == size(v_w_pred)) && all(size(conf_v_w) == size(v_w_conf)) && all(size(pseu_v_k) == size(v_k_pred)) && all(size(conf_v_k) == size(v_k_conf))
        disp('Vib Checked!')
    else
        error('Vib has inconsistency!')
    end
    
    if all(size(pseu_i_w) == size(i_w_pred)) && all(size(conf_i_w) == size(i_w_conf)) && all(size(pseu_i_k) == size(i_k_pred)) && all(size(conf_i_k) == size(i_k_conf))
        disp('IMU Checked!')
    else
        error('IMU has inconsistency!')
    end
    
    if save_file
        clearvars -except pseu_v_w conf_v_w pseu_v_k conf_v_k pseu_i_w conf_i_w pseu_i_k conf_i_k
        save("../output/fused_result.mat")
        disp('Fused results saved!')
    end
else
    disp('Target Domain Result: ')
    [v_w_conf, v_w_pred] = max(tar_vib_conf_w,[],2);
    [v_k_conf, v_k_pred] = max(tar_vib_conf_k,[],2);
    v_k_pred = v_k_pred + 3; % living area statrs from 4;
    v_k_pred(v_k_pred>6) = v_k_pred(v_k_pred>6) + 1; % this data doesn't have class 7 event detected;
    
    v_pred = [v_w_pred; v_k_pred];
    cm_v = confusionchart(vib_ext(:,1),v_pred,'Normalization','row-normalized');
    cm_v = cm_v.NormalizedValues;
    close all;
    acc_v_tar = trace(cm_v)/length(cm_v); % Avarage activity rcognition accuracy
    disp(['Final vibration accuracy: ', num2str(acc_v_tar)])
    
    % Process the IMU data then
    [i_w_conf, i_w_pred] = max(tar_imu_conf_w,[],2);
    [i_k_conf, i_k_pred] = max(tar_imu_conf_k,[],2);
    i_k_pred = i_k_pred + 3;
    i_pred = [i_w_pred; i_k_pred];
    i_conf = [i_w_conf; i_k_conf];
    cm_i = confusionchart(ext(:,1),i_pred,'Normalization','row-normalized');
    cm_i = cm_i.NormalizedValues;
    close all;
    acc_i_tar = trace(cm_i)/length(cm_i);
    disp(['Final IMU accuracy: ', num2str(acc_i_tar)])

    % Fuse the result
    % event to digit
    work_start_idx = [start_type, start_mouse, start_write]'; clearvars start_write start_type start_mouse
    work_end_idx = [end_type, end_mouse, end_write]'; clearvars end_write end_type end_mouse
    digit_pred_v_work = event2digit(work_cleaned, work_start_idx, work_end_idx, v_w_pred);
    digit_conf_v_work = event2digit(work_cleaned, work_start_idx, work_end_idx, v_w_conf);
    
    % this part the index is based on the ground truth for an accuracte evaluation
    % also, you can purly use the event index come from your event detection
    % code and the data length to do the fusion
    % it is not necessary to write with suce tedious steps :-)
    
    cook_start_idx = [start_cut, start_stir]'; clearvars start_cut start_stir
    cook_end_idx = [end_cut, end_stir,]'; clearvars end_cut end_stir
    clean_start_idx = [start_wipe, start_drawer]'; clearvars start_wipe start_drawer
    clean_end_idx = [end_wipe, end_drawer]'; clearvars end_wipe end_drawer
    % for cooking
    cooking_sub_array = find(vib_ext(:,1) == 4|vib_ext(:,1) == 5);
    cooking_sub_array = cooking_sub_array-length(v_w_pred);
    digit_pred_v_cook = event2digit(cook_cleaned, cook_start_idx, cook_end_idx, v_k_pred(cooking_sub_array));
    digit_conf_v_cook = event2digit(cook_cleaned, cook_start_idx, cook_end_idx, v_k_conf(cooking_sub_array));
    % for cleaning
    cleaning_sub_array = find(vib_ext(:,1) == 6|vib_ext(:,1) == 9);
    cleaning_sub_array = cleaning_sub_array-length(v_w_pred);
    digit_pred_v_clean = event2digit(clean_cleaned, clean_start_idx, clean_end_idx, v_k_pred(cleaning_sub_array));
    digit_conf_v_clean = event2digit(clean_cleaned, clean_start_idx, clean_end_idx, v_k_conf(cleaning_sub_array));
    % for vacuuming
    vac_sub_array = find(vib_ext(:,1) == 8);
    vac_sub_array = vac_sub_array-length(v_w_pred);
    digit_pred_v_vac = event2digit(vac_cleaned, start_vac, end_vac, v_k_pred(vac_sub_array));
    digit_conf_v_vac = event2digit(vac_cleaned, start_vac, end_vac, v_k_conf(vac_sub_array));
    
    % digit to slding window
    win_pred_v_work = labelExtraction(digit_pred_v_work, 1.5, 10);
    win_conf_v_work = labelExtraction(digit_conf_v_work, 1.5, 10);
    win_pred_v_cook = labelExtraction(digit_pred_v_cook, 1.5, 10);
    win_conf_v_cook = labelExtraction(digit_conf_v_cook, 1.5, 10);
    win_pred_v_clean = labelExtraction(digit_pred_v_clean, 1.5, 10);
    win_conf_v_clean = labelExtraction(digit_conf_v_clean, 1.5, 10);
    win_pred_v_vac = labelExtraction(digit_pred_v_vac, 1.5, 3);
    win_conf_v_vac = labelExtraction(digit_conf_v_vac, 1.5, 3);
    v_pred_win = [win_pred_v_work;win_pred_v_cook;win_pred_v_clean;win_pred_v_vac];
    v_conf_win = [win_conf_v_work;win_conf_v_cook;win_conf_v_clean;win_conf_v_vac];
    [fused_pred_win, fused_conf_win] = winBaseFusion(i_pred, i_conf, v_pred_win, v_conf_win);
    cm = confusionchart(ext(:,1),fused_pred_win,'Normalization','row-normalized');
    cm = cm.NormalizedValues;
    close all;
    acc_fused = trace(cm)/length(cm);
    disp(['Fused accuracy: ', num2str(acc_fused)])
end