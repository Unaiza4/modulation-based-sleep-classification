%% === STFT spectrograms (EEGSNET-style: fs=64, 0–32 Hz, 76x60x3) ===

% 1) Configuration – EEGSNET-like
fs_raw         = 100;     % original sampling rate in your CSV 
fs_target      = 64;      % EEGSNET: resample C4-M1 to 64 Hz
epoch_duration = 30;      % seconds
samples_per_epoch_raw = fs_raw * epoch_duration;

channels     = {'C4-M1'};
input_folder = ''    % your file path
output_root  = ''    % <- new EEGSNET-style STFT output folder
FILE_PATTERN = 'S*_PSG_df_updated.csv';                        % process all

% STFT params (Hamming window with overlap, at 64 Hz)
% (EEGSNET text doesn't give exact numbers, but this is a reasonable choice)
win_len   = 128;          % 2 seconds at 64 Hz
win       = hamming(win_len);
noverlap  = 96;           % 75% overlap
nfft      = 256;          % power-of-2, gives 0–32 Hz with fs=64

% Image size per EEGSNET: 76 × 60 × 3
TARGET_H = 76; 
TARGET_W = 60;
nColors = 256;

r1 = round(nColors * 0.6);
r2 = nColors - r1;

g1 = round(nColors * 0.7);
g2 = nColors - g1;

b1 = round(nColors * 0.6);
b2 = nColors - b1;

cmap = zeros(nColors,3);

cmap(:,1) = [linspace(0, 1, r1),    linspace(1, 1, r2)]';     % Red
cmap(:,2) = [linspace(0, 1, g1),    linspace(1, 0.5, g2)]';   % Green
cmap(:,3) = [linspace(1, 0.3, b1),  linspace(0.3, 0, b2)]';   % Blue

% Band-pass filter 0.3–32 Hz at fs_target=64 (EEG band of interest)
[b_bp, a_bp] = butter(4, [0.3 31.9]/(fs_target/2), 'bandpass');

% 2) Files
files = dir(fullfile(input_folder, FILE_PATTERN));

for fileIdx = 1:numel(files)
    fileName   = files(fileIdx).name;
    subject_id = extractBefore(fileName, '_PSG');
    if isempty(subject_id), subject_id = erase(fileName, '.csv'); end

    fprintf('\n⏳ Processing (EEGSNET STFT) %s...\n', subject_id);

    T = readtable(fullfile(input_folder, fileName), 'VariableNamingRule','preserve');

    if ~ismember('Sleep_Stage', T.Properties.VariableNames)
        warning('Sleep_Stage column missing in %s, skipping...', fileName);
        continue;
    end

    labels        = string(T.Sleep_Stage);
    total_samples = height(T);

    % 30-s epochs based on ORIGINAL sampling rate (fs_raw = 100 Hz)
    num_epochs = floor(total_samples / samples_per_epoch_raw);
    if num_epochs < 1
        warning('Not enough samples in %s, skipping...', fileName);
        continue;
    end

    % Trim to whole number of epochs and get epoch-wise labels
    labels       = labels(1:num_epochs * samples_per_epoch_raw);
    epoch_labels = labels(1:samples_per_epoch_raw:end);

    for ch = 1:numel(channels)
        channel = channels{ch};
        if ~ismember(channel, T.Properties.VariableNames)
            warning('Channel %s missing in %s. Skipping...', channel, fileName);
            continue;
        end

        sig_raw = T.(channel);
        sig_raw = sig_raw(1:num_epochs * samples_per_epoch_raw);
        X_raw   = reshape(sig_raw, samples_per_epoch_raw, num_epochs);   % [time_raw x epoch]

        for i = 1:num_epochs
            stage = epoch_labels(i);

            % Skip P, empty, or NaN stages as in your original
            if stage == "P" || stage == "" || strcmpi(stage,"NaN")
                continue;
            end

            % 3) Take one 30-s epoch at original fs, then resample to 64 Hz
            x_raw = double(X_raw(:, i));          % 30 s at 100 Hz (3000 samples)

            % Resample to EEGSNET fs_target = 64 Hz
            x_resamp = resample(x_raw, fs_target, fs_raw);  % ~1920 samples for 30 s

            % Band-pass 0.3–32 Hz at 64 Hz
            x_filt = filtfilt(b_bp, a_bp, x_resamp);

            % Z-score normalize per epoch
            x_norm = (x_filt - mean(x_filt)) ./ (std(x_filt) + eps);

            % 4) STFT at 64 Hz (Hamming window + overlap)
            [S, F, ~] = spectrogram(x_norm, win, noverlap, nfft, fs_target);
            % With fs_target=64 and nfft=256, F automatically spans 0..32 Hz.

            % 5) Convert to power image and scale
            % 5) Convert to power image and scale
            P = 10 * log10(abs(S) + eps);

            % Optional contrast normalization
            lo = prctile(P(:), 5);
            hi = prctile(P(:), 95);
            P  = max(lo, min(hi, P));

            % --- ✔ ADD THIS (smooths ONLY output image, not the STFT) ---
            P = imgaussfilt(P, 1.2);      % smooth style like example-2
            P = medfilt2(P, [3 3]);       % small blocky smoothing

            % Map to 0–255 and resize
            P_norm = mat2gray(P);
                             % scale 0–1
            I8     = uint8(255 * P_norm);                      % uint8 image
            Pimg   = imresize(I8, [TARGET_H TARGET_W], 'bilinear');

            % Map to RGB (jet colormap), final size 76×60×3
            RGB = ind2rgb(Pimg, cmap);

            % 6) Save with EEGSNET-style shape (76x60x3), organized by subject/stage/channel
            out_dir  = fullfile(output_root, subject_id, stage, channel);
            if ~exist(out_dir,'dir'), mkdir(out_dir); end

            out_file = fullfile(out_dir, sprintf('%s_%s_epoch%03d.png', subject_id, channel, i));
            imwrite(RGB, out_file);
        end
    end

    fprintf('✅ Done (EEGSNET STFT) %s\n', subject_id);
end

fprintf('\n🎯 EEGSNET-style STFT output root: %s\n', output_root);
fprintf('   Pattern: %s\n', fullfile(input_folder, FILE_PATTERN));
