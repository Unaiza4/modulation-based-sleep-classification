
%% ===== EEGSNet-aligned CWT Export (DREAMT, C4-M1, 30 s, 76x60) =====
clear; clc;

% ---------------- Paths & settings ----------------
DATA_DIR   = 
OUT_ROOT   = 
CHANNEL    = 'C4-M1';

FS_IN      = 100;      % DREAMT raw
FS_OUT     = 64;       % EEGSNet uses 64 Hz
EPOCH_SEC  = 30;
SAMPLES_IN  = FS_IN  * EPOCH_SEC;   
SAMPLES_OUT = FS_OUT * EPOCH_SEC;   

% Keep these AASM stages only
KEEP_STAGES = {'W','N1','N2','N3','R'};   % Map 'REM' -> 'R'

% -------------- Discover DREAMT CSVs --------------
files = dir(fullfile(DATA_DIR, 'S*_PSG_df_updated*.csv'));
if isempty(files)
    files = dir(fullfile(DATA_DIR, 'S*_PSG_df*.csv'));
end
assert(~isempty(files), 'No DREAMT CSV files found in %s', DATA_DIR);

% -------------- Prebuild CWT filterbank -----------
fb = cwtfilterbank('SignalLength', SAMPLES_OUT, ...
                   'SamplingFrequency', FS_OUT, ...
                   'Wavelet', 'amor', ...
                   'VoicesPerOctave', 12);

fprintf('Found %d files. Output -> %s\n', numel(files), OUT_ROOT);

for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    [~, fname] = fileparts(fpath);

    % Parse subject ID like 'S002'
    subjTok = regexp(fname, '(S\d{3})', 'tokens', 'once');
    if isempty(subjTok)
        warning('Skip (no subject id): %s', fname); continue;
    end
    subj = subjTok{1};

    % -------- Load table, verify columns --------
    T = readtable(fpath, 'VariableNamingRule','preserve');
    if ~ismember(CHANNEL, T.Properties.VariableNames)
        warning('Channel "%s" missing in %s. Skipping.', CHANNEL, fname); continue;
    end
    if ~ismember('Sleep_Stage', T.Properties.VariableNames)
        warning('"Sleep_Stage" missing in %s. Skipping.', fname); continue;
    end

    x100 = T.(CHANNEL);
    lbl  = string(strtrim(T.Sleep_Stage));

    % Normalize label text (REM -> R); drop missing
    lbl(upper(lbl)=="REM") = "R";
    valid = ~ismissing(lbl);
    x100  = x100(valid);
    lbl   = lbl(valid);

    % -------- Truncate to full 30s epochs (100 Hz) --------
    N100 = numel(x100);
    nEpoch_in = floor(N100 / SAMPLES_IN);
    if nEpoch_in == 0
        warning('%s: <1 epoch at 100 Hz. Skipping.', subj); continue;
    end
    x100 = x100(1 : nEpoch_in*SAMPLES_IN);
    lbl  = lbl(1  : nEpoch_in*SAMPLES_IN);

    % -------- Resample to 64 Hz (match EEGSNet) --------
    % Use polyphase resampling (p=64, q=100)
    x64 = resample(double(x100), FS_OUT, FS_IN);

    % Map labels to resampled timeline via nearest sample
    % Build epoch labels from original 30s epochs (as-is)
    epoch_labels = strings(nEpoch_in,1);
    for e = 1:nEpoch_in
        sIn = (e-1)*SAMPLES_IN + 1;
        eIn = e*SAMPLES_IN;
        cIn = round((sIn + eIn)/2);    % center sample (equals epoch label)
        epoch_labels(e) = lbl(cIn);
    end

    % Now truncate resampled signal to full 30s epochs at 64 Hz
    N64 = numel(x64);
    nEpoch_out = floor(N64 / SAMPLES_OUT);
    % Keep the min to avoid off-by-one length differences
    nEpoch = min(nEpoch_in, nEpoch_out);
    if nEpoch == 0
        warning('%s: <1 epoch at 64 Hz. Skipping.', subj); continue;
    end
    x64 = x64(1 : nEpoch*SAMPLES_OUT);
    epoch_labels = epoch_labels(1:nEpoch);

    % -------- Generate & save per-epoch CWT images --------
    for e = 1:nEpoch
        stage = char(epoch_labels(e));
        if ~ismember(stage, KEEP_STAGES), continue; end

        % Output folder: <OUT_ROOT>\<Subject>\<Stage>\C4-M1\
        outDir = fullfile(OUT_ROOT, subj, stage, CHANNEL);
        if ~exist(outDir, 'dir'); mkdir(outDir); end

        % Epoch signal at 64 Hz
        sIdx = (e-1)*SAMPLES_OUT + 1;
        eIdx = e*SAMPLES_OUT;
        sig  = x64(sIdx:eIdx);

        % CWT scalogram -> power
        cfs = fb.wt(sig);
        P   = abs(cfs).^2;

        % dB-like contrast + normalize to [0,1]
        Pdb = 10*log10(P + eps);
        Pdb = Pdb - min(Pdb(:));
        if max(Pdb(:)) > 0, Pdb = Pdb ./ max(Pdb(:)); end

        % Resize to EEGSNet input size 76x60 (H x W)
        I   = imresize(Pdb, [76 60], 'bilinear');

        % Convert to RGB (parula colormap) -> 76x60x3
        cmap = parula(256);
        I8   = uint8(round(I * 255));
        IRGB = ind2rgb(I8, cmap);

        % Save PNG
        outName = sprintf('%s_%s_%s_e%04d.png', subj, strrep(CHANNEL,'-','_'), stage, e);
        imwrite(IRGB, fullfile(outDir, outName));
    end

    fprintf('%s: saved %d epochs for %s\n', subj, nEpoch, CHANNEL);
end

fprintf('✅ Done. CWT spectrograms saved under: %s\n', OUT_ROOT);
