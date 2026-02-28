clear; clc;

fullPath = fullfile(pwd, "outputs_grid", "full_dump.json"); % <- 改路径
raw = jsondecode(fileread(fullPath));

names = fieldnames(raw);
N = numel(names);

% 预分配
exp_name = strings(N,1);
n_mels   = zeros(N,1);
n_mfcc   = zeros(N,1);
width    = zeros(N,1);
classes  = zeros(N,1);

best_val_acc  = nan(N,1);
best_val_loss = nan(N,1);

float_acc = nan(N,1);
float_loss = nan(N,1);
float_ms  = nan(N,1);
float_kb  = nan(N,1);

int8_acc = nan(N,1);
int8_loss = nan(N,1);
int8_ms  = nan(N,1);
int8_kb  = nan(N,1); % 注意：你目前这里可能全是 0（量化统计未生效）
% 标注 acc top3
[~, idxAcc] = maxk(float_acc, 3);
for k = idxAcc'
    text(float_ms(k), float_kb(k), float_acc(k), exp_name(k), 'FontSize', 8);
end


for i = 1:N
    k = names{i};
    S = raw.(k);

    exp_name(i) = string(S.experiment.name);
    n_mels(i)   = S.experiment.audio.n_mels;
    n_mfcc(i)   = S.experiment.audio.n_mfcc;
    width(i)    = S.experiment.model.width_mult;
    classes(i)  = S.experiment.model.num_classes;

    best_val_acc(i)  = S.train.best_val_acc;
    best_val_loss(i) = S.train.best_val_loss;

    float_acc(i) = S.float.test.acc;
    float_loss(i)= S.float.test.loss;
    float_ms(i)  = S.float.infer_ms;
    float_kb(i)  = S.float.weights_bytes / 1024.0;

    if isfield(S, "int8_ptq")
        int8_acc(i) = S.int8_ptq.test.acc;
        int8_loss(i)= S.int8_ptq.test.loss;
        int8_ms(i)  = S.int8_ptq.infer_ms;
        int8_kb(i)  = S.int8_ptq.weights_bytes / 1024.0;
    end
end

T = table(exp_name,n_mels,n_mfcc,width,classes,best_val_acc,best_val_loss,...
          float_acc,float_loss,float_ms,float_kb,...
          int8_acc,int8_loss,int8_ms,int8_kb);

disp(T(:,["exp_name","n_mels","n_mfcc","width","float_acc","float_ms","float_kb","int8_acc","int8_ms","int8_kb"]));

%% 1) 3D tradeoff (float)
figure;
scatter3(float_ms, float_kb, float_acc, 70, width, 'filled');
xlabel("Float latency (ms)"); ylabel("Float size (KB)"); zlabel("Float test acc");
title("Float tradeoff: acc-latency-size"); grid on; colorbar;
view(40, 20);

%% 2) Heatmap: float_acc (rows=feat, cols=width)
feat = strcat(string(n_mels), "_", string(n_mfcc));
featVals = unique(feat, 'stable');
widthVals = unique(width, 'stable');

Macc = nan(numel(featVals), numel(widthVals));
Mms  = nan(numel(featVals), numel(widthVals));
Mkb  = nan(numel(featVals), numel(widthVals));

for r = 1:numel(featVals)
    for c = 1:numel(widthVals)
        idx = (feat==featVals(r)) & (width==widthVals(c));
        if any(idx)
            Macc(r,c) = float_acc(find(idx,1));
            Mms(r,c)  = float_ms(find(idx,1));
            Mkb(r,c)  = float_kb(find(idx,1));
        end
    end
end

figure; imagesc(Macc); colorbar; grid on;
xticks(1:numel(widthVals)); xticklabels(string(widthVals));
yticks(1:numel(featVals)); yticklabels(featVals);
xlabel("width_mult"); ylabel("n_mels_n_mfcc");
title("Heatmap: float test acc");

figure; imagesc(Mms); colorbar; grid on;
xticks(1:numel(widthVals)); xticklabels(string(widthVals));
yticks(1:numel(featVals)); yticklabels(featVals);
xlabel("width_mult"); ylabel("n_mels_n_mfcc");
title("Heatmap: float latency (ms)");

figure; imagesc(Mkb); colorbar; grid on;
xticks(1:numel(widthVals)); xticklabels(string(widthVals));
yticks(1:numel(featVals)); yticklabels(featVals);
xlabel("width_mult"); ylabel("n_mels_n_mfcc");
title("Heatmap: float size (KB)");

%% 3) best_val_acc vs test_acc
figure;
scatter(best_val_acc, float_acc, 70, width, 'filled');
xlabel("best val acc"); ylabel("test acc (float)");
title("Generalization check: val vs test"); grid on; colorbar;

%% 4) 固定 width=1.0 的趋势（按 n_mels 降序）
w0 = 1.0;
idx = (width==w0);
S = T(idx,:);
[~,ord] = sort(S.n_mels, 'descend');
S = S(ord,:);

figure;
yyaxis left; plot(S.n_mels, S.float_acc, '-o'); ylabel("float test acc");
yyaxis right; plot(S.n_mels, S.float_ms, '-s'); ylabel("float ms");
xlabel("n_mels"); title("Trend at width=1.0"); grid on;
