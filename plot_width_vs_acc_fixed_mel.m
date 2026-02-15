clear; clc;
summaryPath = fullfile(pwd, "outputs_grid", "summary.json");
raw = jsondecode(fileread(summaryPath));
T = struct2table(raw);

featPairs = unique([T.n_mels, T.n_mfcc], "rows", "stable");
widthVals = sort(unique(T.width), "descend");

figure; hold on; grid on;
xlabel("width\_mult"); ylabel("test acc");
title("Width effect: float (solid) vs int8 (dashed)");

leg = strings(size(featPairs,1)*2, 1);
k = 0;

for i = 1:size(featPairs,1)
    nm = featPairs(i,1); nf = featPairs(i,2);
    S = T(T.n_mels==nm & T.n_mfcc==nf, :);
    [~, ord] = sort(S.width, "descend"); S = S(ord,:);

    yF = nan(size(widthVals));
    yQ = nan(size(widthVals));
    for j = 1:numel(widthVals)
        idx = S.width == widthVals(j);
        if any(idx)
            r = find(idx,1);
            yF(j) = S.float_acc(r);
            yQ(j) = S.int8_acc(r);
        end
    end

    plot(widthVals, yF, "-o", "LineWidth", 1.5);
    plot(widthVals, yQ, "--s", "LineWidth", 1.5);

    k = k + 1; leg(k) = sprintf("float mel=%d mfcc=%d", nm, nf);
    k = k + 1; leg(k) = sprintf("int8  mel=%d mfcc=%d", nm, nf);
end

legend(leg, "Location", "bestoutside");
