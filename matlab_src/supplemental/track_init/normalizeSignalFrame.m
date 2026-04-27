function signal_normalized = normalizeSignalFrame(avgSignal, cfg)
% NORMALIZESIGNALFRAME  Blur, asinh-compress, and unit-normalize a frame.
%
% Lifted from probObjCt.m local function so estimateDetectionCount can
% be called from any script with track_init on the path.

    blurred = imgaussfilt(avgSignal, cfg.blur_sigma);
    blurred(1:cfg.crop_rows, :) = NaN;
    signal_scaled = asinh(blurred);
    if (max(signal_scaled(:)) - min(signal_scaled(:))) ~= 0
        signal_normalized = (signal_scaled - min(signal_scaled(:))) ./ ...
            (max(signal_scaled(:)) - min(signal_scaled(:)));
    else
        signal_normalized = signal_scaled;
    end
end
