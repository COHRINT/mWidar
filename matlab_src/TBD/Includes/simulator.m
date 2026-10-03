%%% Simulator.m
%%% Hold all methods for simulating mWidar signal given object pos

classdef simulator < mWidar

    properties
        
        %%% flags
        debug
        normalize
        blur

        %%% Filepaths
        RECOVERY_FILEPATH % G -> Recovery matrix
        SAMPLING_FILEPATH; % M -> sampling matrix
        

        %%% Forward model matrices
        M
        G

        %%% blurring parameter
        sigma

        %%% Object count
        ct

        %%% Noise varaince. Noise is enabled if var > 0
        var   % is noise is enabled, we will use this variance

    end

    methods
        %%% Simulator constructor
        function obj = simulator(varargin)

            % Parse optional inputs
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'Normalize', true, @islogical);
            addParameter(p, 'Blur', true, @islogical);
            addParameter(p, 'Sigma', 2);
            addParameter(p,'Objects', 1)
            addParameter(p,'Var', 0);

            parse(p, varargin{:})

            obj.debug = p.Results.Debug;
            obj.normalize = p.Results.Normalize;
            obj.blur = p.Results.Blur;
            obj.sigma = p.Results.Sigma;
            obj.ct = p.Results.Objects;
            obj.var = p.Results.Var;


            %%% Filepaths
            obj.RECOVERY_FILEPATH = "supplemental/recovery.mat";
            obj.SAMPLING_FILEPATH = "supplemental/sampling.mat";
            
            ds = load(obj.RECOVERY_FILEPATH);
            obj.G = ds.G;

            ds = load(obj.SAMPLING_FILEPATH);
            obj.M = ds.M;

        end
        
        %% Signal Generation

        %%% Generate single image
        %%% [signal, SNR] = sim.generate_mWidar_image(pos, 'meters', true)
        %%% SNR is the peak SNR of this frame in dB (see get_SNR); it is
        %%% measured on the noiseless signal, so callers can label a frame with
        %%% the SNR it was actually generated at.
        function [signal, SNR] = generate_mWidar_image(obj, pos, varargin)
            
            % parse
            p = inputParser;
            addParameter(p,'meters', false, @islogical)
            addParameter(p,'pixels', false, @islogical)
            parse(p, varargin{:})

            meters = p.Results.meters; % Is the given position in meters
            pixels = p.Results.pixels; % Is the given position in pixels
            
            % Throw error if both are true
            assert(~(meters && pixels))

            % Throw error is neither are true
            assert(meters || pixels)

            signal = zeros(obj.npx);
            SNR = NaN;

            if meters
                obj.debug_print("SELECTED METERS")
                [signal, SNR] = obj.generate_mWidar_image_meters(pos);
            end

            if pixels
                obj.debug_print("SELECTED PIXELS")
                [signal, SNR] = obj.generate_mWidar_image_pixels(pos);
            end

        end
    end
    methods(Hidden)
        %%% Generate image when pos is in meters
        function [signal, SNR] = generate_mWidar_image_meters(obj, pos)
            
            S = zeros(obj.npx);
            
            
            for i = 1:obj.ct
                if isempty(pos{i}), continue; end
                px = pos{i}(1);
                py = pos{i}(2);
                
                if isnan(px) || isnan(py)
                    continue;
                end

                %%5 Get grid cell corresponding to m pos 
                if obj.checkbound_x(px) && obj.checkbound_y(py)
                    Gx = find(px <= obj.xgrid,1,'first');
                    Gy = find(py <= obj.ygrid,1,'first');

                    %%% Check it exists within obj bounds
                    if obj.checkbound_idx(Gx) && obj.checkbound_idx(Gy)
                        S(Gy,Gx) = 1;
                    end
                end

            end

            % No object contributed to this frame (none present, or out of
            % scene): start from a blank canvas rather than bailing out, so
            % noise still gets added below.
            if all(S == 0)
                obj.debug_print("NO OBJECT IN SCENE, USING BLANK SIGNAL")
                blurred = zeros(obj.npx);
            else
                % mWidar forward model
                signal_flat = S';
                signal_flat = signal_flat(:);
                signal_flat = obj.M * signal_flat;
                signal_flat = obj.G' * signal_flat;
                raw_signal = reshape(signal_flat, obj.npx,obj.npx)';
                blurred = obj.blur_signal(raw_signal);
            end

            % get signal SNR
            SNR = obj.get_SNR(blurred);

            %%% Add noise if var > 0, even when no object is present
            if obj.var > 0
                obj.debug_print('Adding Noise')
                for i = 1:obj.npx
                    for j = 1:obj.npx
                        % add noise. Half gaussian with varaince of var
                        n = abs(sqrt(obj.var)*randn());
                        blurred(i,j) = blurred(i,j) + n;
                    end
                end
            end

            signal = obj.normalize_signal(blurred);

        end



        function [signal, SNR] = generate_mWidar_image_pixels(obj, pos)

            S = zeros(obj.npx);

            for i = 1:obj.ct
                if isempty(pos{i}), continue; end

                Gx = pos{i}(1);
                Gy = pos{i}(2);

                if isnan(Gx) || isnan(Gy)
                    continue;
                end

                    %%% Check it exists within scene bounds
                if obj.checkbound_idx(Gx) && obj.checkbound_idx(Gy)
                    S(Gy,Gx) = 1;
                end
            end

            % No object contributed to this frame (none present, or out of
            % scene): start from a blank canvas rather than bailing out, so
            % noise still gets added below.
            if all(S == 0)
                obj.debug_print("NO OBJECT IN SCENE, USING BLANK SIGNAL")
                blurred = zeros(obj.npx);
            else
                % mWidar forward model
                signal_flat = S';
                signal_flat = signal_flat(:);
                signal_flat = obj.M * signal_flat;
                signal_flat = obj.G' * signal_flat;
                raw_signal = reshape(signal_flat, obj.npx,obj.npx)';
                blurred = obj.blur_signal(raw_signal);
            end

            % get signal SNR
            SNR = obj.get_SNR(blurred);

            %%% Add noise if var > 0, even when no object is present
            if obj.var > 0
                obj.debug_print('Adding Noise')
                for i = 1:obj.npx
                    for j = 1:obj.npx
                        % add noise. Half gaussian with varaince of var
                        n = abs(sqrt(obj.var)*randn());
                        blurred(i,j) = blurred(i,j) + n;
                    end
                end
            end

            signal = obj.normalize_signal(blurred);
        end


        %% Helper Functions

        %%% Blur signal if enabled, ow just return raw signal
        function blurred = blur_signal(obj,raw)
            if obj.blur
                blurred = imgaussfilt(raw,obj.sigma);
            else
                blurred = raw;
            end
        end

        %%% Normalize signal if enabled, ow just return raw signal
        function normalized = normalize_signal(obj,raw)
            if obj.normalize
                if max(raw(:)) == min(raw(:))
                    obj.debug_print("MIN AND MAX OF UNNORMALIZED SIGNAL ==")
                    normalized = raw;
                    return
                end
                normalized = (raw - min(raw(:))) / (max(raw(:)) - min(raw(:)));
            else
                normalized = raw;
            end
        end

        %%% If debug is on, will print str, ow it will do nothing. Will add debug and simulator flag on its own
        function [] = debug_print(obj,str)
            if obj.debug
                str = "[DEBUG][SIMULATOR]" + str + "\n";
                fprintf(str)
            end
        end
        
        %%% Peak SNR of a frame in dB.
        %%% NOTE: Signal here is before any noise is added
        %%% The noise is a half gaussian, so its power is var*(1 - 2/pi).
        %%% The degenerate cases are reported rather than hidden, so a plot can
        %%% leave them as gaps instead of drawing a bogus number:
        %%%   var == 0 (noise off)   -> +Inf with signal, NaN on a blank frame
        %%%   no target in the scene -> -Inf (peak is zero)
        function SNR = get_SNR(obj, signal)
            peak = max(signal(:));

            if obj.var <= 0
                if peak > 0
                    SNR = Inf; % noiseless
                else
                    SNR = NaN; % no signal and no noise, nothing to report
                end
                return
            end

            SNR_peak = peak^2 / (obj.var * (1 - 2/pi));
            SNR = 10 * log10(SNR_peak); % -Inf when peak == 0
        end


    end

end