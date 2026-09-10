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
        S
        M
        G

        %%% blurring parameter
        sigma

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

            parse(p, varargin{:})

            obj.debug = p.Results.Debug;
            obj.normalize = p.Results.Normalize;
            obj.blur = p.Results.Blur;
            obj.sigma = p.Results.Sigma;


            %%% Filepaths
            obj.RECOVERY_FILEPATH = "supplemental/recovery.mat";
            obj.SAMPLING_FILEPATH = "supplemental/sampling.mat";

            obj.S = zeros(obj.npx);
            
            ds = load(obj.RECOVERY_FILEPATH);
            obj.G = ds.G;

            ds = load(obj.SAMPLING_FILEPATH);
            obj.M = ds.M;

        end
        
        %% Signal Generation

        %%% Generate single image
        function signal = generate_mWidar_image(obj, pos, varargin)
            
            % parse

            %%5 TODO: Add functionality here to generate image for multiple objects in scene

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

            if meters
                obj.debug_print("SELECTED METERS")
                signal = obj.generate_mWidar_image_meters(pos);
            end

            if pixels
                obj.debug_print("SELECTED PIXELS")
                signal = obj.generate_mWidar_image_pixels(pos);
            end

        end
    end
    methods(Hidden)
        %%% Generate image when pos is in meters
        function signal = generate_mWidar_image_meters(obj, pos)

            px = pos(1);
            py = pos(2);

            %%5 Get grid cell corresponding to m pos 
            if obj.checkbound_x(px) && obj.checkbound_y(py)
                Gx = find(px <= obj.xgrid,1,'first');
                Gy = find(py <= obj.ygrid,1,'first');

                %%% Check it exists within obj bounds
                if obj.checkbound_idx(Gx) && obj.checkbound_idx(Gy)
                    obj.S(Gy,Gx) = 1;
                end
            end

            % TODO: Gracefully handle the case where the obj is out of scene
            if all(obj.S == 0)
                obj.debug_print("OBJECT IS OUT OF SCENE, RETURNING EMPTY SIGNAL")
                signal = [];
                return;
            end

            % mWidar forward model
            signal_flat = obj.S';
            signal_flat = signal_flat(:);
            signal_flat = obj.M * signal_flat;
            signal_flat = obj.G' * signal_flat;
            raw_signal = reshape(signal_flat, obj.npx,obj.npx);
            blurred = obj.blur_signal(raw_signal);
            signal = obj.normalize_signal(blurred);

        end



        function signal = generate_mWidar_image_pixels(obj, pos)
            
            Gx = pos(1);
            Gy = pos(2);

                %%% Check it exists within scene bounds
            if obj.checkbound_idx(Gx) && obj.checkbound_idx(Gy)
                obj.S(Gy,Gx) = 1;
            end
            

            % TODO: Gracefully handle the case where the obj is out of scene
            if all(obj.S == 0)
                obj.debug_print("OBJECT IS OUT OF SCENE, RETURNING EMPTY SIGNAL")
                signal = [];
                return;
            end

            % mWidar forward model
            signal_flat = obj.S';
            signal_flat = signal_flat(:);
            signal_flat = obj.M * signal_flat;
            signal_flat = obj.G' * signal_flat;
            raw_signal = reshape(signal_flat, obj.npx,obj.npx);
            blurred = obj.blur_signal(raw_signal);
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

        function b = checkbound_x(obj,x)
            b = x > -2 && x < 2;
        end

        function b = checkbound_y(obj,y)
            b = y > 0 && y < 4;
        end

        function b = checkbound_idx(obj,i)
            b = i > 0 && i < 128;
        end
    end

end