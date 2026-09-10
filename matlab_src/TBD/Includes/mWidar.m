%%% mWidar.m 
%%% Class with no methods, holds all simulation parameters. parent class to all other classes

classdef mWidar

    properties
        %%% Scene setup
        Lscene % Physican length of the scene in meters
        npx % # of pixels
        xgrid
        ygrid
        pxgrid
        pygrid
        pxygrid
        dx
        dy
        max_x
        min_x
        max_y
        min_y
    end
    methods
        function obj = mWidar()
            %%% Scene setup
            obj.Lscene = 4;
            obj.npx = 128;
            obj.xgrid = linspace(-2,2,obj.npx);
            obj.ygrid = linspace(0,4,obj.npx);
            [obj.pxgrid, obj.pygrid] = meshgrid(obj.xgrid, obj.ygrid);
            obj.pxygrid = [obj.pxgrid(:), obj.pygrid(:)];
            obj.dx = obj.xgrid(2) - obj.xgrid(1);
            obj.dy = obj.ygrid(2) - obj.ygrid(1);        
            obj.max_x = 2;
            obj.min_x = -2;
            obj.max_y = 4;
            obj.min_y = 0;
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

        function b = checkbound(obj, pos)
            b = obj.checkbound_x(pos(1)) && obj.checkbound_y(pos(2));
        end

    end
end