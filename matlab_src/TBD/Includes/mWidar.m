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
        end
    end
end