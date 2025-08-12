% Default plotting settings for mWidar project
set(0, 'DefaultFigureColormap', parula(64))

% LaTeX interpreter settings for publication-quality plots
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

% Figure and axes settings
set(groot, 'defaultFigureColor', [0.94, 0.94, 0.94]); % Light gray background
set(groot, 'defaultAxesColor', [0.94, 0.94, 0.94]); % Light gray axes background
set(groot, 'defaultFigurePosition', [100, 100, 800, 600]);
set(groot, 'defaultFigurePaperType', 'usletter');
set(groot, 'defaultFigurePaperPositionMode', 'auto');

% Axes settings for better readability
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultAxesBox', 'on');
set(groot, 'defaultAxesGridAlpha', 0.3);
set(groot, 'defaultAxesMinorGridAlpha', 0.1);

% Line and marker settings
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultLineMarkerSize', 8);

% Text settings
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultTextFontName', 'Times');

% Legend settings
set(groot, 'defaultLegendBox', 'off');
set(groot, 'defaultLegendLocation', 'best');
set(groot, 'defaultLegendFontSize', 11);

% Colorbar settings
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

% Print and export settings for high-quality figures
set(groot, 'defaultFigureInvertHardcopy', 'off');
set(groot, 'defaultFigureRenderer', 'painters');

