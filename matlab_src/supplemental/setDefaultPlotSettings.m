function setDefaultPlotSettings()
% SETDEFAULTPLOTSETTINGS Configure default MATLAB plotting settings
%   This function sets up consistent plotting defaults for all figures
%   including LaTeX interpreters, line styles, colors, and font sizes.
%
%   Usage:
%       setDefaultPlotSettings();
%
%   Author: Generated for mWidar project
%   Date: 2025-07-24

%% Plotting settings

% LaTeX interpreter for text - comprehensive coverage
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
set(0, 'DefaultAxesLabelFontSizeMultiplier', 1.0);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.1);

% These are the key missing ones for title(), xlabel(), ylabel()
try
    set(0, 'DefaultAxesTitleInterpreter', 'latex');
    set(0, 'DefaultAxesXLabelInterpreter', 'latex');
    set(0, 'DefaultAxesYLabelInterpreter', 'latex');
    set(0, 'DefaultAxesZLabelInterpreter', 'latex');
catch
    % For older MATLAB versions, these properties might not exist
    fprintf('Warning: Some LaTeX interpreter defaults not supported in this MATLAB version.\n');
    fprintf('You may need to specify ''Interpreter'', ''latex'' manually for titles and labels.\n');
end

% Default figure properties
set(0, 'DefaultFigureColor', 'w'); % White background
set(0, 'DefaultAxesColor', 'none'); % Transparent axes background
set(0, 'DefaultAxesBox', 'on'); % Box around axes
set(0, 'DefaultLineLineWidth', 2); % Default line width
set(0, 'DefaultLineMarkerSize', 6); % Default marker size
set(0, 'DefaultAxesGridLineStyle', '--'); % Dashed grid lines
set(0, 'DefaultAxesGridAlpha', 0.5); % Grid transparency
set(0, 'DefaultAxesMinorGridAlpha', 0.3); % Minor grid transparency
set(0, 'DefaultAxesMinorGridLineStyle', ':'); % Dotted minor grid lines

% Text Size - Increased for better visibility
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultAxesLineWidth', 1.2);
set(0, 'DefaultLegendFontSize', 12);
set(0, 'DefaultColorbarFontSize', 12);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.1); % Smaller title font

fprintf('Default plotting settings applied successfully.\n');

end
