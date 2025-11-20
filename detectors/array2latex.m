function [latex_str] = array2latex(array, precision, labels, caption)
    % array2latex - Convert a numeric array to a LaTeX tabular format string with booktabs.
    % Syntax: latex_str = array2latex(array)
    % Inputs:
    %    array - A numeric array to be converted.
    %    precision - (Optional) Can be either:
    %                - Scalar: Number of decimal places for all columns (default: 2)
    %                - Cell array: Format strings for each column (e.g., {'%d', '%.3f', '%.2f'})
    %    labels - (Optional) Cell array of labels. Can be:
    %             - Single dimension: column labels only
    %             - Multi-dimensional: labels(1,:) = column labels, labels(2,:) = row labels
    %    caption - (Optional) Caption for the table (default: none).
    % Outputs:
    %    latex_str - A string containing the LaTeX representation of the array.
    % Examples:
    %    % Using scalar precision (all columns get same format):
    %    latex_str = array2latex(array, 3, labels, 'My Table Caption');
    %    
    %    % Using custom formats per column:
    %    formats = {'%d', '%.3f', '%.2f'};
    %    latex_str = array2latex(array, formats, labels, 'My Table Caption');

    if nargin < 2
        precision = 2; % Default precision
    end

    if nargin < 3
        labels = {}; % Default no labels
    end

    if nargin < 4
        caption = ''; % Default no caption
    end

    % Determine if precision is a scalar (use as decimal places) or cell array (use as formats)
    if iscell(precision)
        % Cell array provided - use as custom formats
        formats = precision;
        use_custom_formats = true;
        [num_rows, num_cols] = size(array);
        
        % Validate formats array size
        if length(formats) ~= num_cols
            error('Number of format strings (%d) must match number of columns (%d)', length(formats), num_cols);
        end
    else
        % Scalar provided - use as precision for all columns
        use_custom_formats = false;
        if ~isscalar(precision) || ~isnumeric(precision)
            error('Precision must be a scalar number or a cell array of format strings');
        end
    end

    % Start building the LaTeX table
    latex_str = [sprintf('\\begin{table}[htbp]\n\\centering\n\\caption{%s}\n', caption)];
    latex_str = [latex_str sprintf('%% Requires \\usepackage{booktabs}\n')];
    latex_str = [latex_str sprintf('\\begin{tabular}{')];
    [num_rows, num_cols] = size(array);
    
    % Check if labels is multi-dimensional
    col_labels = {};
    row_labels = {};
    has_row_labels = false;
    
    if ~isempty(labels)
        if iscell(labels) && size(labels, 1) >= 2
            % Multi-dimensional: first row is column labels, second row is row labels
            col_labels = labels(1, :);
            if size(labels, 1) >= 2 && size(labels, 2) >= num_rows
                row_labels = labels(2, 1:num_rows);
                has_row_labels = true;
            end
        else
            % Single dimension: assume column labels only
            col_labels = labels;
        end
    end
    
    % Add extra column for row labels if they exist
    if has_row_labels
        latex_str = [latex_str 'c|'];
    end
    
    for col = 1:num_cols
        latex_str = [latex_str 'c'];
    end

    latex_str = [latex_str sprintf('}\n\\toprule\n')];
    
    % Add column labels if provided
    if ~isempty(col_labels)
        header_str = '';
        if has_row_labels
            header_str = ' & '; % Empty cell for row label column
        end
        header_str = [header_str strjoin(col_labels, ' & ')];
        latex_str = [latex_str sprintf('%s \\\\\n\\midrule\n', header_str)];
    end

    % Add array data
    if use_custom_formats
        % Use custom formats from cell array
        format_str = ''; % Not used in this case
    else
        % Use scalar precision for all columns
        format_str = ['%.' num2str(precision) 'f'];
    end

    for row = 1:num_rows
        row_data = [];
        
        % Add row label if it exists
        if has_row_labels && row <= length(row_labels)
            row_data = [row_data row_labels{row} ' & '];
        end
        
        % Add the numeric data
        for col = 1:num_cols
            if use_custom_formats
                formatted_value = sprintf(formats{col}, array(row, col));
            else
                formatted_value = sprintf(format_str, array(row, col));
            end
            row_data = [row_data formatted_value];
            if col < num_cols
                row_data = [row_data ' & '];
            end
        end
        latex_str = [latex_str sprintf('%s \\\\\n', row_data)];
    end

    latex_str = [latex_str sprintf('\\bottomrule\n\\end{tabular}\n\\end{table}\n')];
end
