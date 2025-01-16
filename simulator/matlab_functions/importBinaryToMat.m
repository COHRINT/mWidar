%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% importBinaryToMat.m
%
% Anthony La Barca
%
% Function to import binary files into MATLAB matricies for use in the
% mWidar simulator.
%
% INPUTS:
%   filename: string, the name of the binary file to import
%   rows: int, the number of rows in the matrix
%   cols: int, the number of columns in the matrix
%
% OUTPUTS:
%   M: matrix, the matrix of the binary file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M] = importBinaryToMat(filename, rows, cols)
    % Open the file
    fileID = fopen(filename);

    % Read
    M = fread(fileID, 'double');

    % Close the file
    fclose(fileID);

    % Reshape the matrix
    M = reshape(M, rows, cols);

end


