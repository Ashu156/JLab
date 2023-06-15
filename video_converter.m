% This script searches a user-defined directory for .h264 files and converts them to .mp4 files.
% Author: Ashutosh Shukla
% Written and tested in MATLAB 2023a

%%
tic;
clear; 
close all; 
clc;

%%

ffmpegPath = 'C:\Users\ashutoshshukla\ffmpeg\bin\ffmpeg.exe';  % Specify the path to the FFmpeg executable (if not in system PATH)

from_ext = '*.h264';
to_ext = '.mp4';

folderPath = uigetdir; % Specify the path to the folder containing .h264 files
filePattern = fullfile(folderPath, from_ext);  % Create the file pattern to match .h264 files
dirInfo = dir(filePattern);  % Get the directory information for the matching files

h264Files = {dirInfo.name};  % Extract the filenames from the directory information

disp(h264Files);  % Display the list of .h264 files


%% Convert .h264 to .mp4

for i = 1:length(dirInfo) % Iterate over all the .h264 files found in the directory

   % Full path to the input file
    inputFile = fullfile(folderPath, dirInfo(i).name);  

    % Extract the file name and extension from the input file
    [filepath, name, ~] = fileparts(inputFile);

    % Generate the output file name with the desired extension
    outputFile = fullfile(filepath, [name, to_ext]);

    
    % Display the conversion of input filename to output filename
    fprintf('Converting: %s --> %s\n', inputFile, outputFile);

    % Convert and save the file
    command = sprintf('%s -i "%s" -c:v copy "%s"', ffmpegPath, inputFile, outputFile);

    % Status of conversion
    [status, result] = system(command);


    % Displays a message based on whether the file got successfully converted
    if status == 0
    disp('Video conversion successful.');
    
    else
    disp('Video conversion failed.');
    disp(result);
    end
    
    
end

%% end of script

