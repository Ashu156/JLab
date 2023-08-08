% This script searches a user-defined directory for .h264 files and converts them to .mp4 files.
% Author: Ashutosh Shukla
% Written and tested in MATLAB 2023a
% Needs ffmpeg pre-installed on the system and  and the path to ffmpeg.exe file provided in line 14

%%
tic;
clear;
close all;
clc;

%% Specify paths and extensions

ffmpegPath = 'C:\Users\ashutoshshukla\ffmpeg\bin\ffmpeg.exe';  % Path to FFmpeg executable
from_ext = '*.h264'; % Original video file extension
to_ext = '.mp4';     % Desired video file extension

masterFolderPath = uigetdir; % Select master folder containing subfolders
subfolders = dir(fullfile(masterFolderPath, '*')); % Get list of subfolders

%% Convert .h264 to .mp4 in all subfolders

for folderIdx = 1:length(subfolders)
    if subfolders(folderIdx).isdir && ~strcmp(subfolders(folderIdx).name, '.') && ~strcmp(subfolders(folderIdx).name, '..')
        currentSubfolder = fullfile(masterFolderPath, subfolders(folderIdx).name);
        filePattern = fullfile(currentSubfolder, from_ext);
        dirInfo = dir(filePattern);
        h264Files = {dirInfo.name};
        
        fprintf('Processing files in subfolder: %s\n', currentSubfolder);
        
        for fileIdx = 1:length(h264Files)
            inputFile = fullfile(currentSubfolder, h264Files{fileIdx});
            [~, name, ~] = fileparts(inputFile);
            outputFile = fullfile(currentSubfolder, [name, to_ext]);

            % Check if the .mp4 file already exists
            if exist(outputFile, 'file')
                fprintf('Skipping conversion: %s (Output file already exists)\n', h264Files{fileIdx});
                continue; % Skip to the next iteration
            end

            fprintf('Converting: %s --> %s\n', inputFile, outputFile);

            command = sprintf('%s -i "%s" -c:v copy "%s"', ffmpegPath, inputFile, outputFile);

            [status, result] = system(command);

            if status == 0
                disp('Video conversion successful.');
            else
                disp('Video conversion failed.');
                disp(result);
            end
        end
    end
end

toc;

%% end of script
