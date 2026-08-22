% site specific script to load site information

% enter orientation of sonics.  Sonic order: tables sorted alphabetically followed by columns sorted in descending order
info.sonicOrientation = [36]; %UU1

% enter manufacturer of SAT.  1 for Campbell, 0 for RMYoung.  RMYoung v =
% Campbell u!; 3 for Gill WindmasterPro
info.sonicManufact = [2];  

% enter orientation of tower relative to sonic head
info.tower = 210; % FM 

% tower elevation
info.siteElevation = 1980; % (m)

% site slope angle
info.angle = 8.2; % mean of 20 m radius buffer

% enter expected table names.  Missing tables will be filled with NaNs to create consistency 
% when multiple output files are concatenated with getData.m
info.tableNames = {'FMDOL_10Hz'}; % modified by Diane

% enter table scan frequencies corresponding to tableNames
info.tableScanFrequency = [10];  %[Hz]

% enter number of columns in each .csv table.  Note that the number of columns in the output structure will 
% be 3 less than the number in the .csv file.  This is because the 4 column date vector is replaced with a Matlab's 
% single-column serial time.  Also, note that View Pro frequently cuts of column 1 (the year!) of the .csv file. 
info.tableNumberOfColumns = [21]; % modified by Diane (21 = +Temp_6.35, RH_6.35)