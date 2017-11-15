function [database] = retrieve_database(DataDir)

%=========================================================================
% This function retrieves locations of the SIFT database.

% Input: DataDir = Path of the database
% Output: database = Structure containing the required details
%          database.path = path for each image file
%          database.label = label for each image file
%          database.className = Name of each class
%=========================================================================

database.nclass = 0;
database.imnum = 0;
database.path = {};
database.label = [];
database.cname = {};

fprintf('Retrieving SIFT features...');
categories = dir(DataDir);   % Subfolders in the database
for iter1 = 1:length(categories)
    ClassName = categories(iter1).name;
    %     if ~strcmp(ClassName, '.') & ~strcmp(ClassName, '..'),
    database.nclass = database.nclass + 1;
    database.cname{database.nclass} = ClassName;
    
    Images = dir([DataDir '\' ClassName '\' '*.mat']);
    i_num = length(Images);
    database.label = [database.label ones(i_num,1)*database.nclass];
    database.imnum = database.imnum + i_num;
    
    for iter2 = 1:i_num
        i_path = [DataDir '\' ClassName '\' Images(iter2).name];
        database.path = [database.path i_path];
    end
    %     end
end

fprintf('Done!!\n')
end