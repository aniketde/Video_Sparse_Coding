%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Classification using Non-negative Sparse Coding,
% Low-Rank and Sparse Decomposition

% Written by Aniket Deshmukh and Naveen Murthy
% Course Project for EECS 556


%Step 1: Download Caltech101 images and put it in folder called image\test. 
%Step 2: Download and install PROPACK
%Step 3: Download large_scale_svm package by Kai Yu, Aug. 2008
%Step 4: Download and put SIFT descriptor script in sift folder. 
%Step 5: Dowload sparse coding program from
%https://github.com/igkiou/sparse_linear_model/tree/master/sparse_coding
%and put it in sparse_coding folder
%Step 6: Download RPCA from
%https://github.com/posenhuang/singingvoiceseparationrpca/tree/master/inexact_alm_rpca
%and put it in folder inexact_alm_rpca.
%Step 7: Run main_556.m

% Original Reference: Zhang, Chunjie, Jing Liu, Qi Tian, Changsheng Xu,
% Hanqing Lu, and Songde Ma. "Image classification by non-negative sparse coding,
% low-rank and sparse decomposition." In Computer Vision and Pattern Recognition (CVPR),
% 2011 IEEE Conference on, pp. 1673-1680. IEEE, 2011.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear all,close all
addpath('large_scale_svm');
addpath('sift');
addpath(genpath('sparse_coding'));
addpath(genpath('exact_alm_rpca'));

FolderName = 'test';   % Name of database being used
ImageDir = ['image\' FolderName];
DataDir = ['data\' FolderName];

% Initializtion for SIFT extraction
skip_SIFT = true;              % if 'skip_SIFT' is true, load the saved SIFT features
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;

% Initialization for Non-negative sparse coding
skip_SC = true;
NumBases = 1024;
NumSamples = 10000; % no.of features to be sampled for the entire database
beta = 1e-5;
NumIters = 50;

% Initialization of SPM + Max-pooling
pyramid = [1 2 4];  % Levels in SPM
gamma = 0.15;
knn = 20;

% Classification using SVM
NumTrials = 15;
lambda = 0.1;
NumTrain = 10;   % Number of training images per class

%% SIFT Feature Extraction
if skip_SIFT,
    database = retrieve_database(DataDir);
else
    [database, lenStat] = CalculateSiftDescriptor(ImageDir, DataDir, gridSpacing, patchSize, maxImSize, nrml_threshold);
end;

%% Non-negative Sparse Coding
DictDir = ['dictionary/dict_' FolderName '_' num2str(NumBases) '.mat'];
if skip_SC
    fprintf('Loading Dictionary...\n')
    load(DictDir)
else
    fprintf('Sparse Coding...\n')
    X = random_sampling(database,NumSamples);
    [V, U, stat] = reg_sparse_coding(X, NumBases, eye(NumBases), beta, gamma, NumIters);
end

%% Spatial Pyramid Matching + Max-Pooling
fprintf('Spatial Pyramid Matching + Max Pooling...\n')
DimFeat = sum(NumBases*pyramid.^2);
NumFeat = length(database.path);

H = zeros(DimFeat, NumFeat);
H_label = zeros(NumFeat, 1);

for iter1 = 1:NumFeat
    if mod(iter1,20) == 0
        fprintf('.')
    end
    load(database.path{iter1});
    H(:, iter1) = SPM_max_pooling(feaSet, V, pyramid, gamma);
    H_label(iter1) = database.label(iter1);
end

%% Low Rank and Sparse Matrix Decomposition + Locality-constrained Linear Coding (LLC)
fprintf('Low Rank and Sparse Decomposition...\n')
tradeoff = [0.55];
Acc_tradeoff = zeros(1,numel(tradeoff));
Time = zeros(1,numel(tradeoff));

tic
[N, L] = exact_alm_rpca(H, tradeoff);
B_LLC = [L N];
Coeff = LLC_coding_appr(B_LLC',H',900);
Coeff = Coeff';

%% Classification using linear SVM
fprintf('Classification...\n')
[dimFea, nFea] = size(Coeff);
clabel = unique(H_label);
accuracy = zeros(NumTrials, 1);

fprintf('Trial: ');
for iter1 = 1:NumTrials,
    fprintf('%d, ', iter1);
    TrainIdx = [];
    TestIdx = [];
    
    for iter2 = 1:database.nclass,
        idx_label = find(H_label == clabel(iter2));
        num = length(idx_label);
        RandIdx = randperm(num);
        
        TrainIdx = [TrainIdx; idx_label(RandIdx(1:NumTrain))];
        TestIdx = [TestIdx; idx_label(RandIdx(NumTrain+1:end))];
    end;
    
    TrainFeat = [Coeff(:, TrainIdx)] ;
    TrainLabel = H_label(TrainIdx);
    
    TestFeat = Coeff(:, TestIdx);
    TestLabel = H_label(TestIdx);
    
    [w, b, class_name] = li2nsvm_multiclass_lbfgs(TrainFeat', TrainLabel', lambda);
    
    [C, Y] = li2nsvm_multiclass_fwd(TestFeat', w, b, class_name);
    
    acc = zeros(length(class_name), 1);
    
    for iter2 = 1 : length(class_name),
        c = class_name(iter2);
        idx = find(TestLabel == c);
        curr_pred_label = C(idx);
        curr_gnd_label = TestLabel(idx);
        acc(iter2) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end;
    
    accuracy(iter1) = mean(acc);
end;

fprintf('Mean accuracy: %f\n', mean(accuracy));
fprintf('Standard deviation: %f\n', std(accuracy));

