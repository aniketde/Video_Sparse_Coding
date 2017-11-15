function [OutClass] = final_classify(I,V,H,H_label,B_LLC,Coeff,database)
NumTrials = 1;
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;

pyramid = [1 2 4];  % Levels in SPM
gamma = 0.15;
knn = 20;
lambda = 0.1;

[feaTest] = ExtractSIFT(I, gridSpacing, patchSize, maxImSize, nrml_threshold);
H_test = SPM_max_pooling(feaTest, V, pyramid, gamma);

tradeoff = 0.55;

[N_test, L_test] = exact_alm_rpca(H_test, tradeoff);
Coeff_test = LLC_coding_appr(B_LLC',H_test',900);
Coeff_test = Coeff_test';

%% Classification using linear SVM
fprintf('Classification...\n')
[dimFea, nFea] = size(Coeff);

TrainFeat = Coeff;
TrainLabel = H_label;
TestFeat = Coeff_test;
[w, b, class_name] = li2nsvm_multiclass_lbfgs(TrainFeat', TrainLabel', lambda);
[C, Y] = li2nsvm_multiclass_fwd(TestFeat', w, b, class_name);
OutClass = C;
disp(database.cname{C});

end
