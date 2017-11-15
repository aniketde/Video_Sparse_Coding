# Video_Sparse_Coding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Image Classification using Non-negative Sparse Coding, Low-Rank and Sparse Decomposition

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

% Original Reference: Zhang, Chunjie, Jing Liu, Qi Tian, Changsheng Xu, Hanqing Lu, and Songde Ma. "Image classification by non-negative sparse coding, low-rank and sparse decomposition." In Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on, pp. 1673-1680. IEEE, 2011.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


In the presence of a vast amount of digital video data, the protection of intellectual property of the creator is of utmost importance. Video copy detection, which is  used  to  find  copyright  infringements,  is  also  useful  in  video  information  retrieval.  We propose a novel technique for video copy detection using an image classification framework and sparse coding.  The underlying image classification framework is based on non-negative sparse coding, low-rank and sparse matrix decomposition techniques along with Spatial Pyramid Matching (LR-Sc+SPM). SIFT  features  from  each  image  are  encoded  using  non-negative  sparse  coding. Using Spatial Pyramid Matching + Max pooling, we capture the spatial relations between the sparse codes. Low-rank and sparse matrix decomposition is then used to exploit correlations and dissimilarities between images of the same class.  Extending this to video copy detection, we create a framework where scene change detection is performed using edge maps.   The resulting scenes are divided into classes and, using the same image classification framework, a multi-class linear SVM is trained.  We evaluate our proposed algorithm against two state-of-the-art techniques for video copy detection with accuracy and computational time being the metrics
