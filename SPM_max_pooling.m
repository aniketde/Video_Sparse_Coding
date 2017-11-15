function H = SPM_max_pooling(feaSet, V, pyramid, gamma)

NumCodewords = size(V, 2);
nFea = size(feaSet.feaArr, 2);
ImgWidth = feaSet.width;
ImgHeight = feaSet.height;
bin_ID = zeros(nFea, 1);

SparseCodes = zeros(NumCodewords, nFea);

beta = 1e-4;
A = V'*V + 2*beta*eye(NumCodewords);
Q = -V'*feaSet.feaArr;

for iter1 = 1:nFea,
    SparseCodes(:, iter1) = L1QP_FeatureSign_yang(gamma, A, Q(:, iter1));
end

SparseCodes = abs(SparseCodes);


Levels = length(pyramid); % number of levels
nBins = pyramid.^2; % no. of bins per level
tBins = sum(nBins); % total spatial bins
H = zeros(NumCodewords, tBins);
index = 0;

for iter1 = 1:Levels,
    BlockW = ImgWidth / pyramid(iter1);
    BlockH = ImgHeight / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / BlockW);
    yBin = ceil(feaSet.y / BlockH);
    bin_ID = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins(iter1),     
        index = index + 1;
        bin_ID_tic = find(bin_ID == iter2);
        if isempty(bin_ID_tic),
            continue;
        end      
        H(:, index) = max(SparseCodes(:, bin_ID_tic), [], 2);
    end
end

H = H(:);
H = H./sqrt(sum(H.^2));