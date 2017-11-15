function X = random_sampling(database,NumSamp)

NumPerImage = round(NumSamp/database.imnum);
NumSamp = NumPerImage * database.imnum;

load(database.path{1});
DimFeatures = size(feaSet.feaArr, 1);

X = zeros(DimFeatures, NumSamp);
count = 0;

for iter1 = 1:database.imnum
    load(database.path{iter1});
    NumFeatures = size(feaSet.feaArr, 2);
    RandId = randperm(NumFeatures);
    X(:, count+1:count+NumPerImage) = feaSet.feaArr(:, RandId(1:NumPerImage));
    count = count + NumPerImage;
end
end