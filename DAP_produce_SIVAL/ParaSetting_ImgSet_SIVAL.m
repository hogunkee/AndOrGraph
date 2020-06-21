%% These parameters are set for middle-level feature extraction

function [HOG,Kmeans,Learning,ImgRoot]=ParaSetting_ImgSet_SIVAL()

ImgRoot='./DAP_produce_SIVAL/ImgSet/';

HOG.ScaleLargest=0.60;
HOG.ScaleLevelNum=7;
HOG.ScaleDecrease=0.8;
HOG.SampleWindowStep=2;
HOG.VoteOriNum=8;
HOG.CellNum=8;
HOG.MinCellScale=3;
HOG.GausianFilterScale=10;
HOG.MinAvgGradientRate=0.7;
HOG.MinAvgGradient=4;

Kmeans.KRate=0.2; % in the first iteration, get (KRate*PatchNumber) clusters
Kmeans.MinSize=5;
Kmeans.MaxIterNum=8;

Learning.ThreadNum=max(matlabpool('size'),1);
Learning.ThreadBuffer_ImgNum=50;
Learning.TopK=5;    %10 for each cluster, collect top-K samples to train a SVM classifier
Learning.CluCollectK=0.1; % for each cluster, collect top-(max(CluCollectK*PositiveSampleNum,TopK)) samples to train a SVM classifier in the final iteration
Learning.MinScore=-0; %0 the mininum score for the SVM classifier to collect samples
Learning.MinCollectScore=-0.5; %-0.5; the mininum score for the SVM classifier to collect samples in the final iteration

Learning.CollectNum=150;
Learning.MaxCollectNumPerModel=3;

Learning.TopCluNum=1000; %250
Learning.NegSampleNum=5000;
Learning.IterNum=5;
