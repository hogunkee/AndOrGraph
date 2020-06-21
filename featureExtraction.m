function featureExtraction(Name_batch)

addpath('./DAP_produce_SIVAL');
addpath('./libsvm/matlab');
addpath('./Patch_mex');

LearnDiscriminativePatches(Name_batch); % produce middle-level patches
ARGSet_produce(Name_batch); % produce the positive and negative ARGs
