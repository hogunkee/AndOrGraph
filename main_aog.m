function main_aog(Name_batch)

addpath('./DAP_produce_SIVAL');
addpath('./TRW_S');
addpath('./DAP');
addpath('./libsvm/matlab');

load(sprintf('./mat/model_%s.mat',Name_batch),'model_set');
load(sprintf('./mat/ARGs_%s.mat',Name_batch),'ARGSet');
load(sprintf('./mat/ARGs_neg_%s.mat',Name_batch),'ARGSet_neg');

Para=getGraphMiningParameters(); % graph mining without learning attribute weights (sometimes, it has more stable learning process)
%Para=getGraphMiningParameters_withLearningWeights(); % graph mining without learning attribute weights

try
    p=gcp;
    Para.ThreadNum=p.NumWorkers; % start workers for parallel computing
catch
    Para.ThreadNum=0; % do not apply parallel computing
end
if(Para.IsFlip)
    ARGSet=Setting_for_Flip(ARGSet,Para); % produce the flipped ARGs
    Para.MaxARGNumPerIteration=Para.MaxARGNumPerIteration/2;
    Para.StartMaxARGNumPerIteration=Para.StartMaxARGNumPerIteration/2;
end

TargetDir='./mat/Models/';
DAP_main(model_set,Name_batch,ARGSet,Para,TargetDir,ARGSet_neg); % graph mining procedure

%% visualize graph matching performance
load(sprintf('%s%s_model.mat',TargetDir,Name_batch),'record','time'); % load the AoG model
model_set=record(end).model_set; % the final AoG model
GoodARGList=record(end).GoodARGList;
ARGSet=ARGSet(GoodARGList.list); % the target ARGs (we use the top-ranked positive ARGs as the target ARGs)
match=getMatch(model_set,ARGSet,0,Para); % the graph matching process
for j=1:length(ARGSet)
    show_MatchingResult_SIVAL(model_set,ARGSet(j).arg.ARGInfo.name,match(:,j)',Para,model_set.link); % show matching results
end
