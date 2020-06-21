function Para=getGraphMiningParameters()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Focus on these parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para.PenaltyThreshold=0.90; % parameter \tau in the article
Para.Lambda=1.0; % parameter \lambda in the article
Para.IsFlip=true; % whether to consider the left-right symmetry of objects

Para.MaxARGNumPerIteration=0.30; % collect top-MaxARGNumPerIteration ARGs in the final iteration
Para.StartMaxARGNumPerIteration=0.15; % collect top-MaxARGNumPerIteration ARGs in the first iteration

Para.LearningItems.IsWeightTraining=true; % whether to train the attribute weights w_i^u and w_j^p. For some categories, choosing "false" may lead to a more stable learning process.
Para.MaxIterationNum=5; % iteration number for graph mining
Para.IterationNum_NoStructureModification=1; % number of addtional iterations for attribute estimation without modifying the structure
Para.MaxMRFIterNum=500; % in a certain graph matching process, the TRW-S iteration number to solve the MRF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Para.ThreadNum=matlabpool('size');

Para.LearningItems.IsStructureModification=true;
Para.LearningItems.IsTrainingPenalty=true;
Para.LearningItems.IsLinkLearning=false;
Para.LearningItems.IsWeightTraining_.IsTrainAttributeWeight=true;
Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri=false;
Para.LearningItems.IsWeightTraining_.IsProtectLastPairwiseWFromTraining=false;
Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri=false;
Para.MatchRateCost=-0;
Para.LinkNumRate=1.0;
Para.MaxTerminalNum=3;
Para.GoodMatchRate=0.5;
Para.ReDivideTerminalsInterval=1;
Para.AddNodeAlpha=1.0;
Para.PenaltyTrainingBias=2.0;
% "u_none" in the paper are set as
% U_pos+(U_neg-U_pos)*Para.PenaltyTrainingBias, where U_pos and U_neg are
% the average unary energy in the positive ARGs and the average unary
% energy in the negative ARGs, repspectively
% "p_none" in the paper are set as
% P_pos+(P_neg-P_pos)*Para.PenaltyTrainingBias

Para.AW.MaxStep=0.5;
Para.AW.Roof=0.95;
Para.AW.Gradient=8;
