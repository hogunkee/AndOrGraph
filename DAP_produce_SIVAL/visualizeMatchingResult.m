function visualizeMatchingResult(Name_batch,TargetDir,Para,ARGSet)
load(sprintf('%s%s_model.mat',TargetDir,Name_batch),'record','time');
model_set=record(end).model_set; % the final AoG mined
GoodARGList=record(end).GoodARGList;
ARGSet=ARGSet(GoodARGList.list); % the target ARGs
match=getMatch(model_set,ARGSet,0,Para); % the graph matching process
for j=1:length(ARGSet)
    show_MatchingResult_SIVAL(model_set,ARGSet(j).arg.ARGInfo.name,match(:,j)',Para,model_set.link); % show matching results
end
