function TypeIdx=getNodeType(modelType,argNodeType)
TypeIdx=-1;
for i=1:length(argNodeType)
    if(strcmp(modelType,argNodeType(i).type))
        TypeIdx=i;
        break;
    end
end
