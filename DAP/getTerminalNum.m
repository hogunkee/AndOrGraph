function TerminalNum=getTerminalNum(model)
TerminalNum=zeros(model.nodeNum,1);
for i=1:model.nodeNum
    TerminalNum(i)=size(model.NodeInfo(i).unaryF(1).f,2);
end
