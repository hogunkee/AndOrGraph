function correspond=getFlipCorrespond(ARGSet,IsFlip)
len=length(ARGSet);
if(IsFlip)
    correspond=ones(len,1).*(-1);
    for i=1:len
        if(ARGSet(i).flip==-1)
            continue;
        end
        for j=(i+1):length(ARGSet)
            if(strcmp(ARGSet(i).arg.ARGInfo.name,ARGSet(j).arg.ARGInfo.name))
                correspond(i)=j;
                correspond(j)=i;
                ARGSet(i).flip=-1;
                ARGSet(j).flip=-1;
                break;
            end
        end
    end
else
    correspond=ones(len,1).*(-1);
end
