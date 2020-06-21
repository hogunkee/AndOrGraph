function list=getFlipIndex(energy,ARGSet,IsFlip)
if(IsFlip)
    list=[];
    for i=1:length(ARGSet)
        if(ARGSet(i).flip==-1)
            continue;
        end
        for j=(i+1):length(ARGSet)
            if(strcmp(ARGSet(i).arg.ARGInfo.name,ARGSet(j).arg.ARGInfo.name))
                if(energy(i)>energy(j))
                    list=[list,j];
                else
                    list=[list,i];
                end
                ARGSet(i).flip=-1;
                ARGSet(j).flip=-1;
                break;
            end
        end
        if(ARGSet(i).flip~=-1)
            list=[list,i];
            ARGSet(i).flip=-1;
        end
    end
else
    list=1:length(energy);
end
