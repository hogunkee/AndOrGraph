function [nodelist_vale,nodelist_order]=getIntersectNodeList(nodelist,list_withstandardorder)
[~,list_1,list_2]=intersect(nodelist,list_withstandardorder);
[~,tmp]=sort(list_2);
nodelist_order=list_1(tmp);
nodelist_vale=nodelist(nodelist_order);
