function totry
load('try_b.mat','E','f1','f2','Para');
Para.MaxMRFIterNum=10000;
eps=0.00000000000000001;
% clc;mex TRWS_mex.cu
% tic;x=TRWS_mex(E,f1,f2,[eps,Para.MaxMRFIterNum],1);disp(x');toc;
tic;x_0=TRWS_mex(E,f1,f2,[eps,Para.MaxMRFIterNum],0);disp(x_0');toc;
% tic;V=0:(size(f1,2)-1);[x,LB,phi,M_bw]=alg_trws_mex(int32(V),int32(E),-f1,-f2,[eps,ceil(Para.MaxMRFIterNum/10)],[]);disp(x);toc;
