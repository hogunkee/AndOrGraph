function x=TRWSProcess(V,E,f1,f2,Para)
%eps=0.00000000000000001;
eps=0.000000000001;
x=TRWS_mex(E,f1,f2,[eps,Para.MaxMRFIterNum],0);
x=double(x')+1;
