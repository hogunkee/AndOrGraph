function test
mex TRWS_mex.cpp
load('try.mat','E','f1','f2','params');
x=TRWS_mex(E,f1,f2,params);
disp(x);
