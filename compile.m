cd Patch_mex/
mex HOG_mex.cpp
mex kmeans_getDistUpdate_mex.cpp
cd ..

cd TRW_S
mex TRWS_mex.cpp
cd ..

cd libsvm/matlab/
make;
cd ../../
