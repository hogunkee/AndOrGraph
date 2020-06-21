#include "mex.h"
#include "stdio.h"
#include "string.h"
#include <windows.h>
#include <assert.h>
#include <stdlib.h>
#include "instances.h"
#include "MRFEnergy.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu/mxGPUArray.h"

#include "cuda.h"


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
    bool IsUseGPU;
    if(nrhs==5)
        IsUseGPU=bool(*((double*)mxGetPr(prhs[4])));
    else if(nrhs==4)
        IsUseGPU=true;
    else
        mexErrMsgTxt("\nErrors in input.\n");
	MRFEnergy<TypeGeneral>* mrf;
	MRFEnergy<TypeGeneral>::NodeId* nodes;
	MRFEnergy<TypeGeneral>::Options options;
	TypeGeneral::REAL energy, lowerBound;

    double* EdgeTerminals=((double*)mxGetPr(prhs[0]));
    int EdgeNum,LabelNum,nodeNum;
    EdgeNum=mxGetN(prhs[0]);
    TypeGeneral::REAL* f1=((double*)mxGetPr(prhs[1]));
    LabelNum=mxGetM(prhs[1]);
    nodeNum=mxGetN(prhs[1]);
    TypeGeneral::REAL* f2=((double*)mxGetPr(prhs[2]));
    TypeGeneral::REAL* op=((double*)mxGetPr(prhs[3]));
    
    int i;
    double tmp=0;
    for(i=0;i<EdgeNum*LabelNum*LabelNum;i++){
        tmp=min(tmp,f2[i]);
    }
    if(tmp<0)
        mexErrMsgTxt("\nErrors: having negative pairwise elements.\n");
    mrf=new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
	nodes=new MRFEnergy<TypeGeneral>::NodeId[nodeNum];
    mexPrintf("add nodes\n");
	
    for(i=0;i<nodeNum;i++){
        nodes[i]=mrf->AddNode(TypeGeneral::LocalSize(LabelNum), TypeGeneral::NodeData(&f1[LabelNum*i]));
    }
    mexPrintf("add edges\n");
    for(i=0;i<EdgeNum;i++){
        mrf->AddEdge(nodes[int(EdgeTerminals[i*2])],nodes[int(EdgeTerminals[i*2+1])],TypeGeneral::EdgeData(TypeGeneral::GENERAL,&f2[LabelNum*LabelNum*i]));
    }
    mexPrintf("set ordering\n");
	// Function below is optional - it may help if, for example, nodes are added in a random order
	mrf->SetAutomaticOrdering();
	/////////////////////// TRW-S algorithm //////////////////////
	options.m_iterMax=int(op[1]); // maximum number of iterations
    options.m_eps=op[0];
    mexPrintf("energy minimization\n");
	mrf->Minimize_TRW_S(IsUseGPU,options,lowerBound,energy);
    mexPrintf("trws done\n");
	// read solution
    double* x;
    plhs[0]=mxCreateDoubleMatrix(nodeNum,1,mxREAL);
	x=mxGetPr(plhs[0]);
    for(i=0;i<nodeNum;i++){
        x[i]=mrf->GetSolution(nodes[i]);
    }
	delete nodes;
	delete mrf;
}
