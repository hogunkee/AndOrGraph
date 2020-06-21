#include <stdio.h>
#include "MRFEnergy.h"

// Example: minimizing an energy function with Potts terms.
// See type*.h files for other types of terms.


#include <stdio.h>
#include "MRFEnergy.h"

void testGeneral()
{
	MRFEnergy<TypeGeneral>* mrf;
	MRFEnergy<TypeGeneral>::NodeId* nodes;
	MRFEnergy<TypeGeneral>::Options options;
	TypeGeneral::REAL energy, lowerBound;

	const int nodeNum = 3; // number of nodes
	TypeGeneral::REAL Dx[3];
	TypeGeneral::REAL Dy[3];
	TypeGeneral::REAL Dz[2];
	TypeGeneral::REAL V[3*2];
	int x, y, z;

	mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
	nodes = new MRFEnergy<TypeGeneral>::NodeId[nodeNum];

	// construct energy
	Dx[0] = 0; Dx[1] = 1; Dx[2] = 2;
	nodes[0] = mrf->AddNode(TypeGeneral::LocalSize(3), TypeGeneral::NodeData(Dx));
	Dy[0] = 3; Dy[1] = 4; Dy[2] = 5;
	nodes[1] = mrf->AddNode(TypeGeneral::LocalSize(3), TypeGeneral::NodeData(Dy));
	mrf->AddEdge(nodes[0], nodes[1], TypeGeneral::EdgeData(TypeGeneral::POTTS, 6));
	Dz[0] = 7; Dz[1] = 8;
	nodes[2] = mrf->AddNode(TypeGeneral::LocalSize(2), TypeGeneral::NodeData(Dz));
	for (y=0; y<3; y++)
	{
		for (z=0; z<2; z++)
		{
			V[y + z*3] = y*y + z;
		}
	}
	mrf->AddEdge(nodes[1], nodes[2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, V));

	// Function below is optional - it may help if, for example, nodes are added in a random order
	// mrf->SetAutomaticOrdering();

	/////////////////////// TRW-S algorithm //////////////////////
	options.m_iterMax = 30; // maximum number of iterations
	mrf->Minimize_TRW_S(options, lowerBound, energy);

	// read solution
	x = mrf->GetSolution(nodes[0]);
	y = mrf->GetSolution(nodes[1]);

	printf("Solution: %d %d\n", x, y);

	//////////////////////// BP algorithm ////////////////////////
	mrf->ZeroMessages(); // in general not necessary - it may be faster to start 
	                     // with messages computed in previous iterations

	options.m_iterMax = 30; // maximum number of iterations
	mrf->Minimize_BP(options, energy);

	// read solution
	x = mrf->GetSolution(nodes[0]);
	y = mrf->GetSolution(nodes[1]);

	printf("Solution: %d %d\n", x, y);

	// done
	delete nodes;
	delete mrf;
}

void main()
{
	testGeneral();
}
