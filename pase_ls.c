#include <stdlib.h>
#include <math.h>
#include "pase_ls.h"

PASE_Int PASE_LinearSolverCreate(PASE_Solver* linear_solver, MPI_Comm comm)
{
   HYPRE_Solver* pcg_solver = linear_solver;	
   /* Create PCG solver for PASE */
   PASE_ParCSRPCGCreate(comm, pcg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_PCGSetMaxIter(*pcg_solver, 1000); /* max iterations */
   HYPRE_PCGSetTol(*pcg_solver, 1e-8); /* conv. tolerance */
   HYPRE_PCGSetTwoNorm(*pcg_solver, 1); /* use the two norm as the stopping criteria */
   HYPRE_PCGSetPrintLevel(*pcg_solver, 0); /* print solve info */
   HYPRE_PCGSetLogging(*pcg_solver, 1); /* needed to get run info later */

   return 0;
}  


PASE_Int PASE_LinearSolverSetup(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
PASE_ParVector F, PASE_ParVector U)
{
	hypre_PCGSetup(linear_solver, A, F, U);
	return 0;
}

PASE_Int PASE_LinearSolverSolve(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
PASE_ParVector F, PASE_ParVector U)
{
	hypre_PCGSolve(linear_solver, A, F, U);	
	return 0;	
}
PASE_Int PASE_LinearSolverDestroy(PASE_Solver linear_solver, PASE_Solver precond)
{
	PASE_ParCSRPCGDestroy(linear_solver);
	HYPRE_BoomerAMGDestroy(precond);
	return 0;
}
