
#include <stdlib.h>
#include <math.h>
#include "pase_es.h"

PASE_Int PASE_EigenSolverCreate(PASE_Solver* eigen_solver, 
      mv_InterfaceInterpreter* ii, HYPRE_MatvecFunctions* mv, 
      PASE_Solver* linear_solver, PASE_Solver* precond, MPI_Comm comm)
{ 
   HYPRE_Solver* lobpcg_solver;
   lobpcg_solver = eigen_solver;

   int maxIterations = 1000; /* maximum number of iterations */
   int pcgMode = 1;         /* use rhs as initial guess for inner pcg iterations */
   int verbosity = 0;       /* print iterations info */
   double tol = 1.e-8;     /* absolute tolerance (all eigenvalues) */

   PASE_ParCSRSetupInterpreter(ii);
   PASE_ParCSRSetupMatvec(mv);
   /* 创建结束后，是否可以释放interpreter和matvec_fn 
      弄清楚Create干了什么，或者这里重新写一个
      PASE_LOBPCGCreate，使得interpreter和matvec_fn在这里不显示 */    
   HYPRE_LOBPCGCreate(ii, mv, lobpcg_solver);

   HYPRE_LOBPCGSetMaxIter(*lobpcg_solver, maxIterations);
   /* TODO: 搞清楚这是什么意思, 即是否可以以pvx_Hh为初值进行迭代 */
   HYPRE_LOBPCGSetPrecondUsageMode(*lobpcg_solver, pcgMode);
   HYPRE_LOBPCGSetTol(*lobpcg_solver, tol);
   HYPRE_LOBPCGSetPrintLevel(*lobpcg_solver, verbosity);

   return 0;	

}	

PASE_Int PASE_EigenSolverSetup(PASE_Solver eigen_solver,  
      PASE_ParCSRMatrix A, PASE_ParCSRMatrix B, 
      PASE_ParVector x, PASE_ParVector b)
{
   PASE_LOBPCGSetup (eigen_solver, A, b, x);
   PASE_LOBPCGSetupB(eigen_solver, B, x);
   return 0;

}		

PASE_Int PASE_EigenSolverSolve(PASE_Solver eigen_solver,  
      mv_MultiVectorPtr con, mv_MultiVectorPtr vec, PASE_Real* val )
{
   HYPRE_LOBPCGSolve(0, eigen_solver, con, vec, val );
   return 0;
};


PASE_Int PASE_EigenSolverDestroy(PASE_Solver eigen_solver)	
{
   HYPRE_LOBPCGDestroy(eigen_solver);
   return 0;	
}

