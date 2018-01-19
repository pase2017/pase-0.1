/*
 * =====================================================================================
 *
 *       Filename:  pase_lobpcg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年09月14日 13时35分26秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_lobpcg.h"

PASE_Int
PASE_LOBPCGSetup(PASE_Solver solver,  PASE_ParCSRMatrix A, PASE_ParVector b, PASE_ParVector x)
{
   return( hypre_LOBPCGSetup( solver,  A,  b,  x ) );
}

PASE_Int 
PASE_LOBPCGSetupB(PASE_Solver solver, PASE_ParCSRMatrix B, PASE_ParVector x)
{
   return( hypre_LOBPCGSetupB( solver,  B,  x ) );
}

/* 这里假设对于PASE的特征问题LOBPCG没有precond */
PASE_Int 
PASE_LOBPCGSolve( PASE_Int num_lock, PASE_Solver solver, mv_MultiVectorPtr con, 
      mv_MultiVectorPtr vec, PASE_Real* val )
{
   hypre_LOBPCGData* data = (hypre_LOBPCGData*)solver;

   PASE_ParCSRMatrix A = (PASE_ParCSRMatrix)data->A;
   PASE_ParCSRMatrix B = (PASE_ParCSRMatrix)data->B;

   PASE_ParVector *pvx;
   mv_TempMultiVector* tmp =
      (mv_TempMultiVector*) mv_MultiVectorGetData(vec);
   pvx = (PASE_ParVector*)(tmp -> vector);

   hypre_PASEDiag diag_data;
   hypre_PASEDiagCreate(&diag_data, A, B, NULL, NULL, NULL, pvx[0]->b_H);
   hypre_PASEDiagChange(&diag_data, A, B, pvx, NULL, NULL);

   hypre_LOBPCGSolve( num_lock, (void *) solver, con, vec, val );

   hypre_PASEDiagBack(&diag_data, A, B, pvx, NULL, NULL);
   hypre_PASEDiagDestroy(&diag_data);

   return hypre_error_flag;
}


PASE_Int
PASE_LOBPCGSetPrecond( PASE_Solver         solver,
                       PASE_PtrToSolverFcn precond,
                       PASE_PtrToSolverFcn precond_setup,
                       PASE_Solver         precond_solver )
{
   return( hypre_LOBPCGSetPrecond( (void *) solver,
                                   (PASE_Int (*)(void*, void*, void*, void*))precond,
				   (PASE_Int (*)(void*, void*, void*, void*))precond_setup,
                                   (void *) precond_solver ) );
}







