/*
 * =====================================================================================
 *
 *       Filename:  pase_lobpcg.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年09月14日 13时35分33秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef _pase_lobpcg_h_
#define _pase_lobpcg_h_


#include "pase_int.h"
#include "pase_diag.h"
#include "pase_pcg.h"

#ifdef __cplusplus
extern "C" {
#endif

PASE_Int
PASE_LOBPCGSetup(PASE_Solver lobpcg_solver,  PASE_ParCSRMatrix A, PASE_ParVector b, PASE_ParVector x);
PASE_Int 
PASE_LOBPCGSetupB(PASE_Solver solver, PASE_ParCSRMatrix B, PASE_ParVector x);

PASE_Int 
PASE_LOBPCGSolve( HYPRE_Int num_lock, HYPRE_Solver solver, mv_MultiVectorPtr con, 
      mv_MultiVectorPtr vec, HYPRE_Real* val );


PASE_Int
PASE_LOBPCGSetPrecond( PASE_Solver         solver,
                       PASE_PtrToSolverFcn precond,
                       PASE_PtrToSolverFcn precond_setup,
                       PASE_Solver         precond_solver );









#ifdef __cplusplus
}
#endif


#endif
