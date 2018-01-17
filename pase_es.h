/*
 * =====================================================================================
 *
 *       Filename:  pase_es.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年01月17日 09时50分48秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef _pase_es_h_
#define _pase_es_h_

#include "pase_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

PASE_Int PASE_EigenSolverCreate(PASE_Solver* eigen_solver, 
      mv_InterfaceInterpreter* ii, HYPRE_MatvecFunctions* mv, 
      PASE_Solver* linear_solver, PASE_Solver* precond, MPI_Comm comm);

PASE_Int PASE_EigenSolverSetup(PASE_Solver eigen_solver,  
      PASE_ParCSRMatrix A, PASE_ParCSRMatrix B, 
      PASE_ParVector x, PASE_ParVector b);

PASE_Int PASE_EigenSolverSolve(PASE_Solver eigen_solver,  
      mv_MultiVectorPtr con, mv_MultiVectorPtr vec, PASE_Real* val );


PASE_Int PASE_EigenSolverDestroy(PASE_Solver eigen_solver);


#ifdef __cplusplus
}
#endif

#endif
