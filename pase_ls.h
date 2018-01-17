/*
 * =====================================================================================
 *
 *       Filename:  pase_ls.h
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

#ifndef _pase_ls_h_
#define _pase_ls_h_

#include "pase_pcg.h"

#ifdef __cplusplus
extern "C" {
#endif

PASE_Int PASE_LinearSolverCreate(PASE_Solver* linear_solver, MPI_Comm comm);


PASE_Int PASE_LinearSolverSetup(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
      PASE_ParVector F, PASE_ParVector U);

PASE_Int PASE_LinearSolverSolve(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
      PASE_ParVector F, PASE_ParVector U);

PASE_Int PASE_LinearSolverDestroy(PASE_Solver linear_solver, PASE_Solver precond);


#ifdef __cplusplus
}
#endif

#endif
