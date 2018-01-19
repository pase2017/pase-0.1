/*
 * =====================================================================================
 *
 *       Filename:  pash.h
 *
 *    Description:  后期可以考虑将行参类型都变成void *, 以方便修改和在不同计算机上调试
 *                  一般而言, 可以让用户调用的函数以PASE_开头, 内部函数以pase_开头
 *
 *        Version:  1.0
 *        Created:  2017年08月29日 14时15分22秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  LIYU 
 *   Organization:  LSEC
 *
 * =====================================================================================
 */

#ifndef _pase_pcg_h_
#define _pase_pcg_h_

#include "pase_int.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef PASE_Int (*PASE_PtrToSolverFcn)(PASE_Solver,
                                        PASE_ParCSRMatrix,
                                        PASE_ParVector,
                                        PASE_ParVector);

PASE_Int
PASE_ParCSRPCGCreate( MPI_Comm comm, PASE_Solver *solver );
PASE_Int
PASE_ParCSRPCGDestroy( PASE_Solver solver );
PASE_Int 
PASE_ParCSRPCGSetup( PASE_Solver solver,
                     PASE_ParCSRMatrix A,
                     PASE_ParVector b,
                     PASE_ParVector x);
PASE_Int 
PASE_ParCSRPCGSolve( PASE_Solver solver,
                     PASE_ParCSRMatrix A,
                     PASE_ParVector b,
                     PASE_ParVector x);
	


#ifdef __cplusplus
}
#endif

#endif
