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

PASE_Int
PASE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver );
PASE_Int
PASE_ParCSRPCGDestroy( HYPRE_Solver solver );

#ifdef __cplusplus
}
#endif

#endif
