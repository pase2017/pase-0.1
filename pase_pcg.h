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

#include "pase_mv.h"



#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int
pase_ParKrylovCommInfo( void   *A, HYPRE_Int *my_id, HYPRE_Int *num_procs);
void *
pase_ParKrylovCreateVector( void *vvector );
HYPRE_Int
pase_ParKrylovDestroyVector( void *vvector );
HYPRE_Int
pase_ParKrylovMatvec( void *matvec_data, HYPRE_Complex alpha, void *A, void *x, HYPRE_Complex  beta, void *y );
HYPRE_Real
pase_ParKrylovInnerProd( void *x, void *y );
HYPRE_Int
pase_ParKrylovCopyVector( void *x, void *y );
HYPRE_Int
pase_ParKrylovClearVector( void *x );
HYPRE_Int
pase_ParKrylovScaleVector( HYPRE_Complex  alpha, void *x );
HYPRE_Int
pase_ParKrylovAxpy( HYPRE_Complex alpha, void *x, void *y );
HYPRE_Int
pase_ParKrylovIdentity( void *vdata, void *A, void *b, void *x );

HYPRE_Int
PASE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int
PASE_ParCSRPCGDestroy( HYPRE_Solver solver );

HYPRE_Int PASE_ParCSRSetupInterpreter( mv_InterfaceInterpreter* interpreter);
HYPRE_Int PASE_ParCSRSetupMatvec( HYPRE_MatvecFunctions* matvec_fn);



#ifdef __cplusplus
}
#endif

#endif
