/*
 * =====================================================================================
 *
 *       Filename:  pash_int.h
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

#ifndef _pase_int_h_
#define _pase_int_h_

#include "pase_mv.h"



#ifdef __cplusplus
extern "C" {
#endif

PASE_Int
pase_ParKrylovCommInfo( void   *A, PASE_Int *my_id, PASE_Int *num_procs);
void *
pase_ParKrylovCreateVector( void *vvector );
PASE_Int
pase_ParKrylovDestroyVector( void *vvector );
PASE_Int
pase_ParKrylovMatvec( void *matvec_data, PASE_Real alpha, void *A, void *x, PASE_Real  beta, void *y );
PASE_Real
pase_ParKrylovInnerProd( void *x, void *y );
PASE_Int
pase_ParKrylovCopyVector( void *x, void *y );
PASE_Int
pase_ParKrylovClearVector( void *x );
PASE_Int
pase_ParKrylovScaleVector( PASE_Real  alpha, void *x );
PASE_Int
pase_ParKrylovAxpy( PASE_Real alpha, void *x, void *y );
PASE_Int
pase_ParKrylovIdentity( void *vdata, void *A, void *b, void *x );

PASE_Int PASE_ParCSRSetupInterpreter( mv_InterfaceInterpreter* interpreter);
PASE_Int PASE_ParCSRSetupMatvec( HYPRE_MatvecFunctions* matvec_fn);



#ifdef __cplusplus
}
#endif

#endif
