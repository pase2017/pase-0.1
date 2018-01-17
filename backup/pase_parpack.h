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

#ifndef _pase_parpack_h_
#define _pase_parpack_h_

#include "pase_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 用以进行Parpack的各种运算 */
typedef struct
{
   void*  (*MatvecCreate)     ( void *A, void *x );
   PASE_Int (*Matvec)        ( void *matvec_data, PASE_Real alpha, void *A,
                                void *x, PASE_Real beta, void *y );
   PASE_Int (*MatvecDestroy) ( void *matvec_data );
   
   void*  (*MatMultiVecCreate)     ( void *A, void *x );
   PASE_Int (*MatMultiVec)        ( void *data, PASE_Real alpha, void *A,
                                     void *x, PASE_Real beta, void *y );
   PASE_Int (*MatMultiVecDestroy) ( void *data );

} pase_ParpackFunctions;

typedef struct
{

   PASE_Int             num_eigenvalues;
   PASE_Real            tolerance;
   PASE_Int             max_iterations;
   PASE_Int             precond_usage_mode;
   PASE_Int             iteration_mumber;

   void*                 A;
   void*                 B;
   void*                 precond_solver;
   PASE_Int             (*Precond)(void*,void*,void*,void*);
   PASE_Int             (*PrecondSetup)(void*,void*,void*,void*);
   /* 可能还需要求解器和矩阵向量运算, 这样就不能用PASE自带的预条件, 可以利用AMG得到的分层矩阵进行预条件
    * 只能利用自己的PCG, */
   pase_ParpackFunctions*        matvec_functions;

} pase_ParpackData;


PASE_Int PASE_PCGFunctionsCreate ( PASE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, PASE_Real beta, PASE_ParVector y );

PASE_Int PASE_ParpackSetup( PASE_Solver          parpack_solver, 
                            PASE_ParCSRMatrix    A, 
			    PASE_ParCSRMatrix    B, 
			    PASE_Solver          precond_solver, 
			    PASE_Int             max_iter, 
			    PASE_Real            tol, 
			    PASE_Int             num_eigenvalues );
PASE_Int PASE_ParpackSolve( PASE_Solver          parpack_solver,
			    PASE_Real*        eigenvalues,
			    PASE_ParVector*       eigenvectors );





#ifdef __cplusplus
}
#endif

#endif
