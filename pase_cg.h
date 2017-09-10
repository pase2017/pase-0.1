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

#ifndef _pase_cg_h_
#define _pase_cg_h_

#include "pase_mv.h"



#ifdef __cplusplus
extern "C" {
#endif


/* 用以进行CG的各种运算 */
typedef struct
{
   char *      (*CAlloc)        ( size_t count, size_t elt_size );
   PASE_Int    (*Free)          ( char *ptr );
   PASE_Int    (*CommInfo)      ( void  *A, PASE_Int   *my_id,
                                   PASE_Int   *num_procs );
   /* 返回x的L2范数*/
   PASE_Real   (*Norm2)        (void *x);	
   void *      (*CreateVector)  ( void *vector );
   PASE_Int    (*DestroyVector) ( void *vector );
   void *      (*MatvecCreate)  ( void *A, void *x );
   /* y = alpha Ax + beta y */
   PASE_Int    (*Matvec)        ( PASE_Real alpha, void *A,
                                   void *x, PASE_Real beta, void *y );
   PASE_Int    (*MatMultiVec)   ( PASE_Int block_size, PASE_Real alpha, void *A,
                                   void *x, PASE_Real beta, void *y);
   PASE_Int    (*MatvecDestroy) ( void *matvec_data );
   PASE_Real   (*InnerProd)     ( void *x, void *y );
   PASE_Int    (*CopyVector)    ( void *x, void *y );
   PASE_Int    (*ClearVector)   ( void *x );
   PASE_Int    (*ScaleVector)   ( PASE_Real alpha, void *x );
   /* y = alpha x + y  */
   PASE_Int    (*Axpy)          ( PASE_Real alpha, void *x, void *y );

   PASE_Int    (*Precond)(void *vdata , void *A , void *b , void *x);
   PASE_Int    (*PrecondSetup)(void *vdata , void *A , void *b , void *x);

} pase_CGFunctions;

typedef struct
{
   PASE_Real   tol;
   PASE_Real   atolf;
   PASE_Real   cf_tol;
   PASE_Real   a_tol;
   PASE_Real   rtol;
   PASE_Int    max_iter;
   PASE_Int    two_norm;
   PASE_Int    rel_change;
   PASE_Int    recompute_residual;
   PASE_Int    recompute_residual_p;
   PASE_Int    stop_crit;
   PASE_Int    converged;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

   PASE_Int   	owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    	*matvec_data;
   void    	*precond_data;

   pase_CGFunctions * functions;

   /* log info (always logged) */
   PASE_Int      num_iterations;
   PASE_Real   rel_residual_norm;

   PASE_Int   print_level; /* printing when print_level>0 */
   PASE_Int   logging;  /* extra computations for logging when logging>0 */
   PASE_Real  *norms;
   PASE_Real  *rel_norms;

} pase_CGData;

PASE_Int pase_CGFunctionsCreate (pase_CGFunctions *functions);

/* 尝试利用已有分层信息进行预条件, 或者根本不需要预条件 */
/* 直接编写并行CG, 基于PASE_ParCSRMatrix, 考虑多向量的情形, 可以利用Gauss消元法求解(PASE) */

PASE_Int PASE_ParCSRCGSetup( PASE_Solver	   solver,
			       PASE_ParCSRMatrix   A, 
			       PASE_ParVector      b, 
			       PASE_ParVector      x );
PASE_Int PASE_ParCSRCGSolve( PASE_Solver        solver,
                               PASE_ParCSRMatrix   A,
                               PASE_ParVector      b,
                               PASE_ParVector      x );


#ifdef __cplusplus
}
#endif

#endif
