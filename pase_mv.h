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

#ifndef _pase_mv_h_
#define _pase_mv_h_

#include "pase_hypre.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef struct pase_ParCSRMatrix_struct 
{
   
   MPI_Comm              comm;

   HYPRE_Int             N_H;
   HYPRE_Int             block_size;

   /* N_H阶的并行矩阵 */
   HYPRE_ParCSRMatrix    A_H;
   /* blockSize个N_H阶的并行向量 */
   mv_MultiVectorPtr     aux_Hh;
   /* blockSize个N_H阶的并行向量 */
   mv_MultiVectorPtr     aux_hH;
   /* blockSize*blockSize的数组 */
   HYPRE_CSRMatrix*      aux_hh;

} pase_ParCSRMatrix;
typedef struct pase_ParCSRMatrix_struct *PASE_ParCSRMatrix;


typedef struct pase_ParVector_struct 
{
   
   MPI_Comm              comm;

   HYPRE_Int             N_H;
   HYPRE_Int             block_size;

   /* N_H阶的并行矩阵 */
   HYPRE_ParVector       b_H;
   /* blockSize的数组 */
   HYPRE_Vector*         aux_h;

} pase_ParVector;
typedef struct pase_ParVector_struct *PASE_ParVector;


HYPRE_Int PASE_ParCSRMatrixCreate( MPI_Comm comm , 
                                   HYPRE_Int N_H, HYPRE_Int blockSize,
                                   HYPRE_ParCSRMatrix aux_HH, 
				   mv_MultiVectorPtr aux_Hh,
				   mv_MultiVectorPtr aux_hH,
				   HYPRE_Real* aux_hh, 
				   PASE_ParCSRMatrix matrix );
HYPRE_Int PASE_ParCSRMatrixDestroy( PASE_ParCSRMatrix matrix );

/* y = alpha A + beta y */
HYPRE_Int PASE_ParCSRMatrixMatvec ( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y );
HYPRE_Int PASE_ParCSRMatrixMatvecT( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y );


/* 注意向量的类型 */
HYPRE_Int PASE_ParCSRMatrixMatvec_HYPRE_ParVector ( HYPRE_Real alpha, PASE_ParCSRMatrix A, HYPRE_ParVector x, HYPRE_Real beta, HYPRE_ParVector y );


#ifdef __cplusplus
}
#endif

#endif
