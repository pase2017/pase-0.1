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
   hypre_ParCSRMatrix*   A_H;
   hypre_ParCSRMatrix*   A_h;
   hypre_ParCSRMatrix*   P;
   /* TODO:
    * 利用mv_TempMultiVector进行存储*/
   hypre_ParVector**     u_h;
   /* blockSize个N_H阶的并行向量 */
   hypre_ParVector**     aux_Hh;
   /* blockSize个N_H阶的并行向量, 对于对称矩阵, 这个指针直接直向aux_Hh */
   hypre_ParVector**     aux_hH;
   /* blockSize*blockSize的数组 */
   hypre_CSRMatrix*      aux_hh;

} pase_ParCSRMatrix;
typedef struct pase_ParCSRMatrix_struct *PASE_ParCSRMatrix;


typedef struct pase_ParVector_struct 
{
   
   MPI_Comm              comm;

   HYPRE_Int             N_H;
   HYPRE_Int             block_size;

   /* N_H阶的并行矩阵 */
   hypre_ParVector*      b_H;
   HYPRE_Int             owns_ParVector;
   /* blockSize的数组 */
   hypre_Vector*         aux_h;

} pase_ParVector;
typedef struct pase_ParVector_struct *PASE_ParVector;


/* 这里不应该如此简单的给这样直接的行参, 应该给A_H A_h 以及block_size个h上的向量, 形成PASE矩阵 */
HYPRE_Int PASE_ParCSRMatrixCreate( MPI_Comm comm , 
                                   HYPRE_Int block_size,
                                   HYPRE_ParCSRMatrix A_H, 
                                   HYPRE_ParCSRMatrix P,
                                   HYPRE_ParCSRMatrix A_h, 
				   HYPRE_ParVector*   u_h, 
				   PASE_ParCSRMatrix* matrix, 
				   HYPRE_ParVector    workspace_H, 
				   HYPRE_ParVector    workspace_h
				   );
HYPRE_Int PASE_ParCSRMatrixDestroy( PASE_ParCSRMatrix  matrix );

HYPRE_Int PASE_ParVectorCreate(    MPI_Comm comm , 
                                   HYPRE_Int N_H,
                                   HYPRE_Int block_size,
                                   HYPRE_ParVector    b_H, 
                                   HYPRE_Int*         partitioning, 
				   PASE_ParVector*    vector
				   );
HYPRE_Int PASE_ParVectorDestroy( PASE_ParVector vector );

HYPRE_Int PASE_ParCSRMatrixSetAuxSpace( MPI_Comm comm , 
				   PASE_ParCSRMatrix  matrix, 
                                   HYPRE_Int block_size,
                                   HYPRE_ParCSRMatrix P,
                                   HYPRE_ParCSRMatrix A_h, 
				   HYPRE_ParVector*   u_h, 
                                   HYPRE_Int          begin_idx,
				   HYPRE_ParVector    workspace_H, 
				   HYPRE_ParVector    workspace_h
				   );

HYPRE_Int PASE_ParCSRMatrixPrint( PASE_ParCSRMatrix matrix , const char *file_name );
HYPRE_Int PASE_ParVectorPrint( PASE_ParVector vector , const char *file_name );
HYPRE_Int PASE_ParVectorCopy(PASE_ParVector x, PASE_ParVector y);
HYPRE_Int PASE_ParVectorInnerProd( PASE_ParVector x, PASE_ParVector y, HYPRE_Real *prod);
HYPRE_Int PASE_ParVectorCopy(PASE_ParVector x, PASE_ParVector y);
HYPRE_Int PASE_ParVectorAxpy( HYPRE_Real alpha , PASE_ParVector x , PASE_ParVector y );
HYPRE_Int PASE_ParVectorSetConstantValues( PASE_ParVector v , HYPRE_Real value );
HYPRE_Int PASE_ParVectorScale ( HYPRE_Real alpha , PASE_ParVector y );
HYPRE_Int PASE_ParVectorSetRandomValues( PASE_ParVector v, HYPRE_Int seed );

HYPRE_Int PASE_ParVectorGetParVector( HYPRE_ParCSRMatrix P, HYPRE_Int block_size, HYPRE_ParVector *vector_h, 
      PASE_ParVector vector_Hh, HYPRE_ParVector vector );



/* y = alpha A + beta y */
HYPRE_Int PASE_ParCSRMatrixMatvec ( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y );
HYPRE_Int PASE_ParCSRMatrixMatvecT( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y );





HYPRE_Int
PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_Comm comm , 
				   PASE_ParCSRMatrix  matrix, 
                                   HYPRE_Int block_size,
                                   HYPRE_ParCSRMatrix P,
                                   PASE_ParCSRMatrix A_h, 
				   PASE_ParVector*   u_h, 
				   HYPRE_ParVector    workspace_H, 
				   PASE_ParVector    workspace_hH
				   );

HYPRE_Int
PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_Comm comm , 
                                   HYPRE_Int block_size,
                                   HYPRE_ParCSRMatrix A_H, 
                                   HYPRE_ParCSRMatrix P,
                                   PASE_ParCSRMatrix A_h, 
				   PASE_ParVector*   u_h, 
				   PASE_ParCSRMatrix* matrix, 
				   HYPRE_ParVector    workspace_H, 
				   PASE_ParVector    workspace_hH
				   );


















/* 注意向量的类型 */
HYPRE_Int PASE_ParCSRMatrixMatvec_HYPRE_ParVector ( HYPRE_Real alpha, PASE_ParCSRMatrix A, HYPRE_ParVector x, HYPRE_Real beta, HYPRE_ParVector y );


#ifdef __cplusplus
}
#endif

#endif
