/*
 * =====================================================================================
 *
 *       Filename:  pase_mv.c
 *
 *    Description:  PASE_ParCSRMatrix PASE_ParVetor各种操作
 *
 *        Version:  1.0
 *        Created:  2017年09月08日 13时13分17秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_mv.h"


/**
 * @brief 创建PASE并行矩阵
 *
 * @param comm 
 * @param N_H            aux_HH的阶数
 * @param blockSize      aux_Hh这个MultiVector的向量个数
 * @param aux_HH         HYPRE_ParCSRMatrix, 直接有HYPRE生成, 可能是AMG生成的
 * @param aux_Hh         MultiVector, 一共blockSize个N_H维HYPRE_ParVector
 * @param aux_hH         对于对称矩阵, 这个指针直接直向aux_Hh
 * @param aux_hh         是一个全局数组, 由各个特征向量的A内积或B内积生成.         
 * 
 * @param matrix         基于上述输入, 得到块矩阵
 *
 * @return               如果blockSize为0的情形也要考虑到
 *
 * 
 * 本函数主要是对PASE_ParCSRMatrix的成员赋值, 检查哪些是指针哪些需要malloc
 *
 */
HYPRE_Int PASE_ParCSRMatrixCreate( MPI_Comm comm , 
                                   HYPRE_Int N_H, HYPRE_Int block_size,
                                   HYPRE_ParCSRMatrix A_H, 
				   mv_MultiVectorPtr aux_Hh,
				   mv_MultiVectorPtr aux_hH,
				   HYPRE_Real* aux_hh, 
				   PASE_ParCSRMatrix matrix )
{

   /* calloc并返回一个(pase_ParCSRMatrix *)的指针 */
   matrix = hypre_CTAlloc(pase_ParCSRMatrix, 1);
   /* 是否需要为调用成员给出一个宏 */
   matrix->A_H = A_H;
   matrix->N_H = N_H;

   return 0;
}



/**
 * @brief PASE_ParCSRMatrix销毁
 *
 * @param matrix
 *
 * @return 
 *
 * 将各个成员, 若是指针则置为NULL, 若有malloc则free
 *
 */
HYPRE_Int PASE_ParCSRMatrixDestroy( PASE_ParCSRMatrix matrix )
{
   return 0;
}

/**
 * @brief y = alpha A x + beta y
 *
 * @param alpha
 * @param A
 * @param x
 * @param beta
 * @param y
 *
 * @return 
 *
 * 这里x, y也是PASE_ParVector, 它们的成员b_H要与A的成员aux_HH的分布一致 
 * 判断矩阵和向量之间是否可以进行运算
 *
 */
HYPRE_Int PASE_ParCSRMatrixMatvec ( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y )
{

   return 0;
}

/**
 * @brief y = alpha A^T x + beta y
 *
 * @param alpha
 * @param A
 * @param x
 * @param beta
 * @param y
 *
 * @return 
 *
 * 这里x也是PASE_ParVector
 * 判断矩阵和向量之间是否可以进行运算
 *
 */
HYPRE_Int PASE_ParCSRMatrixMatvecT( HYPRE_Real alpha, PASE_ParCSRMatrix A, PASE_ParVector x, HYPRE_Real beta, PASE_ParVector y )
{

   return 0;
}



/**
 * @brief y = alpha A^T x + beta y 函数名的意思是强调向量的类型
 *
 * @param alpha
 * @param A           PASE_ParCSRMatrix
 * @param x           HYPRE_ParVector
 * @param beta
 * @param y           HYPRE_ParVector
 *
 * @return 
 *  
 * PASE并行块矩阵与HYPRE_Vector的矩阵向量运算, 会用在自己编写的PCG程序中, 以及调用Parpack时进行矩阵乘以向量
 * 如果可以利用HYPRE_ParVector进行PCG的编写, 则将HYPRE_Vector换做HYPRE_ParVector, 这样更对
 * 这里要判断矩阵向量是否可以进行运算.
 *
 * 大致算法:
 * 首先将x全收集(allgather)形成全局向量, 然后将之分为两个部分, 再进行运算, 多用指针
 * 一定要注意的是, 这里x, y的分布式存储和A的分布式存储完全不同.
 *
 */
HYPRE_Int PASE_ParCSRMatrixMatvec_HYPRE_ParVector ( HYPRE_Real alpha, PASE_ParCSRMatrix A, HYPRE_ParVector x, HYPRE_Real beta, HYPRE_ParVector y )
{

   return 0;
}

HYPRE_Int PASE_ParCSRMatrixMatvec_HYPRE_Vector ( HYPRE_Real alpha, PASE_ParCSRMatrix A, HYPRE_Vector x, HYPRE_Real beta, HYPRE_Vector y )
{

   return 0;
}
