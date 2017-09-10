/*
 * =====================================================================================
 *
 *       Filename:  pase_pcg.c
 *
 *    Description:  PASE_ParCSRMatrix下PCG求解线性方程组
 *
 *        Version:  1.0
 *        Created:  2017年09月08日 15时41分38秒
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
#include "pase_pcg.h"

/**
 * @brief 利用PASE_ParCSRMatrix的矩阵结构进行PCG设置
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 * 就是对solve进行赋初值, PASE_Solver这个结构是空指针, 此函数指向pase_PCGData
 * 主要基于PASE_ParCSRMatrixMatvec_HYPRE_ParVector进行矩阵乘向量的运算
 * 参考HYPRE中的linear solver PCG
 *
 */
PASE_Int PASE_ParCSRPCGSetup( PASE_Solver	  solver,
			      PASE_ParCSRMatrix   A, 
			      PASE_ParVector      b, 
			      PASE_ParVector      x )
{

   pase_PCGData *data = (pase_PCGData*)solver;
   pase_PCGFunctions *functions = data->functions;
   pase_PCGFunctionsCreate(functions);

}


/**
 * @brief 进行PCG迭代
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 * 注意返回的解向量的类型
 * 需要写一些Set函数设置PCG的一些参数, 如最大迭代次数以及tol
 * 参考HYPRE的PCG
 */
PASE_Int PASE_ParCSRPCGSolve( PASE_Solver        solver,
                              PASE_ParCSRMatrix   A,
                              PASE_ParVector      b,
                              PASE_ParVector      x )
{

   pase_PCGData *data = (pase_PCGData*)solver;
   pase_PCGFunctions *functions = data->functions;
   PASE_ParVector y;
   PASE_Real d = (functions->InnerProd(x, y));
   printf ( "%f\n", d );

   return 0;

}



PASE_Int pase_PCGFunctionsCreate (pase_PCGFunctions *functions)
{

   return 0;
}


