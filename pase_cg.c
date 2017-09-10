/*
 * =====================================================================================
 *
 *       Filename:  pase_pcg.c
 *
 *    Description:  PASE_ParCSRMatrix下CG求解线性方程组
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
#include "pase_cg.h"

/**
 * @brief 利用PASE_ParCSRMatrix的矩阵结构进行CG设置
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 * 就是对solve进行赋初值, PASE_Solver这个结构是空指针, 此函数指向pase_CGData
 * 主要基于PASE_ParCSRMatrixMatvec_HYPRE_ParVector进行矩阵乘向量的运算
 * 参考HYPRE中的linear solver CG
 *
 */
PASE_Int PASE_ParCSRCGSetup( PASE_Solver	  solver,
			      PASE_ParCSRMatrix   A, 
			      PASE_ParVector      b, 
			      PASE_ParVector      x )
{

   pase_CGData *data = (pase_CGData*)solver;
   pase_CGFunctions *functions = data->functions;
   pase_CGFunctionsCreate(functions);

}


/**
 * @brief 进行CG迭代
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 * 注意返回的解向量的类型
 * 需要写一些Set函数设置CG的一些参数, 如最大迭代次数以及tol
 * 参考HYPRE的CG
 */
PASE_Int PASE_ParCSRCGSolve( PASE_Solver        solver,
                              PASE_ParCSRMatrix   A,
                              PASE_ParVector      b,
                              PASE_ParVector      x )
{

   pase_CGData *data = (pase_CGData*)solver;
   pase_CGFunctions *functions = data->functions;
   PASE_ParVector r, p; 
   PASE_Real alpha, beta; // d = (functions->InnerProd(x, y));
   functions->MatMultiVec(blocksize,1.0,A,x,1.0,r);
   
for(PASE_Int iter=1; iter<max_it;iter++)
{
rho = functions->InnerProd(r,r);
if(iter>1)
{ beta = rho/rho_1;
 p = r+beta*p;
}
else
{p=r;
}
q=A*p;
alpha = rho/(functions->InnerProd(p,q));
x = x + alpha*p;
r=r-alpha*p;
error = functions->Norm2(r)/bnorm2;
if(error<=tol)
returen;
rho_1 = rho;
}
if(error>tol)
{
flag = 1;
}
   printf ( "%f\n", d );

   return 0;

}



PASE_Int pase_CGFunctionsCreate (pase_CGFunctions *functions)
{

   return 0;
}


