/*
 * =====================================================================================
 *
 *       Filename:  pase_parpack.c
 *
 *    Description:  PASE_ParCSRMatrix进行特征值求解
 *
 *        Version:  1.0
 *        Created:  2017年09月08日 16时02分15秒
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

#include "pase_parpack.h"


/**
 * @brief 
 *
 * @param parpack_solver
 * @param A
 * @param B
 * @param precond
 * @param precond_setup
 * @param precond_solver
 * @param max_iter
 * @param tol
 * @param num_eigenvalues
 *
 * @return 
 *
 * parpack_solver的初始化, 将之重定义为pase_ParpackData
 *
 */
PASE_Int PASE_ParpackSetup( PASE_Solver          parpack_solver, 
                             PASE_ParCSRMatrix     A, 
			     PASE_ParCSRMatrix     B, 
			     PASE_Solver          precond_solver, 
			     PASE_Int             max_iter, 
			     PASE_Real            tol, 
			     PASE_Int             num_eigenvalues )
{
   return 0;
}

/**
 * @brief 
 *
 * @param parpack_solver
 * @param eigenvalues
 * @param eigenvectors
 *
 * @return 
 *
 * 利用parpack_solver的成员函数进行矩阵向量运算以及它们的创建和销毁
 *
 */
PASE_Int PASE_ParpackSolve( PASE_Solver          parpack_solver,
			     PASE_Real*          eigenvalues,
			     PASE_ParVector*     eigenvectors )
{
   return 0;
}





