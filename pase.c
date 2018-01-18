/*
   pase.c

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is PASE (Parallels Auxiliary Space Eigen-solver)
                 with LOBPCG and AMG preconditioner.

		 首先通过最粗+aux求特征值问题与细空间求解问题, 得到特征向量的好初值
		 然后V循环从细到粗在各个细+aux上进行pcg算法, 生成各个层的pase矩阵
		 最后V循环从粗到细在各个细+aux上进行pcg算法, 利用已生成的pase矩阵, 更新解(右端项)
   
   Created:      2017.11.11

   Author:       Li Yu (liyu@lsec.cc.ac.cn).
*/
#include "pase.h"

PASE_Int pase_TwoGridSolve( HYPRE_Solver      solver, 
                            HYPRE_Int         block_size,
                            HYPRE_ParVector*  constraints, 
                            HYPRE_ParVector*  eigenvectors, 
                            HYPRE_Real*       eigenvalues )
{
   HYPRE_ParVector *pvx_pre;
   HYPRE_ParVector *pvx;
   HYPRE_Int num_conv, iter, begin_idx, idx_eig, i;
   hypre_PASEData *pase_data = (hypre_PASEData*)solver;
   PASE_MultiGrid multi_grid = pase_data->multi_grid;
   HYPRE_ParVector tmp_pvx;

   HYPRE_Int num_levels;
   num_levels = pase_MultiGridDataNumLevels(multi_grid);
   HYPRE_Real tmp_double, residual, tolerance, min_gap;

   tolerance = 1E-6;
   min_gap   = 1E-1;

   HYPRE_Solver lobpcg_solver = hypre_PASEDataEigenSolver (pase_data);
   HYPRE_Solver linear_solver = hypre_PASEDataLinearSolver(pase_data);
   HYPRE_Int  (*linear_setup)(void*, void*, void*, void*) = (pase_data->pase_functions).LinearSetup;
   HYPRE_Int  (*linear_solve)(void*, void*, void*, void*) = (pase_data->pase_functions).LinearSolve;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   /*共4层: Q0 = P0P1P2  Q1 = P1P2  Q2 = P2 */
   hypre_ParCSRMatrix **Q_array;

   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   A_array = pase_MultiGridDataAArray(multi_grid); 
   B_array = pase_MultiGridDataBArray(multi_grid); 
   Q_array = pase_MultiGridDataQArray(multi_grid); 

   U_array = pase_MultiGridDataUArray(multi_grid); 
   F_array = pase_MultiGridDataFArray(multi_grid); 

   PASE_ParCSRMatrix  parcsr_A_Hh, parcsr_B_Hh;
   PASE_ParVector*    pvx_Hh;    
   parcsr_A_Hh = pase_data->A_pase;
   parcsr_B_Hh = pase_data->B_pase;
   pvx_Hh      = pase_data->vecs_pase;

   mv_MultiVectorPtr eigenvectors_Hh, constraints_Hh;
   eigenvectors_Hh = pase_data->mv_eigs_pase;
   constraints_Hh  = pase_data->mv_cons_pase;

   HYPRE_Int max_its = pase_data->max_iters;
   max_its = 10;

   pvx     = eigenvectors;
   pvx_pre = hypre_CTAlloc (HYPRE_ParVector, block_size);
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      pvx_pre[idx_eig] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)U_array[0]);
   }
   linear_setup((void*)linear_solver, (void*)A_array[0], (void*)U_array[0], (void*)F_array[0]);

   num_conv  = 0;
   begin_idx = num_conv;

   iter = 0;
   while (iter < max_its && num_conv < block_size)
   {
      /* 只从不收敛的特征值开始求解线性问题 */
      for (idx_eig = num_conv; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 linear_solve((void*)linear_solver, (void*)A_array[0], (void*)F_array[0], (void*)pvx[idx_eig]);
      }
      /* 这里设置了begin_idx这个参数, 使得特征子空间一起收敛 */
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[0],
	    A_array[0], pvx, begin_idx, U_array[num_levels-1], U_array[0] );
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[0],
	    B_array[0], pvx, begin_idx, U_array[num_levels-1], U_array[0] );
      for ( idx_eig = begin_idx; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	 pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
      }
      /* LOBPCG eigensolver */
      HYPRE_LOBPCGSolve(num_conv, lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

      /* 将pvx_Hh插值到0层 */
      for (idx_eig = num_conv; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorGetParVector( Q_array[0], block_size, pvx, pvx_Hh[idx_eig], pvx_pre[idx_eig] );
      }

      /* TODO:是否可以从num_conv开始, 应该是特征子空间整个一起收敛 */
      /* 是否需要计算每个特征值的残量, 再利用activeMark进行标记, 使得对应的特征值再进行光滑 */
      begin_idx = num_conv;
      for (idx_eig = num_conv; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx_pre[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx_pre[idx_eig], &tmp_double);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx_pre[idx_eig], -eigenvalues[idx_eig], F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], F_array[0], &residual);
	 residual = sqrt(residual/tmp_double);
	 /* TODO:收敛准则的科学性 */
	 if (residual < tolerance+tolerance*eigenvalues[idx_eig])
	 {
	    ++num_conv;
	 }
	 else 
	 {
	    break;
	 }
      }
      if (num_conv > 0 && num_conv < block_size)
      {
	 /* 比较已收敛的特征值与后一个特征值是否是重根, 若是则再往前寻找特征子空间的开始 */
	 if ( fabs(eigenvalues[num_conv-1]-eigenvalues[num_conv]) < min_gap*fabs(eigenvalues[num_conv-1]) )
	 {
	    for (i = num_conv-2; i >= begin_idx-1; --i)
	    {
	       if ( fabs(eigenvalues[num_conv-1]-eigenvalues[i]) > min_gap*fabs(eigenvalues[num_conv-1]) )
	       {
		  num_conv = i+1;
		  break;
	       }
	    }
	 }
      }
      printf ( "begin_idx = %d, num_conv = %d\n", begin_idx, num_conv );

      ++iter;

      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 tmp_pvx          = pvx[idx_eig];
	 pvx[idx_eig]     = pvx_pre[idx_eig];
	 pvx_pre[idx_eig] = tmp_pvx;
      }

   }
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_pre[idx_eig]);
   }
   hypre_TFree(pvx_pre);


   return 0;
}


/* 以num_levels-1层的特征向量为初值进行迭代 */
PASE_Int pase_FullMultiGridSolve( HYPRE_Solver      solver, 
                                  HYPRE_Int         block_size,
                                  HYPRE_ParVector*  constraints, 
                                  HYPRE_ParVector*  eigenvectors, 
                                  HYPRE_Real*       eigenvalues )
{
   HYPRE_ParVector *pvx_pre;
   HYPRE_ParVector *pvx;

   hypre_PASEData *pase_data = (hypre_PASEData*)solver;
   PASE_MultiGrid multi_grid = pase_data->multi_grid;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /*共4层: Q0 = P0P1P2  Q1 = P1P2  Q2 = P2 */
   hypre_ParCSRMatrix **Q_array;

   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   A_array = pase_MultiGridDataAArray(multi_grid); 
   B_array = pase_MultiGridDataBArray(multi_grid); 
   P_array = pase_MultiGridDataPArray(multi_grid); 
   Q_array = pase_MultiGridDataQArray(multi_grid); 

   U_array = pase_MultiGridDataUArray(multi_grid); 
   F_array = pase_MultiGridDataFArray(multi_grid); 


   HYPRE_Int num_levels, level, idx_eig;
   num_levels = pase_MultiGridDataNumLevels(multi_grid);

   HYPRE_Solver lobpcg_solver, linear_solver, precond;
   linear_solver = hypre_PASEDataLinearSolver(pase_data);

   /* LOBPCG precond */
   HYPRE_BoomerAMGCreate(&precond);
   HYPRE_BoomerAMGSetOldDefault(precond);
   HYPRE_BoomerAMGSetNumSweeps (precond,  1);
   HYPRE_BoomerAMGSetTol       (precond,  0.0); /*  conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter   (precond,  1); /*  do only one iteration! */
   HYPRE_BoomerAMGSetPrintLevel(precond,  0); /*  print amg solution info */

   PASE_ParCSRMatrix  parcsr_A_Hh, parcsr_B_Hh;
   PASE_ParVector*    pvx_Hh;    
   parcsr_A_Hh = pase_data->A_pase;
   parcsr_B_Hh = pase_data->B_pase;
   pvx_Hh      = pase_data->vecs_pase;

   mv_MultiVectorPtr eigenvectors_Hh, constraints_Hh;
   eigenvectors_Hh = pase_data->mv_eigs_pase;
   constraints_Hh  = pase_data->mv_cons_pase;

   /* 首先赋予最粗空间上的初值 */
   HYPRE_LOBPCGCreate(pase_data->interpreter_hypre,  pase_data->matvec_fn_hypre,  &(pase_data->eigen_solver));
   lobpcg_solver = pase_data->eigen_solver;
   /*  最粗层特征值问题的参数选取 */
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver,  50);
   /*  use rhs as initial guess for inner pcg iterations */
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver,  1);
   HYPRE_LOBPCGSetTol(lobpcg_solver,  0.0);
   HYPRE_LOBPCGSetPrintLevel(lobpcg_solver,  0);
   mv_MultiVectorSetRandom (pase_data->mv_eigs_hypre,  775);
   HYPRE_LOBPCGSetPrecond(lobpcg_solver,  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, 
	 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,  precond);
   /*  这里(HYPRE_Matrix)的存在意义是变成一个空结构 */
   HYPRE_LOBPCGSetup (lobpcg_solver, (HYPRE_Matrix)(pase_data->A_hypre), (HYPRE_Vector)(pase_data->b_hypre), (HYPRE_Vector)(pase_data->x_hypre));
   HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)(pase_data->B_hypre), (HYPRE_Vector)(pase_data->x_hypre));
   HYPRE_LOBPCGSolve (0,  lobpcg_solver, pase_data->mv_cons_hypre, pase_data->mv_eigs_hypre, eigenvalues);
   HYPRE_LOBPCGDestroy(lobpcg_solver);
   HYPRE_BoomerAMGDestroy(precond);

   
   HYPRE_BoomerAMGCreate(&precond);
   HYPRE_BoomerAMGSetOldDefault(precond);
   HYPRE_BoomerAMGSetNumSweeps (precond,  1);
   HYPRE_BoomerAMGSetTol       (precond,  0.0); /*  conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter   (precond,  1); /*  do only one iteration! */
   HYPRE_BoomerAMGSetPrintLevel(precond,  0); /*  print amg solution info */


   HYPRE_LOBPCGCreate(pase_data->interpreter_pase,  pase_data->matvec_fn_pase,  &(pase_data->eigen_solver));
   lobpcg_solver = pase_data->eigen_solver;
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver,  10);
   /*  use rhs as initial guess for inner pcg iterations */
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver,  1);
   HYPRE_LOBPCGSetTol(lobpcg_solver,  0.0);
   HYPRE_LOBPCGSetPrintLevel(lobpcg_solver,  0);

   PASE_LOBPCGSetup (lobpcg_solver, pase_data->A_pase, pase_data->b_pase, pase_data->x_pase);
   PASE_LOBPCGSetupB(lobpcg_solver, pase_data->B_pase, pase_data->x_pase);

   pvx     = hypre_CTAlloc (HYPRE_ParVector, block_size);
   pvx_pre = hypre_CTAlloc (HYPRE_ParVector, block_size);
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      pvx[idx_eig] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)U_array[num_levels-2]);
   }
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      /* 生成当前网格下的特征向量 */
      HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[num_levels-2], pase_data->vecs_hypre[idx_eig], 0.0, pvx[idx_eig] );
   }

   HYPRE_Int  (*linear_setup)(void*, void*, void*, void*) = (pase_data->pase_functions).LinearSetup;
   HYPRE_Int  (*linear_solve)(void*, void*, void*, void*) = (pase_data->pase_functions).LinearSolve;

   HYPRE_Int (*Matvec_matvec) (void*, PASE_Real , void*, void*, PASE_Real, void*) = pase_data->matvec_fn_hypre->Matvec;


   int row, col;
   hypre_CSRMatrix *bTy, *cTy, *yTBy;
   HYPRE_ParVector *pvx_H;
   pvx_H = pase_data->vecs_hypre;
   /* 对角化 */
   bTy  = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   cTy  = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   yTBy = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   hypre_CSRMatrixInitialize( bTy );
   hypre_CSRMatrixInitialize( cTy );
   hypre_CSRMatrixInitialize( yTBy);
   HYPRE_Int*  matrix_i = bTy->i;
   HYPRE_Int*  matrix_j = bTy->j;
   HYPRE_Real* matrix_data;
   for (row = 0; row < block_size+1; ++row)
   {
      matrix_i[row] = row*block_size;
   }
   for ( row = 0; row < block_size; ++row)
   {
      for ( col = 0; col < block_size; ++col)
      {
	 matrix_j[col+row*block_size] = col;
      }
   }
   hypre_CSRMatrixCopy(bTy, cTy,  0);
   hypre_CSRMatrixCopy(bTy, yTBy, 0);

   HYPRE_ParVector   *B_pvx_H;
   mv_MultiVectorPtr B_eigenvectors_H = NULL;
   {
      B_eigenvectors_H = 
	 mv_MultiVectorCreateFromSampleVector(pase_data->interpreter_hypre, block_size, pase_data->x_hypre);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(B_eigenvectors_H);
      B_pvx_H = (HYPRE_ParVector*)(tmp -> vector);
   }

   HYPRE_BoomerAMGSetup(precond, pase_data->A_hypre, pase_data->x_hypre, pase_data->b_hypre);







   for ( level = num_levels-2; level >= 0; --level )
   {
      /* pvx_Hh (特征向量)  pvx(当前层解问题向量)  pvx_pre(是更新后的特征向量并插值到了更细层) */
      /*------------------------Create a preconditioner and solve the linear system-------------*/
      /* Now setup and solve! */
      linear_setup((void*)linear_solver, (void*)A_array[level], (void*)F_array[level], (void*)U_array[level]);
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 Matvec_matvec (NULL, eigenvalues[idx_eig], (void*)B_array[level], (void*)pvx[idx_eig], 0.0, (void*)F_array[level] );
	 linear_solve((void*)linear_solver, (void*)A_array[level], (void*)F_array[level], (void*)pvx[idx_eig]);
      }
      /* 重置parcsr_A_Hh和parcsr_B_Hh */
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[level],
	    A_array[level], pvx, 0, U_array[num_levels-1], U_array[level] );
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[level],
	    B_array[level], pvx, 0, U_array[num_levels-1], U_array[level] );

      /* 对角化 */
      {
	 /* 求解线性方程组 */
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    HYPRE_BoomerAMGSolve(precond, parcsr_A_Hh->A_H, parcsr_A_Hh->aux_hH[idx_eig], pvx_H[idx_eig]);
	 }
	 
	 /* bTy */
	 matrix_data = bTy->data;
	 for (row = 0; row < block_size; ++row)
	 {
	    for ( col = 0; col < block_size; ++col)
	    {
	       HYPRE_ParVectorInnerProd(parcsr_A_Hh->aux_hH[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	       parcsr_A_Hh->aux_hh->data[row*block_size+col] =
		  parcsr_A_Hh->aux_hh->data[row*block_size+col] - matrix_data[row*block_size+col];
	    }
	 }
	 /* cTy */
	 matrix_data = cTy->data;
	 for (row = 0; row < block_size; ++row)
	 {
	    for ( col = 0; col < block_size; ++col)
	    {
	       HYPRE_ParVectorInnerProd(parcsr_B_Hh->aux_hH[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] =
		  parcsr_B_Hh->aux_hh->data[row*block_size+col] - matrix_data[row*block_size+col];
	    }
	 }
	 for (row = 0; row < block_size; ++row)
	 {
	    for ( col = 0; col < block_size; ++col)
	    {
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] =
		  parcsr_B_Hh->aux_hh->data[row*block_size+col] - matrix_data[col*block_size+row];
	    }
	 }
	 /* By */
	 for (row = 0; row < block_size; ++row)
	 {
	    HYPRE_ParCSRMatrixMatvec ( 1.0, parcsr_B_Hh->A_H, pvx_H[row], 0.0, B_pvx_H[row] );
	    HYPRE_ParVectorAxpy(-1.0, B_pvx_H[row], parcsr_B_Hh->aux_hH[row]);
	 }
	 /* yTBy */
	 matrix_data = yTBy->data;
	 for (row = 0; row < block_size; ++row)
	 {
	    for ( col = 0; col < block_size; ++col)
	    {
	       HYPRE_ParVectorInnerProd(B_pvx_H[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] =
		  parcsr_B_Hh->aux_hh->data[row*block_size+col] + matrix_data[row*block_size+col];
	    }
	 }

	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	    HYPRE_ParVectorCopy(pvx_H[idx_eig], pvx_Hh[idx_eig]->b_H); 
	    pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }
      }
      /* 对角化 */

      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig],  0.0);
	 pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
      }


      parcsr_A_Hh->diag = 1;
      HYPRE_LOBPCGSolve(0, lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
      parcsr_A_Hh->diag = 0;

      /* 对角化 */
      {
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    for (row = 0; row < block_size; ++row)
	    {
	       HYPRE_ParVectorAxpy(-(pvx_Hh[idx_eig]->aux_h->data[row]), pvx_H[row], pvx_Hh[idx_eig]->b_H);
	    }
	 }
      }
      /* 对角化 */



      /* 将pvx_Hh插值到更细层 */
      if (level == 0)
      {
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    pvx_pre[idx_eig] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)U_array[0]);
	 }
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /*  将pvx_Hh插值到0层,  然后再插值到更细层 */
	    PASE_ParVectorGetParVector( Q_array[0],  block_size,  pvx,  pvx_Hh[idx_eig],  pvx_pre[idx_eig] );
	 }
      }
      else
      {
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    pvx_pre[idx_eig] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)U_array[level-1]);
	 }
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 将pvx_Hh插值到level层, 然后再插值到更细层 */
	    PASE_ParVectorGetParVector( Q_array[level], block_size, pvx, pvx_Hh[idx_eig], U_array[level] );
	    /* 生成当前网格下的特征向量 */
	    Matvec_matvec(NULL, 1.0, (void*)P_array[level-1], (void*)U_array[level], 0.0, (void*)pvx_pre[idx_eig] );
	 }
      }

      /* 释放这一层解问题向量的空间 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx[idx_eig]);
      }
      /* pvx_pre是当前层更新的特征向量 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 pvx[idx_eig] = pvx_pre[idx_eig];
      }
   }

   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
     eigenvectors[idx_eig] = pvx[idx_eig];
   }
   hypre_TFree(pvx);
   hypre_TFree(pvx_pre);


   HYPRE_BoomerAMGDestroy(precond);
   /* 对角化 */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(B_pvx_H[idx_eig]);
   }
   hypre_TFree(B_pvx_H);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(B_eigenvectors_H));
   hypre_TFree(B_eigenvectors_H);
   hypre_CSRMatrixDestroy(bTy);
   hypre_CSRMatrixDestroy(cTy);
   hypre_CSRMatrixDestroy(yTBy);

   return 0;
}

PASE_Int
pase_initialize( hypre_PASEData* data )
{
   (data->absolute_tol)            = 1.0e-06;
   (data->relative_tol)            = 1.0e-06;
   (data->max_iters)               = 10;
   (data->max_levels)              = 3;
   (data->verb_level)         = 0;

   (data->eigenvalues_history)     = utilities_FortranMatrixCreate();
   (data->residual_norms)          = utilities_FortranMatrixCreate();
   (data->residual_norms_history)  = utilities_FortranMatrixCreate();

   return 0;
}

PASE_Int
pase_clean( hypre_PASEData* data )
{
   utilities_FortranMatrixDestroy( data->eigenvalues_history );
   utilities_FortranMatrixDestroy( data->residual_norms );
   utilities_FortranMatrixDestroy( data->residual_norms_history );

   return 0;
}

PASE_Int
HYPRE_PASECreate( HYPRE_Solver* solver )
{
   hypre_PASEData *pase_data;

   pase_data = hypre_CTAlloc(hypre_PASEData,1);

   hypre_PASEDataInterpreterHYPYE(pase_data) = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   hypre_PASEDataInterpreterPASE(pase_data)  = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   hypre_PASEDataMatVecFnHYPRE(pase_data)    = hypre_CTAlloc(HYPRE_MatvecFunctions,1);
   hypre_PASEDataMatVecFnPASE(pase_data)     = hypre_CTAlloc(HYPRE_MatvecFunctions,1);

   HYPRE_ParCSRSetupInterpreter((pase_data->interpreter_hypre));
   PASE_ParCSRSetupInterpreter ((pase_data->interpreter_pase));

   HYPRE_ParCSRSetupMatvec((pase_data->matvec_fn_hypre));
   PASE_ParCSRSetupMatvec ((pase_data->matvec_fn_pase));


   pase_data->multi_grid = NULL;

   pase_data->linear_solver = NULL;
   pase_data->eigen_solver  = NULL;
   pase_data->precond       = NULL;

   pase_data->A_pase = NULL;
   pase_data->B_pase = NULL;
   pase_data->b_pase = NULL;
   pase_data->x_pase = NULL;
   pase_data->vecs_pase = NULL;
   pase_data->cons_pase = NULL;
   pase_data->mv_cons_pase = NULL;
   pase_data->mv_eigs_pase = NULL;

   pase_data->b_hypre = NULL;
   pase_data->x_hypre = NULL;
   pase_data->vecs_hypre = NULL;
   pase_data->cons_hypre = NULL;
   pase_data->mv_cons_hypre = NULL;
   pase_data->mv_eigs_hypre = NULL;


   pase_initialize( pase_data );

   *solver = (HYPRE_Solver)pase_data;

   return hypre_error_flag;
}

PASE_Int 
HYPRE_PASEDestroy( HYPRE_Solver solver )
{
   hypre_PASEData *pase_data = (hypre_PASEData*)solver;
   HYPRE_Int block_size, idx_eig;
   block_size = pase_data->block_size;

   if (pase_data) {
      PASE_MultiGridDestroy(pase_data->multi_grid);

      hypre_TFree(pase_data->interpreter_hypre);
      hypre_TFree(pase_data->interpreter_pase);
      hypre_TFree(pase_data->matvec_fn_hypre);
      hypre_TFree(pase_data->matvec_fn_pase);


      PASE_ParCSRMatrixDestroy(pase_data->A_pase);
      PASE_ParCSRMatrixDestroy(pase_data->B_pase);
      PASE_ParVectorDestroy(pase_data->b_pase);
      PASE_ParVectorDestroy(pase_data->x_pase);
      
      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorDestroy((pase_data->vecs_pase)[idx_eig]);
      }
      hypre_TFree(pase_data->vecs_pase);
      free((mv_TempMultiVector*) mv_MultiVectorGetData(pase_data->mv_eigs_pase));
      hypre_TFree(pase_data->mv_eigs_pase);

      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy((pase_data->vecs_hypre)[idx_eig]);
      }
      hypre_TFree(pase_data->vecs_hypre);
      free((mv_TempMultiVector*) mv_MultiVectorGetData(pase_data->mv_eigs_hypre));
      hypre_TFree(pase_data->mv_eigs_hypre);



      pase_clean( pase_data );
      hypre_TFree( solver );
   }

   return hypre_error_flag;
}

PASE_Int 
HYPRE_PASESetup( HYPRE_Solver  solver,
                 HYPRE_ParCSRMatrix A,
                 HYPRE_ParCSRMatrix B,
                 HYPRE_ParVector    b,
                 HYPRE_ParVector    x)
{
   hypre_PASEData *pase_data = (hypre_PASEData*)solver;

   PASE_MultiGridCreate(&(pase_data->multi_grid), pase_data->max_levels, A, B, b, x);

   return hypre_error_flag;
}

PASE_Int
HYPRE_PASESolve( HYPRE_Solver      solver, 
                 HYPRE_Int         block_size,
                 HYPRE_ParVector*  con, 
                 HYPRE_ParVector*  vec, 
                 HYPRE_Real*       val )
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;

   PASE_Int num_levels, N_H;
   PASE_Int maxit = pase_data->max_iters;

   PASE_MultiGrid multi_grid = pase_data->multi_grid;

   pase_data->block_size = block_size;

   utilities_FortranMatrix* lambda_history;
   utilities_FortranMatrix* residuals;
   utilities_FortranMatrix* residuals_history;
  
   /* 利用multi_grid生成pase_data中的pase矩阵向量以及特征向量 */
   num_levels = pase_MultiGridDataNumLevels(multi_grid);
   {
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, multi_grid->A_array[num_levels-1], 
	    NULL, NULL, NULL, &(pase_data->A_pase), multi_grid->U_array[num_levels-1], NULL); 
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, multi_grid->B_array[num_levels-1], 
	    NULL, NULL, NULL, &(pase_data->B_pase), multi_grid->U_array[num_levels-1], NULL); 
      N_H = hypre_ParCSRMatrixGlobalNumRows(multi_grid->A_array[num_levels-1]);
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(multi_grid->F_array[num_levels-1]);
      PASE_ParVectorCreate( MPI_COMM_WORLD,  N_H,  block_size,  NULL,   partitioning,  &(pase_data->b_pase));
      PASE_ParVectorCreate( MPI_COMM_WORLD,  N_H,  block_size,  NULL,   partitioning,  &(pase_data->x_pase));
      pase_data->mv_eigs_pase = 
	 mv_MultiVectorCreateFromSampleVector(pase_data->interpreter_pase, block_size, (pase_data->x_pase));
      mv_TempMultiVector* tmp =
	 (mv_TempMultiVector*) mv_MultiVectorGetData(pase_data->mv_eigs_pase);
      pase_data->vecs_pase = (PASE_ParVector*)(tmp -> vector);
   }
   /* 生成最粗空间上的HYPRE向量以及特征向量 */
   {
      pase_data->A_hypre = multi_grid->A_array[num_levels-1];
      pase_data->B_hypre = multi_grid->B_array[num_levels-1];
      pase_data->b_hypre = multi_grid->F_array[num_levels-1];
      pase_data->x_hypre = multi_grid->U_array[num_levels-1];

      pase_data->mv_eigs_hypre = 
	 mv_MultiVectorCreateFromSampleVector(pase_data->interpreter_hypre, block_size, (pase_data->x_hypre));
      mv_TempMultiVector* tmp =
	 (mv_TempMultiVector*) mv_MultiVectorGetData(pase_data->mv_eigs_hypre);
      pase_data->vecs_hypre = (HYPRE_ParVector*)(tmp -> vector);
   }

   lambda_history    = pase_data->eigenvalues_history;
   residuals         = pase_data->residual_norms;
   residuals_history = pase_data->residual_norms_history;

   utilities_FortranMatrixAllocateData( block_size, maxit + 1,	lambda_history );
   utilities_FortranMatrixAllocateData( block_size, 1,		residuals );
   utilities_FortranMatrixAllocateData( block_size, maxit + 1,	residuals_history );

   pase_FullMultiGridSolve ( solver, block_size, con, vec, val );

   pase_TwoGridSolve( solver, block_size, con, vec, val );


   HYPRE_LOBPCGDestroy(pase_data->eigen_solver);
   return hypre_error_flag;
}

HYPRE_Int 
HYPRE_PASESetTol( HYPRE_Solver solver, HYPRE_Real tol)
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;
   pase_data->absolute_tol = tol;

   return hypre_error_flag;
}
HYPRE_Int 
HYPRE_PASESetMaxLevels( HYPRE_Solver solver, HYPRE_Int max_levels)
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;
   pase_data->max_levels = max_levels;

   return hypre_error_flag;
}
HYPRE_Int 
HYPRE_PASESetAMG( HYPRE_Solver solver, HYPRE_Solver amg)
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;
   pase_MultiGridDataAMG(pase_data->multi_grid) = amg;

   return hypre_error_flag;
}


HYPRE_Int 
hypre_PASESetLinearSolver(void* solver,
      HYPRE_Int  (*linear)(void*, void*, void*, void*), 
      HYPRE_Int  (*linear_setup)(void*, void*, void*, void*),
      void* linear_solver)
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;
   pase_data->linear_solver  = linear_solver;
   pase_data->pase_functions.LinearSetup = linear_setup;
   pase_data->pase_functions.LinearSolve = linear;

   return hypre_error_flag;
}

HYPRE_Int 
hypre_PASESetPrecond(void* solver,
      HYPRE_Int  (*precond)(void*, void*, void*, void*), 
      HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
      void* linear_solver)
{
   hypre_PASEData* pase_data = (hypre_PASEData*)solver;
   pase_data->linear_solver  = linear_solver;
   pase_data->pase_functions.PrecondSetup = precond_setup;
   pase_data->pase_functions.PrecondSolve = precond;

   return hypre_error_flag;
}

HYPRE_Int 
HYPRE_PASESetLinearSolver( HYPRE_Solver solver, 
      HYPRE_PtrToSolverFcn linear_solve, 
      HYPRE_PtrToSolverFcn linear_setup, 
      HYPRE_Solver linear_solver
      )
{
   return( hypre_PASESetLinearSolver( (void *) solver, 
	    (HYPRE_Int (*)(void*,  void*,  void*,  void*))linear_solve, 
	    (HYPRE_Int (*)(void*,  void*,  void*,  void*))linear_setup, 
	    (void *) linear_solver ) );
}

HYPRE_Int 
HYPRE_PASESetPrecond( HYPRE_Solver solver, 
      HYPRE_PtrToSolverFcn precond_solve, 
      HYPRE_PtrToSolverFcn precond_setup, 
      HYPRE_Solver precond
      )
{
   return( hypre_PASESetLinearSolver( (void *) solver, 
	    (HYPRE_Int (*)(void*,  void*,  void*,  void*))precond_solve, 
	    (HYPRE_Int (*)(void*,  void*,  void*,  void*))precond_setup, 
	    (void *) precond) );
}


//void
//hypre_PASEOperatorA( void* pase_data,  void* x,  void* y )
//{
//   hypre_PASEData*  pase_data = (hypre_PASEData*)pcg_vdata;
//   HYPRE_MatvecFunctions* mv  = pase_data->matvec_fn_hypre;
//
//   (*(mv->Matvec))(NULL,  1.0,  pase_data->A,  x,  0.0,  y);
//
//}
