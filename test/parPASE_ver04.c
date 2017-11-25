/*
   parPASE_ver03

   Interface:    Linear-Algebraic (IJ)

   Compile with: make parPASE_ver03

   Sample run:   parPASE_ver03 -block_size 20 -n 100 -max_levels 6 

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
int main (int argc, char *argv[])
{   
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   int myid, num_procs;	
   int i, k, idx_eig;

   int N, N_H, n;
   int block_size, max_levels;

   int ilower, iupper;
   int local_size, extra;

   double sum_error;
   int level, num_levels;
   int global_time_index;

   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix A;
   HYPRE_IJMatrix B;
   HYPRE_IJVector b;
   HYPRE_IJVector x;
   /* 只做指针用 */
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_ParCSRMatrix parcsr_B;
   HYPRE_ParVector par_b;
   HYPRE_ParVector par_x;

   /* 最粗矩阵向量，只做指针用 */
   HYPRE_ParCSRMatrix parcsr_A_H;
   HYPRE_ParCSRMatrix parcsr_B_H;
   HYPRE_ParVector par_b_H;
   HYPRE_ParVector par_x_H;
   /* 最粗+辅助矩阵向量 */
   PASE_ParCSRMatrix parcsr_A_Hh;
   PASE_ParCSRMatrix parcsr_B_Hh;
   PASE_ParVector par_b_Hh;
   PASE_ParVector par_x_Hh;
   /* 特征值求解, 存储多向量 */
   HYPRE_ParVector* pvx_H;
   PASE_ParVector*  pvx_Hh;
   /* 插值到细空间求解线性问题, 存储多向量 */
   HYPRE_ParVector* pvx_h;
   HYPRE_ParVector* pvx;

   /* 最细+辅助矩阵向量 */
   PASE_ParCSRMatrix* parcsr_A_hh;
   PASE_ParCSRMatrix* parcsr_B_hh;
   PASE_ParVector* par_b_hh;
   PASE_ParVector* par_x_hh;

   /* 特征值 */
   HYPRE_Real *eigenvalues;
   HYPRE_Real *exact_eigenvalues;

   /* V循环上下层特征向量, 存储多向量 */
   PASE_ParVector* pvx_pre;
   PASE_ParVector* pvx_cur;
   mv_MultiVectorPtr eigenvectors_cur = NULL;
   mv_MultiVectorPtr eigenvectors_pre = NULL;


   /* -------------------------求解器声明---------------------- */ 
   HYPRE_Solver lobpcg_solver, pcg_solver, precond;
   PASE_MultiGrid multi_grid;

   /* 最粗空间中的特征向量, HYPRE矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_H = NULL;
   mv_MultiVectorPtr constraints_H  = NULL;
   mv_InterfaceInterpreter* interpreter_H;
   HYPRE_MatvecFunctions matvec_fn_H;

   /* 最粗+辅助空间中的特征向量, PASE矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_Hh = NULL;
   mv_MultiVectorPtr constraints_Hh  = NULL;
   mv_InterfaceInterpreter* interpreter_Hh;
   HYPRE_MatvecFunctions matvec_fn_Hh;
   /* 细空间上的特征向量，随层数一直在变 */
   mv_MultiVectorPtr eigenvectors = NULL;

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   printf("=============================================================\n" );
   printf("PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
   printf("Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
   printf("=============================================================\n" );

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   /* Default problem parameters */
   n = 200;
   max_levels = 5;
   /* AMG第一层矩阵是原来的1/2, 之后都是1/4, 我们要求H空间的维数是所求特征值个数的8倍 */
   block_size = (int) n*n/pow(4, max_levels);
   printf ( "block_size = n*n/pow(4, max_levels)\n" );

   //   n = 10;
   //   max_levels = 3;
   //   block_size = 3;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
	 if ( strcmp(argv[arg_index], "-n") == 0 )
	 {
	    arg_index++;
	    n = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-block_size") == 0 )
	 {
	    arg_index++;
	    block_size = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-max_levels") == 0 )
	 {
	    arg_index++;
	    max_levels = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-help") == 0 )
	 {
	    print_usage = 1;
	    break;
	 }
	 else
	 {
	    arg_index++;
	 }
      }

      if ((print_usage) && (myid == 0))
      {
	 printf("\n");
	 printf("Usage: %s [<options>]\n", argv[0]);
	 printf("\n");
	 printf("  -n <n>              : problem size in each direction (default: 33)\n");
	 printf("  -block_size <n>      : eigenproblem block size (default: 3)\n");
	 printf("  -max_levels <n>      : max levels of AMG (default: 5)\n");
	 printf("\n");
      }

      if (print_usage)
      {
	 MPI_Finalize();
	 return (0);
      }
   }


   /*----------------------- Laplace特征值问题 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   CreateLaplaceEigenProblem(MPI_COMM_WORLD, 2, exact_eigenvalues, block_size, n, 
	 &A, &B, &x, &b);

   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

   /* Create sample rhs and solution vectors */
   HYPRE_IJVectorGetObject(b, (void **) &par_b);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   /* -------------------------- 利用AMG生成各个层的矩阵------------------ */
   PASE_MultiGridCreate(max_levels, parcsr_A, parcsr_B, par_x, par_b, &multi_grid);

   /* -----------------------特征向量存储成MultiVector--------------------- */
   /* define an interpreter for the HYPRE ParCSR interface */
   interpreter_H = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter_H);
   HYPRE_ParCSRSetupMatvec(&matvec_fn_H);

   /* define an interpreter for the PASE ParCSR interface */
   interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   PASE_ParCSRSetupInterpreter(interpreter_Hh);
   PASE_ParCSRSetupMatvec(&matvec_fn_Hh);

   /* 创建PASE向量, 以此一直存储随着层数变化而值变化的向量 */
   {
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(F_array[num_levels-1]);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_x_Hh);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_b_Hh);
   }

   /* Create PCG solver for HYPRE*/
   PASE_LinearSolverCreate(MPI_COMM_WORLD, &pcg_solver, &precond, "HYPRE");

   /* 从粗到细, 最粗特征值, 细解问题: level==num_levels-1为最粗网格, 0为最细网格 */
   for ( level = num_levels-2; level >= 0; --level )
   {
      /* pvx_Hh (特征向量)  pvx_h(上一层解问题向量)  pvx(当前层解问题向量)  */
      printf ( "Current level = %d\n", level );

      /* 最粗层时, 直接就是最粗矩阵, 否则生成Hh矩阵 */
      printf ( "Set A_Hh and B_Hh\n" );
      /* 最开始par_x_H是最粗空间的大小, par_x_Hh是Hh空间的大小 */

      /*------------------------Create a eigensolver and solve the eigenproblem-------------*/
      printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      if ( level == num_levels-2 )
      {
	 /* 指向最粗空间的矩阵和向量 */
	 parcsr_A_H = multi_grid->A_array[num_levels-1];
	 parcsr_B_H = multi_grid->B_array[num_levels-1];
	 par_x_H = multi_grid->U_array[num_levels-1];
	 par_b_H = multi_grid->F_array[num_levels-1];

	 /* eigenvectors - create a multivector */
	 {
	    eigenvectors_H = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, par_x_H);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H);
	    pvx_H = (HYPRE_ParVector*)(tmp -> vector);
	    mv_MultiVectorSetRandom (eigenvectors_H, 775);
	 }

	 PASE_EigenSolverCreate(interpreter_H, &matvec_fn_H, &lobpcg_solver);

	 /* 这里(HYPRE_Matrix)是一个空指针, 这样使得lobpcg可以的底层适用于任何矩阵向量结构 */
	 PASE_EigenSolverSetup(lobpcg_solver, (HYPRE_Matrix)parcsr_A_H, (HYPRE_Matrix)parcsr_B_H, 
	       (HYPRE_Vector)par_x_H, (HYPRE_Vector)par_b_H);
	 PASE_EigenSolverSolve(lobpcg_solver, constraints_H, eigenvectors_H, eigenvalues );
	 /* 这是只在num_levels-2, 销毁H空间上求解特征值问题 */
	 PASE_EigenSolverDestroy(lobpcg_solver);
      }
      else 
      {
	 /* 在循环外创建par_x_Hh, par_b_Hh */
	 /* 释放上一层解问题向量的空间 */
	 if (level < num_levels-3)
	 {
	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
	    }
	    hypre_TFree(pvx_h);
	 }
	 /* TODO: 这个赋值是否可以写在循环的最后面 */
	 /* pvx_h是level+1层的特征向量 */
	 pvx_h = pvx;

	 if (level == num_levels-3)
	 {
//	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[level+1],
//		  A_array[level+1], pvx_h, &parcsr_A_Hh, U_array[num_levels-1], U_array[level+1] );
//	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[level+1],
//		  B_array[level+1], pvx_h, &parcsr_B_Hh, U_array[num_levels-1], U_array[level+1] );
	    /* 生成A_HH是num_levels-1层的矩阵, h是level+1层的矩阵, 向量共有block_size个 */
	    PASE_EigenSystemCreate( muti_grid, num_levels-1, level+1,
		  block_size, pvx_h, &parcsr_A_Hh, &parcsr_B_Hh );

	    /* eigenvectors - create a multivector */
	    {
	       eigenvectors_Hh = 
		  mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	       mv_TempMultiVector* tmp = 
		  (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	       pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	       /* TODO: 这里对Hh进行Random时, 需要确保每个进程的aux要一致, 原程序需要改动 */
	       mv_MultiVectorSetRandom (eigenvectors_Hh, 775);
	    }

	    PASE_EigenSolverCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
	    PASE_EigenSolverSetup(lobpcg_solver, parcsr_A_Hh, parcsr_B_Hh, par_x_Hh, par_b_Hh );
	 } 
	 else 
	 {
//	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[level+1],
//		  A_array[level+1], pvx_h, U_array[num_levels-1], U_array[level+1] );
//	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[level+1],
//		  B_array[level+1], pvx_h, U_array[num_levels-1], U_array[level+1] );
	    PASE_EigenSystemSetAuxSpace( parcsr_A_Hh, parcsr_B_Hh, multi_grid, num_levels-1, level+1, 
		  block_size, pvx_h );
	 }

	 PASE_EigenSolverSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );
      }

      /*------------------------Create a linearsolver and solve the linear system-------------*/

      printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh\n" );
      /* PCG with AMG preconditioner, 已在循环外创建 */

      /* create a multivector and get a pointer */
      {
	 eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, U_array[level]);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
	 pvx = (HYPRE_ParVector*)(tmp -> vector);
      }

      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 if ( level == num_levels-2 )
	 {
	    /* 将最粗空间的特征向量插值到level层 */
	    HYPRE_ParCSRMatrixMatvec ( 1.0, Q_array[level], pvx_H[idx_eig], 0.0, pvx[idx_eig] );
	 }
	 else 
	 {
	    /* 利用最粗层的U将Hh空间中的H部分的向量投影到上一层, 因为需要上一层的全局基函数进行线性组合 */
	    /* 将pvx_Hh插值到level+1层 */
	    PASE_ParVectorGetParVector( multi_grid->Q_array[level+1], block_size, pvx_h, pvx_Hh[idx_eig], multi_grid->U_array[level+1] );
	    /* 生成当前网格下的特征向量 */
	    HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->P_array[level], multi_grid->U_array[level+1], 0.0, pvx[idx_eig] );
	 }
      }

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(pcg_solver, multi_grid->A_array[level], multi_grid->F_array[level], multi_grid->U_array[level]);
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], multi_grid->B_array[level], pvx[idx_eig], 0.0, multi_grid->F_array[level] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, multi_grid->A_array[level], multi_grid->F_array[level], pvx[idx_eig]);
      }

      /* 如若已到最细层，那么在最细层用Raylei商求特征值 */
      if (level == 0)
      {
	 sum_error = 0;
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    double tmp_double;
	    HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->A_array[level], pvx[idx_eig], 0.0, multi_grid->F_array[level] );
	    HYPRE_ParVectorInnerProd (multi_grid->F_array[level], pvx[idx_eig], &eigenvalues[idx_eig]);
	    HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->B_array[level], pvx[idx_eig], 0.0, multi_grid->F_array[level] );
	    HYPRE_ParVectorInnerProd (multi_grid->F_array[level], pvx[idx_eig], &tmp_double);
	    eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
	    if (myid == 0)
	    {
	       printf ( "eig = %e, error = %e\n", 
		     eigenvalues[idx_eig], 
		     fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]) );
	       sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]);
	    }
	 }

	 if (myid == 0)
	 {
	    printf ( "the sum of error = %e\n", sum_error );
	 }

      }

      free( (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors) );
      hypre_TFree(eigenvectors);

   }
   /* 上述循环返回pvx,最细层网格上的特征向量 */

   /* 销毁上一层的特征向量(当只有一层时也对) */
   if ( num_levels > 2 )
   {
      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
      }
      hypre_TFree(pvx_h);
   }

   /* Destroy PCG solver and preconditioner */
   PASE_LinearSolverDestroy(pcg_solver, precond);


   /* 若只有两层，则上述算法只做了最粗层的特征值问题和最细层的解问题, 并没有生成A_Hh矩阵 */
   /* 但会进行Hh矩阵的特征值问题的计算 */
   if (num_levels == 2)
   {
//      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[0],
//	    A_array[0], pvx, &parcsr_A_Hh, U_array[num_levels-1], U_array[0] );
//      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[0],
//	    B_array[0], pvx, &parcsr_B_Hh, U_array[num_levels-1], U_array[0] );
      PASE_EigenSystemCreate( muti_grid, num_levels-1, 0,
	    block_size, pvx, &parcsr_A_Hh, &parcsr_B_Hh );
      /* Hh空间上的特征向量 */
      {
	 eigenvectors_Hh = 
	    mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	 pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	 mv_MultiVectorSetRandom (eigenvectors_Hh, 775);
      }
      PASE_EigenSolverCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
      PASE_EigenSolverSetup(lobpcg_solver, parcsr_A_Hh, parcsr_B_Hh, par_x_Hh, par_b_Hh);
   }

   /* 创建最细空间上的多向量pvx_h, 以此为临时向量 */
   {
      eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, U_array[0]);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      pvx_h = (HYPRE_ParVector*)(tmp -> vector);
   }


   /* --------------------基于上述运算, 得到最细空间的特征向量pvx, 下面进行V循环-------------------------------- */
   /* lobpcg依然有效, 可以继续使用, 只是需要调整迭代次数 */

   /* 生成hh特征系统 */
   parcsr_A_hh = hypre_TAlloc(PASE_ParCSRMatrix, num_levels);
   parcsr_B_hh = hypre_TAlloc(PASE_ParCSRMatrix, num_levels);
   par_b_hh    = hypre_TAlloc(PASE_ParVector,    num_levels);
   par_x_hh    = hypre_TAlloc(PASE_ParVector,    num_levels);

   /* TODO: 第一次V循环后, 判断哪个特征值已经收敛, 那么在第二次V循环生成矩阵后, 不再进行收敛特征值的迭代 */
   {
      /* 从细到粗, 进行LOBPCG迭代 */
      printf ( "V-cycle from h to H\n" );

      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 2);
      /* pvx_pre应该是较细的向量, pvx_cur为较粗的向量 */
      for (level = 1; level < num_levels-1; ++level)
      {
	 printf ( "level = %d\n", level );

	 {
	    HYPRE_Int *partitioning;
	    partitioning = hypre_ParVectorPartitioning(multi_grid->U_array[level]);
	    PASE_ParVectorCreate( MPI_COMM_WORLD, multi_grid->U_array[level]->global_size, block_size, NULL, partitioning, &par_x_hh[level]);
	    PASE_ParVectorCreate( MPI_COMM_WORLD, multi_grid->U_array[level]->global_size, block_size, NULL, partitioning, &par_b_hh[level]);
	 }

	 /* 生成当前层特征向量的存储空间 */
	 {
	    eigenvectors_cur = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_hh[level]);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_cur);
	    pvx_cur = (PASE_ParVector*)(tmp -> vector);
	 }

	 /* 基于上一次得到的特征向量: 
	  * 若level==1则利用最细层矩阵生成PASE矩阵, 若level>1则利用上一层的PASE矩阵构造PASE矩阵 */
	 if (level == 1)
	 {
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, multi_grid->A_array[level], multi_grid->P_array[level-1],
		  multi_grid->A_array[0], pvx, &parcsr_A_hh[level], multi_grid->U_array[level], multi_grid->U_array[0] );
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, multi_grid->B_array[level], multi_grid->P_array[level-1],
		  multi_grid->B_array[0], pvx, &parcsr_B_hh[level], multi_grid->U_array[level], multi_grid->U_array[0] );

	    for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       PASE_ParVectorSetConstantValues(pvx_cur[idx_eig], 0.0);
	       pvx_cur[idx_eig]->aux_h->data[idx_eig] = 1.0;
	    }

	    PASE_EigenSolverSetup(lobpcg_solver, parcsr_A_hh[level], parcsr_B_hh[level], par_x_hh[level], par_b_hh[level] );
	    PASE_EigenSolverSolve(lobpcg_solver, constraints_Hh, eigenvectors_cur, eigenvalues );

	    pvx_pre = pvx_cur;
	    eigenvectors_pre = eigenvectors_cur;
	 }
	 else 
	 {
	    PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, multi_grid->A_array[level], multi_grid->P_array[level-1],
		  parcsr_A_hh[level-1], pvx_pre, &parcsr_A_hh[level], multi_grid->U_array[level], par_x_hh[level-1] );
	    PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, multi_grid->B_array[level], multi_grid->P_array[level-1],
		  parcsr_B_hh[level-1], pvx_pre, &parcsr_B_hh[level], multi_grid->U_array[level], par_x_hh[level-1] );

	    for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       /* TODO:测试原来的程序, 这里恐怕有错 */
	       /* 从细到粗 */
	       HYPRE_ParCSRMatrixMatvecT ( 1.0, multi_grid->P_array[level-1], pvx_pre[idx_eig]->b_H, 0.0, pvx_cur[idx_eig]->b_H );
	       hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_cur[idx_eig]->aux_h);
	    }

	    PASE_EigenSolverSetup(lobpcg_solver, parcsr_A_hh[level], parcsr_B_hh[level], 
		  par_x_hh[level], par_b_hh[level] );
	    PASE_EigenSolverSolve(lobpcg_solver, constraints_Hh, eigenvectors_cur, eigenvalues );

	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       PASE_ParVectorDestroy(pvx_pre[idx_eig]);
	    }
	    hypre_TFree(pvx_pre);
	    free( (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre) );
	    hypre_TFree(eigenvectors_pre);

	    pvx_pre = pvx_cur;
	    eigenvectors_pre = eigenvectors_cur;
	 }
      }

      /* 生成A_Hh B_Hh如若 */
      printf ( "Set A_Hh and B_Hh\n" );
      if ( num_levels > 2 )
      {
	 PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_COMM_WORLD, parcsr_A_Hh, block_size, multi_grid->P_array[num_levels-2],
	       parcsr_A_hh[num_levels-2], pvx_pre, multi_grid->U_array[num_levels-1], par_x_hh[num_levels-2] );
	 PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_COMM_WORLD, parcsr_B_Hh, block_size, multi_grid->P_array[num_levels-2],
	       parcsr_B_hh[num_levels-2], pvx_pre, multi_grid->U_array[num_levels-1], par_x_hh[num_levels-2] );

	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* TODO:测试原来的程序, 这里恐怕有错 */
	    /* 从细到粗 */
	    HYPRE_ParCSRMatrixMatvecT ( 1.0, multi_grid->P_array[num_levels-2], pvx_pre[idx_eig]->b_H, 0.0, pvx_Hh[idx_eig]->b_H );
	    hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_Hh[idx_eig]->aux_h);
	 }
      }
      else
      {
	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	    pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }
      }

      /* 求特征值问题 */
      printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 1000);
      PASE_EigenSolverSetup(lobpcg_solver, parcsr_A_Hh, parcsr_B_Hh, par_x_Hh, par_b_Hh);
      PASE_EigenSolverSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );
      /* 所得eigenvectors_Hh如何得到细空间上的特征向量 */


      //      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      //      {
      //	 printf ( "eig = %e, error = %e\n", 
      //	       eigenvalues[idx_eig], fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]) );
      //      }
      //      printf ( "--------------------------------------------------\n" );

      if ( num_levels > 2 )
      {
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorDestroy(pvx_pre[idx_eig]);
	 }
	 hypre_TFree(pvx_pre);
	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
	 hypre_TFree(eigenvectors_pre);
      }

      pvx_pre = pvx_Hh;
      eigenvectors_pre = eigenvectors_Hh;

      /* TODO: 这个与直接用LOBPCG求得的特征值稍有不同 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorGetParVector( multi_grid->Q_array[0], block_size, pvx, pvx_pre[idx_eig], pvx_h[idx_eig] );
      }
      sum_error = 0;
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 double tmp_double;
	 HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->A_array[0], pvx_h[idx_eig], 0.0, multi_grid->F_array[0] );
	 HYPRE_ParVectorInnerProd (multi_grid->F_array[0], pvx_h[idx_eig], &eigenvalues[idx_eig]);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->B_array[0], pvx_h[idx_eig], 0.0, multi_grid->F_array[0] );
	 HYPRE_ParVectorInnerProd (multi_grid->F_array[0], pvx_h[idx_eig], &tmp_double);
	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
	 if (myid == 0)
	 {
	    printf ( "eig = %e, error = %e\n", 
		  eigenvalues[idx_eig], fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]) );
	    sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]);
	 }
      }
      if (myid == 0)
      {
	 printf ( "the sum of error = %e\n", sum_error ); 
      }

      printf ( "V-cycle from H to h\n" );
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 2);
      /* 矩阵保持不变, 只改变右端项 */
      for (level = num_levels-2; level >= 1; --level)
      {
	 printf ( "level = %d\n", level );
	 /* 生成当前层特征向量的存储空间 */
	 {
	    eigenvectors_cur = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_hh[level]);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_cur);
	    pvx_cur = (PASE_ParVector*)(tmp -> vector);
	 }

	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 从粗到细 */
	    HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level], pvx_pre[idx_eig]->b_H, 0.0, pvx_cur[idx_eig]->b_H );
	    hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_cur[idx_eig]->aux_h);
	    //	    PASE_ParVectorSetConstantValues(pvx_cur[idx_eig], 0.0);
	    //	    pvx_cur[idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }

	 PASE_EigenSolverSetup (lobpcg_solver, parcsr_A_hh[level], parcsr_B_hh[level], par_x_hh[level], par_b_hh[level]);
	 PASE_EigenSolverSolve (lobpcg_solver, constraints_Hh, eigenvectors_cur, eigenvalues );

	 /* 这里不释放Hh的特征向量 */
	 if (level < num_levels-2)
	 {
	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       PASE_ParVectorDestroy(pvx_pre[idx_eig]);
	    }
	    hypre_TFree(pvx_pre);
	    free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
	    hypre_TFree(eigenvectors_pre);
	 }

	 pvx_pre = pvx_cur;
	 eigenvectors_pre = eigenvectors_cur;
      }


      /* 得到更新的最细空间的特征向量, 存储在pvx_h中 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorGetParVector( P_array[0], block_size, pvx, pvx_pre[idx_eig], pvx_h[idx_eig] );
      }

      if ( num_levels > 2 )
      {
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorDestroy(pvx_pre[idx_eig]);
	 }
	 hypre_TFree(pvx_pre);
	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
	 hypre_TFree(eigenvectors_pre);

      }


      /* TODO:做成迭代, 则需要将pvx释放, 然后pvx=pvx_h, 再进行V循环 */
   }


   for (level = 1; level < num_levels-1; ++level)
   {
      PASE_ParCSRMatrixDestroy(parcsr_A_hh[level]);
      PASE_ParCSRMatrixDestroy(parcsr_B_hh[level]);
      PASE_ParVectorDestroy(par_b_hh[level]);
      PASE_ParVectorDestroy(par_x_hh[level]);
   }
   hypre_TFree(par_x_hh);
   hypre_TFree(par_b_hh);
   hypre_TFree(parcsr_A_hh);
   hypre_TFree(parcsr_B_hh);


   sum_error = 0;
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      double tmp_double;
      HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->A_array[0], pvx_h[idx_eig], 0.0, multi_grid->F_array[0] );
      HYPRE_ParVectorInnerProd (multi_grid->F_array[0], pvx_h[idx_eig], &eigenvalues[idx_eig]);
      HYPRE_ParCSRMatrixMatvec ( 1.0, multi_grid->B_array[0], pvx_h[idx_eig], 0.0, multi_grid->F_array[0] );
      HYPRE_ParVectorInnerProd (multi_grid->F_array[0], pvx_h[idx_eig], &tmp_double);
      eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
      if (myid == 0)
      {
	 printf ( "eig = %e, error = %e\n", 
	       eigenvalues[idx_eig], fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]) );
	 sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]);
      }
   }


   if (myid == 0)
   {
      printf ( "the sum of error = %e\n", sum_error ); 
   }


   /* ----------------------------------销毁-------------------------------------------- */

   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
   }
   hypre_TFree(pvx_h);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
   hypre_TFree(eigenvectors);


   PASE_EigenSolverDestroy(lobpcg_solver);



   /* 销毁level==num_levels-2层的特征向量 */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_H[idx_eig]);
   }
   hypre_TFree(pvx_H);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H));
   hypre_TFree(eigenvectors_H);
   hypre_TFree(interpreter_H);



   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      PASE_ParVectorDestroy(pvx_Hh[idx_eig]);
   }
   hypre_TFree(pvx_Hh);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
   hypre_TFree(eigenvectors_Hh);
   hypre_TFree(interpreter_Hh);

   PASE_ParCSRMatrixDestroy(parcsr_A_Hh);
   PASE_ParCSRMatrixDestroy(parcsr_B_Hh);

   PASE_ParVectorDestroy(par_x_Hh);
   PASE_ParVectorDestroy(par_b_Hh);


   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx[idx_eig]);
   }
   hypre_TFree(pvx);


   hypre_TFree(eigenvalues);

   DestroyLaplaceEigenProblem(exact_eigenvalues, A, B, x, b);

   /* Destroy muti_grid */
   PASE_MultiGridDestroy(multi_grid);

   hypre_EndTiming(global_time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();


   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
