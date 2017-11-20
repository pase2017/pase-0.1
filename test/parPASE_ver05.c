/*
   parPASE_ver01

   Interface:    Linear-Algebraic (IJ)

   Compile with: make parPASE_ver01

   Sample run:   parPASE_ver01 -block_size 20 -n 100 -max_levels 6 

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
   
   Created:      2017.09.21

   Author:       Li Yu (liyu@lsec.cc.ac.cn).
*/

#include "pase.h"


static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }

int main (int argc, char *argv[])
{
   int i, k, idx_eig;
   int myid, num_procs;
   int N, N_H, n;
   int block_size, max_levels;

   int ilower, iupper;
   int local_size, extra;

   double h, h2;
   double sum_error, residual, tmp_double;
   int level, num_levels;
   //   int time_index; 
   int global_time_index;



   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJMatrix B;
   HYPRE_ParCSRMatrix parcsr_B;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   /* 最粗矩阵向量 */
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
   HYPRE_ParVector* tmp_pvx;

   HYPRE_Real *eigenvalues;
   HYPRE_Real *exact_eigenvalues;

   /* -------------------------求解器声明---------------------- */ 
   HYPRE_Solver amg_solver, lobpcg_solver, pcg_solver, precond;

   /* Using AMG to get multilevel matrix */
   hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* P0P1P2  P1P2  P2 */
   hypre_ParCSRMatrix **Q_array;
   /* rhs and x */
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /* 最粗空间中的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_H = NULL;
   mv_MultiVectorPtr constraints_H  = NULL;
   mv_InterfaceInterpreter* interpreter_H;
   HYPRE_MatvecFunctions matvec_fn_H;

   /* 最粗+辅助空间中的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_Hh = NULL;
   mv_MultiVectorPtr constraints_Hh  = NULL;
   mv_InterfaceInterpreter* interpreter_Hh;
   HYPRE_MatvecFunctions matvec_fn_Hh;

   mv_MultiVectorPtr eigenvectors = NULL;

   /* PASE矩阵向量解释器 */
   mv_InterfaceInterpreter* interpreter;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
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

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;



   /*----------------------- Laplace精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   {
      int tmp_nn = (int) sqrt(block_size) + 3;
      exact_eigenvalues = hypre_CTAlloc (HYPRE_Real, tmp_nn*tmp_nn);
      for (i = 0; i < tmp_nn; ++i) 
      {
	 for (k = 0; k < tmp_nn; ++k) 
	 {
//	    exact_eigenvalues[i*tmp_nn+k] = M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2));
	    exact_eigenvalues[i*tmp_nn+k] = ( 4*sin( (i+1)*M_PI/(2*(n+1)) )*sin( (i+1)*M_PI/(2*(n+1)) ) 
		     + 4*sin( (k+1)*M_PI/(2*(n+1)) )*sin( (k+1)*M_PI/(2*(n+1)) ) );
	 }
      }
      qsort(exact_eigenvalues, tmp_nn*tmp_nn, sizeof(double), cmp);
   }




   /* Each processor knows only of its own rows - the range is denoted by ilower
      and iupper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;


   /* -------------------最细矩阵赋值------------------------ */

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &B);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(B, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);
   HYPRE_IJMatrixInitialize(B);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
      */
   {
      int nnz;
      double values[5];
      int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
	 nnz = 0;

	 /* The left identity block:position i-n */
	 if ((i-n)>=0)
	 {
	    cols[nnz] = i-n;
	    values[nnz] = -1.0;
	    nnz++;
	 }

	 /* The left -1: position i-1 */
	 if (i%n)
	 {
	    cols[nnz] = i-1;
	    values[nnz] = -1.0;
	    nnz++;
	 }

	 /* Set the diagonal: position i */
	 cols[nnz] = i;
	 values[nnz] = 4.0;
	 nnz++;

	 /* The right -1: position i+1 */
	 if ((i+1)%n)
	 {
	    cols[nnz] = i+1;
	    values[nnz] = -1.0;
	    nnz++;
	 }

	 /* The right identity block:position i+n */
	 if ((i+n)< N)
	 {
	    cols[nnz] = i+n;
	    values[nnz] = -1.0;
	    nnz++;
	 }

	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
      }
   }
   {
      int nnz;
      double values[5];
      int cols[5];
      for (i = ilower; i <= iupper; i++)
      {
	 nnz = 1;
	 cols[0] = i;
	 values[0] = 1.0;
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(B, 1, &nnz, &i, cols, values);
      }
   }
   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJMatrixAssemble(B);
   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

   /* Create sample rhs and solution vectors */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   /* -------------------------- 利用AMG生成各个层的矩阵------------------ */

   /* Create solver */
   HYPRE_BoomerAMGCreate(&amg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetInterpType(amg_solver, 0 );
   HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0 );
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
   HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);  /* maximum number of levels */
   //   HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);          /* G-S/Jacobi hybrid relaxation */
   //   HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);         /* uses C/F relaxation */
   //   HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);          /* Sweeeps on each level */
   //   HYPRE_BoomerAMGSetTol(amg_solver, 1e-7);             /* conv. tolerance */

   /* Now setup */
   HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, par_b, par_x);

   /* Get A_array, P_array, F_array and U_array of AMG */
   amg_data = (hypre_ParAMGData*) amg_solver;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   printf ( "The number of levels = %d\n", num_levels );

   B_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   Q_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);

   /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
   B_array[0] = parcsr_B;
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrix  *tmp_parcsr_mat;
      tmp_parcsr_mat = hypre_ParMatmul(B_array[level-1], P_array[level-1] ); 
      B_array[level] = hypre_ParTMatmul (P_array[level-1], tmp_parcsr_mat ); 

      /* 参考AMGSetup中分层矩阵的计算, 需要Owns的作用是释放空间时不要重复释放 */
      hypre_ParCSRMatrixRowStarts(B_array[level]) = hypre_ParCSRMatrixColStarts(B_array[level]);
      hypre_ParCSRMatrixOwnsRowStarts(B_array[level]) = 0;
      hypre_ParCSRMatrixOwnsColStarts(B_array[level]) = 0;
      if (num_procs > 1) hypre_MatvecCommPkgCreate(B_array[level]); 

      hypre_ParCSRMatrixDestroy(tmp_parcsr_mat);
   }
   /* P0P1P2  P1P2  P2    */
   Q_array[num_levels-2] = P_array[num_levels-2];
   for ( level = num_levels-3; level >= 0; --level )
   {
      Q_array[level] = hypre_ParMatmul(P_array[level], Q_array[level+1]); 
   }

   N_H = hypre_ParCSRMatrixGlobalNumRows(A_array[num_levels-1]);
   printf ( "The dim of the coarsest space is %d.\n", N_H );


   /* -----------------------特征向量存储成MultiVector--------------------- */

   /* define an interpreter for the HYPRE ParCSR interface */
   interpreter_H = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter_H);
   HYPRE_ParCSRSetupMatvec(&matvec_fn_H);

   /* define an interpreter for the PASE ParCSR interface */
   interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   PASE_ParCSRSetupInterpreter(interpreter_Hh);
   PASE_ParCSRSetupMatvec(&matvec_fn_Hh);

   /*TODO: 多余, 直接用interpreter_H就行 */
   interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter);


   {
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(F_array[num_levels-1]);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_x_Hh);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_b_Hh);
   }

   /* Create PCG solver */
   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_PCGSetMaxIter(pcg_solver, 100); /* max iterations */
   HYPRE_PCGSetTol(pcg_solver, 1e-15); /* conv. tolerance */
   HYPRE_PCGSetTwoNorm(pcg_solver, 1); /* use the two norm as the stopping criteria */
   HYPRE_PCGSetPrintLevel(pcg_solver, 0); /* print solve info */
   HYPRE_PCGSetLogging(pcg_solver, 1); /* needed to get run info later */

   /* Now set up the AMG preconditioner and specify any parameters */
   HYPRE_BoomerAMGCreate(&precond);
   HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
   HYPRE_BoomerAMGSetInterpType( precond, 0 );
   HYPRE_BoomerAMGSetPMaxElmts( precond, 0 );
   HYPRE_BoomerAMGSetCoarsenType(precond, 6);
   HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
   HYPRE_BoomerAMGSetNumSweeps(precond, 1);
   HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
   /* Set the PCG preconditioner */
   HYPRE_PCGSetPrecond(pcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
	 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

   /* 从粗到细, 最粗特征值, 细解问题  */
   for ( level = num_levels-2; level >= 0; --level )
   {
      /* pvx_Hh (特征向量)  pvx_h(上一层解问题向量)  pvx(当前层解问题向量)  */
      printf ( "Current level = %d\n", level );

      /* 最粗层时, 直接就是最粗矩阵, 否则生成Hh矩阵 */
      printf ( "Set A_Hh and B_Hh\n" );
      if ( level == num_levels-2 )
      {
	 par_x_H = U_array[num_levels-1];
	 par_b_H = F_array[num_levels-1];
	 parcsr_A_H = A_array[num_levels-1];
	 parcsr_B_H = B_array[num_levels-1];
      }
      else {
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
	 /* pvx_h是level+1层的特征向量 */
	 pvx_h = pvx;

	 if (level == num_levels-3)
	 {
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[level+1],
		  A_array[level+1], pvx_h, &parcsr_A_Hh, U_array[num_levels-1], U_array[level+1] );
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[level+1],
		  B_array[level+1], pvx_h, &parcsr_B_Hh, U_array[num_levels-1], U_array[level+1] );
	 } else {
	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[level+1],
		  A_array[level+1], pvx_h, U_array[num_levels-1], U_array[level+1] );
	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[level+1],
		  B_array[level+1], pvx_h, U_array[num_levels-1], U_array[level+1] );
	 }
      }

      /*------------------------Create a preconditioner and solve the eigenproblem-------------*/

      printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      /* LOBPCG eigensolver */
      {
	 int maxIterations = 1000; /* maximum number of iterations */
	 int pcgMode = 1;         /* use rhs as initial guess for inner pcg iterations */
	 int verbosity = 0;       /* print iterations info */
	 double tol = 1.e-8;     /* absolute tolerance (all eigenvalues) */
	 int lobpcgSeed = 775;    /* random seed */

	 if (myid != 0)
	    verbosity = 0;

	 /* 最开始par_x_H是最粗空间的大小, par_x_Hh是Hh空间的大小 */
	 if (level == num_levels-2)
	 {
	    /* eigenvectors - create a multivector */
	    eigenvectors_H = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, par_x_H);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H);
	    pvx_H = (HYPRE_ParVector*)(tmp -> vector);

	    HYPRE_LOBPCGCreate(interpreter_H, &matvec_fn_H, &lobpcg_solver);
	 }
	 else if (level == num_levels-3)
	 {
	    /* TODO:
	     * 可以作为A_Hh的aux_Hh, 即创建A_Hh时将aux_Hh赋予pvx_H
	     * 从存储大小上剩N_H*block_size的存储量, 那么释放的时候就要注意不要重复
	     * */
	    /* eigenvectors - create a multivector */
	    eigenvectors_Hh = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	    pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	    /* TODO: 用aux的同步错误 */
//	    mv_MultiVectorSetRandom (eigenvectors_Hh, lobpcgSeed);

	    HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
	 }

	 if (level >= num_levels-3)
	 {
	    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
	    /* TODO: 搞清楚这是什么意思, 即是否可以以pvx_Hh为初值进行迭代 */
	    HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
	    HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
	    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

	 }

	 if (level == num_levels-2)
	 {
	    mv_MultiVectorSetRandom (eigenvectors_H, lobpcgSeed);
	    /* TODO: 不明白这里(HYPRE_Matrix)的存在意义, 但又不能没有 */
	    HYPRE_LOBPCGSetup (lobpcg_solver, (HYPRE_Matrix)parcsr_A_H, (HYPRE_Vector)par_b_H, (HYPRE_Vector)par_x_H);
	    HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B_H, (HYPRE_Vector)par_x_H);
	    HYPRE_LOBPCGSolve (lobpcg_solver, constraints_H, eigenvectors_H, eigenvalues );
	 }
	 else {

	    for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	       pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
	    }
	    /* TODO: 这里的setup可以放只在num_levels-3中, Solve在所有 */
	    PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
	    PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);
	    HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );
	 }

	 /* clean-up */
	 /* 这是只在num_levels-2, 销毁H空间上求解特征值问题 */
	 if (level == num_levels-2)
	 {
	    HYPRE_LOBPCGDestroy(lobpcg_solver);
	 }
      }


      /*------------------------Create a preconditioner and solve the linear system-------------*/

      printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh\n" );
      /* PCG with AMG preconditioner */
      {

	 /* eigenvectors - create a multivector */
	 eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[level]);
	 /* eigenvectors - get a pointer */
	 {
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
	    pvx = (HYPRE_ParVector*)(tmp -> vector);
	 }

	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    if ( level == num_levels-2 )
	    {
	       HYPRE_ParCSRMatrixMatvec ( 1.0, Q_array[level], pvx_H[idx_eig], 0.0, pvx[idx_eig] );
	    }
	    else {
	       /* 对pvx_Hh进行处理 */
	       /* 利用最粗层的U将Hh空间中的H部分的向量投影到上一层
		* 因为需要上一层的全局基函数进行线性组合 */
	       /* 将pvx_Hh插值到level+1层 */
	       PASE_ParVectorGetParVector( Q_array[level+1], block_size, pvx_h, pvx_Hh[idx_eig], U_array[level+1] );
	       /* 生成当前网格下的特征向量 */
	       HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level], U_array[level+1], 0.0, pvx[idx_eig] );
	    }
	 }

	 /* Now setup and solve! */
	 HYPRE_ParCSRPCGSetup(pcg_solver, A_array[level], F_array[level], U_array[level]);

	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 生成右端项 y = alpha*A*x + beta*y */
	    HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[level], pvx[idx_eig], 0.0, F_array[level] );
	    HYPRE_ParCSRPCGSolve(pcg_solver, A_array[level], F_array[level], pvx[idx_eig]);
	 }

	 if (level == 0)
	 {
	    sum_error = 0;
	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[level], pvx[idx_eig], 0.0, F_array[level] );
	       HYPRE_ParVectorInnerProd (F_array[level], pvx[idx_eig], &eigenvalues[idx_eig]);
	       HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[level], pvx[idx_eig], 0.0, F_array[level] );
	       HYPRE_ParVectorInnerProd (F_array[level], pvx[idx_eig], &tmp_double);
	       eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
	       if (myid == 0)
	       {
		  printf ( "eig = %e, error = %e\n", 
			eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
		  sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
	       }
	    }

	    if (myid == 0)
	    {
	       printf ( "the sum of error = %e\n", sum_error );
	    }

	 }

	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
	 hypre_TFree(eigenvectors);
      }
   }

   /* 销毁Hh空间的特征向量(当只有一层时也对) */
   if ( num_levels > 2 )
   {
      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
      }
      hypre_TFree(pvx_h);
   }




   if (num_levels == 2)
   {
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[0],
	    A_array[0], pvx, &parcsr_A_Hh, U_array[num_levels-1], U_array[0] );
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[0],
	    B_array[0], pvx, &parcsr_B_Hh, U_array[num_levels-1], U_array[0] );

      eigenvectors_Hh = 
	 mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
      pvx_Hh = (PASE_ParVector*)(tmp -> vector);
      mv_MultiVectorSetRandom (eigenvectors_Hh, 775);

      HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 1000);
      HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);
      /* TODO: 这个残量是什么意思 */
      HYPRE_LOBPCGSetTol(lobpcg_solver, 1.e-8);
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, 0);
      PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
      PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);
   }

   /* eigenvectors - create a multivector */
   eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[0]);
   /* eigenvectors - get a pointer */
   {
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      pvx_h = (HYPRE_ParVector*)(tmp -> vector);
   }



   HYPRE_Solver pcg_solver_pase;

   /* Create solver */
   PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver_pase);
   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_PCGSetMaxIter(pcg_solver_pase, 10); /* max iterations */
   HYPRE_PCGSetTol(pcg_solver_pase, 1e-16); /* conv. tolerance */
   HYPRE_PCGSetTwoNorm(pcg_solver_pase, 1); /* use the two norm as the stopping criteria */
   HYPRE_PCGSetPrintLevel(pcg_solver_pase, 0); /* prints out the iteration info */
   HYPRE_PCGSetLogging(pcg_solver_pase, 1); /* needed to get run info later */

   /* 最细+辅助矩阵向量 */
   PASE_ParCSRMatrix* parcsr_A_hh;
   PASE_ParCSRMatrix* parcsr_B_hh;
   PASE_ParVector* par_b_hh;
   PASE_ParVector* par_x_hh;

   parcsr_A_hh = hypre_TAlloc(PASE_ParCSRMatrix, num_levels);
   parcsr_B_hh = hypre_TAlloc(PASE_ParCSRMatrix, num_levels);
   par_b_hh    = hypre_TAlloc(PASE_ParVector,    num_levels);
   par_x_hh    = hypre_TAlloc(PASE_ParVector,    num_levels);

   PASE_ParVector* pvx_pre;
   PASE_ParVector* pvx_cur;
   mv_MultiVectorPtr eigenvectors_cur = NULL;
   mv_MultiVectorPtr eigenvectors_pre = NULL;



   /* 基于上述运算, 得到最细空间的特征向量pvx */

   HYPRE_PCGSetMaxIter(pcg_solver, 10); /* max iterations */
   
   int num_conv = 0;
   int iter = 0;
//   while (sum_error > 1E-6)
   residual = 1.0;
   while (residual > 1E-8 || num_conv == block_size)
   {
      /* 从细到粗, 进行CG迭代 */
      printf ( "V-cycle from h to H\n" );
      for (level = 1; level < num_levels-1; ++level)
      {
	 printf ( "level = %d\n", level );

	 {
	    HYPRE_Int *partitioning;
	    partitioning = hypre_ParVectorPartitioning(U_array[level]);
	    PASE_ParVectorCreate( MPI_COMM_WORLD, U_array[level]->global_size, block_size, NULL, partitioning, &par_x_hh[level]);
	    PASE_ParVectorCreate( MPI_COMM_WORLD, U_array[level]->global_size, block_size, NULL, partitioning, &par_b_hh[level]);
	 }

	 /* 基于上一次得到的特征向量 */
	 if (level == 1)
	 {
	    {
	       HYPRE_Int *partitioning;
	       partitioning = hypre_ParVectorPartitioning(U_array[level-1]);
	       PASE_ParVectorCreate( MPI_COMM_WORLD, U_array[level-1]->global_size, block_size, NULL, partitioning, &par_x_hh[level-1]);
	       PASE_ParVectorCreate( MPI_COMM_WORLD, U_array[level-1]->global_size, block_size, NULL, partitioning, &par_b_hh[level-1]);
	    }

	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, A_array[level], P_array[level-1],
		  A_array[0], pvx, &parcsr_A_hh[level], U_array[level], U_array[0] );
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, B_array[level], P_array[level-1],
		  B_array[0], pvx, &parcsr_B_hh[level], U_array[level], U_array[0] );

	    eigenvectors_pre = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_hh[level-1]);
	    {
	       mv_TempMultiVector* tmp = 
		  (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre);
	       pvx_pre = (PASE_ParVector*)(tmp -> vector);
	    }
	    for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       PASE_ParVectorSetConstantValues(pvx_pre[idx_eig], 0.0);
	       pvx_pre[idx_eig]->aux_h->data[idx_eig] = 1.0;
	    }


	 } 
	 else 
	 {
	    PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, A_array[level], P_array[level-1],
		  parcsr_A_hh[level-1], pvx_pre, &parcsr_A_hh[level], U_array[level], par_x_hh[level-1] );
	    PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, B_array[level], P_array[level-1],
		  parcsr_B_hh[level-1], pvx_pre, &parcsr_B_hh[level], U_array[level], par_x_hh[level-1] );
	 }


	 /* 生成当前层特征向量的存储空间 */
	 eigenvectors_cur = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_hh[level]);
	 {
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_cur);
	    pvx_cur = (PASE_ParVector*)(tmp -> vector);
	 }


	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    HYPRE_ParCSRMatrixMatvecT ( 1.0, P_array[level-1], pvx_pre[idx_eig]->b_H, 0.0, pvx_cur[idx_eig]->b_H );
	    hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_cur[idx_eig]->aux_h);
//	    PASE_ParVectorSetConstantValues(pvx_cur[idx_eig], 0.0);
//	    pvx_cur[idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }


	 printf ( "Solve linear system A_hh x = lambda B_hh x using pase_pcg_solver.\n" );
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 生成右端项 y = alpha*A*x + beta*y */
	    PASE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], parcsr_B_hh[level], pvx_cur[idx_eig], 0.0, par_b_hh[level] );

	    hypre_PCGSetup(pcg_solver_pase, parcsr_A_hh[level], par_b_hh[level], par_x_hh[level]);
	    hypre_PCGSolve(pcg_solver_pase, parcsr_A_hh[level], par_b_hh[level], pvx_cur[idx_eig]);
	 }



	 HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 10);
	 PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_hh[level], par_b_hh[level], par_x_hh[level]);
	 PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_hh[level], par_x_hh[level]);
	 HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_cur, eigenvalues );

	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    HYPRE_ParCSRMatrixMatvec  ( 1.0, P_array[level-1], pvx_cur[idx_eig]->b_H, 0.0, U_array[level-1] );
	    for (i = level-1; i > 0; --i)
	    {
	       HYPRE_ParCSRMatrixMatvec  ( 1.0, P_array[i-1], U_array[i], 0.0, U_array[i-1] );
	    }

	    HYPRE_ParVectorCopy(U_array[0], pvx_h[idx_eig]);
	    for (k = 0; k < block_size; ++k)
	    {
	       /* y = a x + y */
	       HYPRE_ParVectorAxpy(pvx_cur[idx_eig]->aux_h->data[k], pvx[k], pvx_h[idx_eig] );
	    }
	 }

	 tmp_pvx = pvx;
	 pvx = pvx_h;
	 pvx_h = tmp_pvx;

	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorDestroy(pvx_pre[idx_eig]);
	 }
	 hypre_TFree(pvx_pre);
	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
	 hypre_TFree(eigenvectors_pre);

	 pvx_pre = pvx_cur;
	 eigenvectors_pre = eigenvectors_cur;
      }

      /* 生成A_Hh */
      printf ( "Set A_Hh and B_Hh\n" );
      if ( num_levels > 2 )
      {
	 PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_COMM_WORLD, parcsr_A_Hh, block_size, P_array[num_levels-2],
	       parcsr_A_hh[num_levels-2], pvx_pre, U_array[num_levels-1], par_x_hh[num_levels-2] );
	 PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_COMM_WORLD, parcsr_B_Hh, block_size, P_array[num_levels-2],
	       parcsr_B_hh[num_levels-2], pvx_pre, U_array[num_levels-1], par_x_hh[num_levels-2] );
	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    HYPRE_ParCSRMatrixMatvecT ( 1.0, P_array[num_levels-2], pvx_pre[idx_eig]->b_H, 0.0, pvx_cur[idx_eig]->b_H );
	    hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_Hh[idx_eig]->aux_h);
	 }
      }
      else {
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[0],
	       A_array[0], pvx, U_array[num_levels-1], U_array[0] );
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[0],
	       B_array[0], pvx, U_array[num_levels-1], U_array[0] );

	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	    pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }
      }





      /* 求特征值问题 */
      printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 1000);
      PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
      PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);
      HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );
      /* 所得eigenvectors_Hh如何得到细空间上的特征向量 */

      sum_error = 0;
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 if (myid == 0)
	 {
	    printf ( "eig = %e, error = %e\n", 
		  eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
	    sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
	 }
      }
      if (myid == 0)
      {
	 printf ( "the sum of error = %e\n", sum_error ); 
      }
      printf ( "--------------------------------------------------\n" );

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

      /* TODO: 这个与直接用LOBPCG求得的特征值稍有不同 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorGetParVector( Q_array[0], block_size, pvx, pvx_pre[idx_eig], pvx_h[idx_eig] );
      }


      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(pcg_solver, A_array[0], F_array[0], U_array[0]);

      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[0], F_array[0], pvx_h[idx_eig]);

	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[0], F_array[0], pvx_h[idx_eig]);
      }


      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &tmp_double);
	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
      }

      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[0], F_array[0], pvx_h[idx_eig]);

	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[0], F_array[0], pvx_h[idx_eig]);
      }


      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &tmp_double);
	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
      }

      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &tmp_double);

	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx_h[idx_eig], -eigenvalues[idx_eig], F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], F_array[0], &residual);
	 residual = sqrt(residual/tmp_double);
	 printf ( "residual = %e\n", residual );
      }





//      sum_error = 0;
//      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//      {
//	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
//	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &eigenvalues[idx_eig]);
//	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
//	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &tmp_double);
//	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
//	 if (myid == 0)
//	 {
//	    printf ( "eig = %e, error = %e\n", 
//		  eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
//	    sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
//	 }
//      }
//      if (myid == 0)
//      {
//	 printf ( "the sum of error = %e\n", sum_error ); 
//      }



//      printf ( "V-cycle from H to h\n" );
//      /* 矩阵保持不变, 只改变右端项 */
//      for (level = num_levels-2; level >= 1; --level)
//      {
//	 printf ( "level = %d\n", level );
//	 /* 生成当前层特征向量的存储空间 */
//	 eigenvectors_cur = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_hh[level]);
//	 {
//	    mv_TempMultiVector* tmp = 
//	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_cur);
//	    pvx_cur = (PASE_ParVector*)(tmp -> vector);
//	 }
//
//
//	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
//	 {
//	    HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level], pvx_pre[idx_eig]->b_H, 0.0, pvx_cur[idx_eig]->b_H );
//	    hypre_SeqVectorCopy(pvx_pre[idx_eig]->aux_h, pvx_cur[idx_eig]->aux_h);
////	    PASE_ParVectorSetConstantValues(pvx_cur[idx_eig], 0.0);
////	    pvx_cur[idx_eig]->aux_h->data[idx_eig] = 1.0;
//	 }
//
//	 HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 20);
//	 PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_hh[level], par_b_hh[level], par_x_hh[level]);
//	 PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_hh[level], par_x_hh[level]);
//	 HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_cur, eigenvalues );
//
//
//	 if (level < num_levels-2)
//	 {
//	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//	    {
//	       PASE_ParVectorDestroy(pvx_pre[idx_eig]);
//	    }
//	    hypre_TFree(pvx_pre);
//	    free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
//	    hypre_TFree(eigenvectors_pre);
//	 }
//
//	 pvx_pre = pvx_cur;
//	 eigenvectors_pre = eigenvectors_cur;
//      }
//
//
//      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//      {
//	 PASE_ParVectorGetParVector( P_array[0], block_size, pvx, pvx_pre[idx_eig], pvx_h[idx_eig] );
//      }
//
//      sum_error = 0;
//      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//      {
//	 if (myid == 0)
//	 {
//	    printf ( "eig = %e, error = %e\n", 
//		  eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
//	    sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
//	 }
//      }
//      if (myid == 0)
//      {
//	 printf ( "the sum of error = %e\n", sum_error ); 
//      }
//      printf ( "--------------------------------------------------\n" );





      if ( num_levels > 2 )
      {
//	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//	 {
//	    PASE_ParVectorDestroy(pvx_pre[idx_eig]);
//	 }
//	 hypre_TFree(pvx_pre);
//	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_pre));
//	 hypre_TFree(eigenvectors_pre);

	 PASE_ParVectorDestroy(par_x_hh[0]);
	 PASE_ParVectorDestroy(par_b_hh[0]);
	 for (level = 1; level < num_levels-1; ++level)
	 {
	    PASE_ParVectorDestroy(par_x_hh[level]);
	    PASE_ParVectorDestroy(par_b_hh[level]);
	    PASE_ParCSRMatrixDestroy(parcsr_A_hh[level]);
	    PASE_ParCSRMatrixDestroy(parcsr_B_hh[level]);
	 }

      }

//      sum_error = 0;
//      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//      {
//	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
//	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &eigenvalues[idx_eig]);
//	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
//	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &tmp_double);
//	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
//	 if (myid == 0)
//	 {
//	    printf ( "eig = %e, error = %e\n", 
//		  eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
//	    sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
//	 }
//      }
//      if (myid == 0)
//      {
//	 printf ( "the sum of error = %e\n", sum_error ); 
//      }


      tmp_pvx = pvx;
      pvx = pvx_h;
      pvx_h = tmp_pvx;

      ++iter;
   }


   printf ( "iter = %d\n", iter );

   sum_error = 0;
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
      HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &tmp_double);
      eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
      if (myid == 0)
      {
	 printf ( "eig = %1.16e, error = %e\n", 
	       eigenvalues[idx_eig], (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
	 sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
      }
   }
   if (myid == 0)
   {
      printf ( "the sum of error = %e\n", sum_error ); 
   }



   hypre_TFree(par_x_hh);
   hypre_TFree(par_b_hh);
   hypre_TFree(parcsr_A_hh);
   hypre_TFree(parcsr_B_hh);





   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
   }
   hypre_TFree(pvx_h);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
   hypre_TFree(eigenvectors);


   /* Destroy PCG solver and preconditioner */
   HYPRE_ParCSRPCGDestroy(pcg_solver);
   HYPRE_BoomerAMGDestroy(precond);

   HYPRE_ParCSRPCGDestroy(pcg_solver_pase);


   HYPRE_LOBPCGDestroy(lobpcg_solver);
//   PASE_ParCSRPCGDestroy(pcg_solver);



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

   PASE_ParCSRMatrixDestroy(parcsr_A_Hh);
   PASE_ParCSRMatrixDestroy(parcsr_B_Hh);

   PASE_ParVectorDestroy(par_x_Hh);
   PASE_ParVectorDestroy(par_b_Hh);


   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx[idx_eig]);
   }
   hypre_TFree(pvx);




   /* Clean up */
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrixDestroy(B_array[level]);
   }
   for ( level = 0; level < num_levels-2; ++level )
   {
      hypre_ParCSRMatrixDestroy(Q_array[level]);
   }
   hypre_TFree(B_array);
   hypre_TFree(Q_array);

   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Destroy amg_solver */
   HYPRE_BoomerAMGDestroy(amg_solver);
   hypre_TFree(interpreter_Hh);
   hypre_TFree(interpreter);

   hypre_EndTiming(global_time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();


   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
