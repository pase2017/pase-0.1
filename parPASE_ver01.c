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
   
   Created:      2017.09.16

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
   /* 最细+辅助矩阵向量 */
   PASE_ParCSRMatrix parcsr_A_hh;
   PASE_ParCSRMatrix parcsr_B_hh;
   PASE_ParVector par_b_hh;
   PASE_ParVector par_x_hh;

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
   n = 33;
   block_size = 3;
   max_levels = 5;

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

   /*----------------------- Laplace精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   {
      int tmp_nn = (int) sqrt(block_size) + 1;
      exact_eigenvalues = hypre_CTAlloc (HYPRE_Real, tmp_nn*tmp_nn);
      for (i = 0; i < tmp_nn; ++i) 
      {
	 for (k = 0; k < tmp_nn; ++k) 
	 {
	    exact_eigenvalues[i*tmp_nn+k] = M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2));
	 }
      }
      qsort(exact_eigenvalues, tmp_nn*tmp_nn, sizeof(double), cmp);
   }


   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;


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
   
   /* TODO:
    * 创建矩阵在循环内部进行, 会有很多次创建和释放, 
    * 后期可以写一个SetAuxValues的子函数, 只修改aux_hh和aux_Hh
    * 而且最粗空间得到的特征向量正好可以作为aux_Hh的存储空间 */


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
   HYPRE_PCGSetMaxIter(pcg_solver, 1000); /* max iterations */
   HYPRE_PCGSetTol(pcg_solver, 1e-8); /* conv. tolerance */
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
	 int maxIterations = 100; /* maximum number of iterations */
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
	    mv_MultiVectorSetRandom (eigenvectors_H, lobpcgSeed);

	    HYPRE_LOBPCGCreate(interpreter_H, &matvec_fn_H, &lobpcg_solver);
	 }
	 else if (level == num_levels-3)
	 {
	    /* TODO:
	     * 这里不要动不动就把不用的删除, 平白造成很多麻烦.
	     * 应该最后再删除H的部分
	     * 可以作为A_Hh的aux_Hh, 即创建A_Hh时将aux_Hh赋予pvx_H
	     * 从存储大小上剩N_H*block_size的存储量, 那么释放的时候就要注意不要重复
	     * */
	    /* eigenvectors - create a multivector */
	    eigenvectors_Hh = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	    pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	    mv_MultiVectorSetRandom (eigenvectors_Hh, lobpcgSeed);

	    HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
	 }
	 
         if (level >= num_levels-3)
	 {
	    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
	    HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
	    HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
	    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);
	 }

	 if (level == num_levels-2)
	 {
	    /* 不明白这里(HYPRE_Matrix)的存在意义, 但又不能没有 */
	    HYPRE_LOBPCGSetup (lobpcg_solver, (HYPRE_Matrix)parcsr_A_H, (HYPRE_Vector)par_b_H, (HYPRE_Vector)par_x_H);
	    HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B_H, (HYPRE_Vector)par_x_H);
	    HYPRE_LOBPCGSolve (lobpcg_solver, constraints_H, eigenvectors_H, eigenvalues );
	 }
	 else {
	    PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
	    PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);
	    HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );
	 }

	 /* clean-up */
	 /* 这是只在num_levels-2 */
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
	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       double tmp_double;
	       HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[level], pvx[idx_eig], 0.0, F_array[level] );
	       HYPRE_ParVectorInnerProd (F_array[level], pvx[idx_eig], &eigenvalues[idx_eig]);
	       HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[level], pvx[idx_eig], 0.0, F_array[level] );
	       HYPRE_ParVectorInnerProd (F_array[level], pvx[idx_eig], &tmp_double);
	       eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
	       if (myid == 0)
	       {
		  printf ( "eig = %f, error = %f\n", 
			eigenvalues[idx_eig]/h2, fabs(eigenvalues[idx_eig]/h2-exact_eigenvalues[idx_eig]) );
	       }
	    }
	 }

	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
	 hypre_TFree(eigenvectors);
      }
   }

   if (num_levels == 2)
   {
      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx_H[idx_eig]);
      }
      hypre_TFree(pvx_H);
      free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H));
      hypre_TFree(eigenvectors_H);
      hypre_TFree(interpreter_H);
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
   /* Destroy PCG solver and preconditioner */
   HYPRE_ParCSRPCGDestroy(pcg_solver);
   HYPRE_BoomerAMGDestroy(precond);


   /* eigenvectors - create a multivector */
   eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[0]);
   /* eigenvectors - get a pointer */
   {
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      pvx_h = (HYPRE_ParVector*)(tmp -> vector);
   }

   /* TODO: 开始进行最细+aux的PCG和最粗+aux的LOBPCG
    * 迭代次数大于最大次数或者特征值之和不再变化, 迭代停止(参考LOBPCG的停机准则)
    * A_Hh不变, 继续更新aux部分
    * 构造A_hh矩阵, 辅助部分也是最细空间上的
    * 现在有pvx即线性方程组的解*/

   PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A, NULL,
	 parcsr_A, pvx, &parcsr_A_hh, U_array[0], U_array[0] );
   PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B, NULL,
	 parcsr_B, pvx, &parcsr_B_hh, U_array[0], U_array[0] );

   {
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(U_array[0]);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N, block_size, NULL, partitioning, &par_x_hh);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N, block_size, NULL, partitioning, &par_b_hh);
   }


   if (num_levels == 2)
   {
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[0],
	    A_array[0], pvx, &parcsr_A_Hh, U_array[num_levels-1], U_array[0] );
      PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[0],
	    B_array[0], pvx, &parcsr_B_Hh, U_array[num_levels-1], U_array[0] );
   }

   /* Create solver */
   PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_PCGSetMaxIter(pcg_solver, block_size); /* max iterations */
   HYPRE_PCGSetTol(pcg_solver, 1e-7); /* conv. tolerance */
   HYPRE_PCGSetTwoNorm(pcg_solver, 1); /* use the two norm as the stopping criteria */
   HYPRE_PCGSetPrintLevel(pcg_solver, 0); /* prints out the iteration info */
   HYPRE_PCGSetLogging(pcg_solver, 1); /* needed to get run info later */

   hypre_PCGSetup(pcg_solver, parcsr_A_hh, par_b_hh, par_x_hh);

   if (max_levels == 2)
   {
      HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
   }
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver, block_size);
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);
   HYPRE_LOBPCGSetTol(lobpcg_solver, 1.e-8);
   HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, 0);
   PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
   PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);


   HYPRE_Int iter, max_iter;
   max_iter = 3;

   printf ( "Iteration\n" );
   for (iter = 0; iter < max_iter; ++iter)
   {
      /* 使得pvx_h是特征向量 */
      {
	 HYPRE_ParVector* tmp;
	 tmp   = pvx_h; 
	 pvx_h = pvx;
	 pvx   = tmp;
      }
      /* 生成A_Hh */
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[0],
	    A_array[0], pvx_h, U_array[num_levels-1], U_array[0] );
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[0],
	    B_array[0], pvx_h, U_array[num_levels-1], U_array[0] );
      /* 求特征值问题 */
      HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );

      /* 投影到最细空间 */
//      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
//      {
//	 PASE_ParVectorGetParVector( Q_array[0], block_size, pvx_h, par_x_Hh, pvx[idx_eig] );
//      }

      /* 生成A_hh */
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_hh, block_size, NULL,
	    A_array[0], pvx_h, U_array[num_levels-1], U_array[0] );
      PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_hh, block_size, NULL,
	    B_array[0], pvx_h, U_array[num_levels-1], U_array[0] );

      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, Q_array[0], pvx_Hh[idx_eig]->b_H, 0.0, par_x_hh->b_H );
	 hypre_SeqVectorCopy(pvx_Hh[idx_eig]->aux_h, par_x_hh->aux_h);
	 PASE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], parcsr_B_hh, par_x_hh, 0.0, par_b_hh );
	 /* Now setup and solve! */
	 hypre_PCGSolve(pcg_solver, parcsr_A_hh, par_b_hh, par_x_hh);

	 PASE_ParVectorGetParVector( NULL, block_size, pvx_h, par_x_hh, pvx[idx_eig] );
      }
   }


   if (myid == 0)
   {
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 double tmp_double;
	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &tmp_double);
	 eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;
	 printf ( "eig = %f, error = %f\n", 
	       eigenvalues[idx_eig]/h2, fabs(eigenvalues[idx_eig]/h2-exact_eigenvalues[idx_eig]) );
      }
   }






   HYPRE_LOBPCGDestroy(lobpcg_solver);
   PASE_ParCSRPCGDestroy(pcg_solver);
   PASE_ParVectorDestroy(par_x_hh);
   PASE_ParVectorDestroy(par_b_hh);
   PASE_ParCSRMatrixDestroy(parcsr_A_hh);
   PASE_ParCSRMatrixDestroy(parcsr_B_hh);


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
      HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
   }
   hypre_TFree(pvx);
   hypre_TFree(pvx_h);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
   hypre_TFree(eigenvectors);




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
