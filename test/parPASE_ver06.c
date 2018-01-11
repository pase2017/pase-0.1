/*
   parPASE_ver06

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

static int partition( HYPRE_Int *part,  HYPRE_Real *array, HYPRE_Int size, HYPRE_Real min_gap)
{   
   int nparts, start, pre_start;
   part[0] = 0;
   nparts = 1;
   pre_start = 0;
   for (start = 1; start < size; ++start)
   {
      if ( fabs(array[pre_start]-array[start]) > min_gap )
      {
	 part[ nparts ] = start;
	 pre_start = start;
	 ++nparts;
      }
   }
   part[ nparts ] = size;
   return nparts;
}


#ifdef PP
int main (int argc, char *argv[])
{
   int i, j;
   double array[6] = {-2, -2, 7, 7, 9, 10};
   int part[6], npart;
   npart = partition( part,  array, 6, 0.5);
   for (i = 0; i < 6; ++i)
   {
      printf ( "%d\n", part[i] );
      if (part[i] == 6)
	 break;
   }
   for (j = 0; j < npart; ++j)
   {
      for (i = part[j]; i < part[j+1]; ++i)
      {
	 printf ( "%f\t", array[i] );
      }
      printf ( "\n" );
   }
   return 0;
}
#endif

static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }

int main (int argc, char *argv[])
{
   int i, k, idx_eig, idx_block, idx;
   int myid, num_procs;
   int N, N_H, n;
   int block_size, max_levels, num_eigens;

   int ilower, iupper;
   int local_size, extra;

   double h, h2;
   double sum_error, tmp_double;
   int level, num_levels;
   //   int time_index; 
   int global_time_index;

   /* 算法的各个参数 */
   int more;/* 多算的特征值数 */
   int iter = 0;/* 迭代次数 */
   int num_conv = 0;/* 收敛个数 */
   int max_its = 5;/* 最大迭代次数 */
   double residual = 1.0;/* 残量 */
   double tolerance = 1E-8;/* 最小残量 */

   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix     A,           B;
   HYPRE_IJVector     x,           b;
   HYPRE_ParCSRMatrix parcsr_A,    parcsr_B;
   HYPRE_ParVector    par_x,       par_b;

   /* 最粗矩阵向量 */
   HYPRE_ParCSRMatrix parcsr_A_H,  parcsr_B_H;
   HYPRE_ParVector    par_x_H,     par_b_H;
   /* 最粗+辅助矩阵向量 */
   PASE_ParCSRMatrix  parcsr_A_Hh, parcsr_B_Hh;
   PASE_ParVector     par_x_Hh,    par_b_Hh;
   /* 特征值求解, 存储多向量 */
   HYPRE_ParVector   *pvx_H;
   PASE_ParVector    *pvx_Hh;
   /* 插值到细空间求解线性问题, 存储多向量 */
   HYPRE_ParVector   *pvx_h, *pvx,*tmp_pvx;

   HYPRE_Real        *tmp_eigenvalues, *eigenvalues, *exact_eigenvalues;

   /* -------------------------求解器声明---------------------- */ 
   HYPRE_Solver amg_solver, lobpcg_solver, pcg_solver, precond;

   /* Using AMG to get multilevel matrix */
   hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /*共4层: Q0 = P0P1P2  Q1 = P1P2  Q2 = P2 */
   hypre_ParCSRMatrix **Q_array;
   /* rhs and x */
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /* 最粗空间中的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_H = NULL;
   mv_MultiVectorPtr constraints_H  = NULL;
   mv_InterfaceInterpreter* interpreter_H;
   HYPRE_MatvecFunctions matvec_fn_H;

   /* PASE 特征值问题的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_Hh = NULL;
   mv_MultiVectorPtr constraints_Hh  = NULL;
   mv_InterfaceInterpreter* interpreter_Hh;
   HYPRE_MatvecFunctions matvec_fn_Hh;

   /* HYPRE 线性方程组的解向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors     = NULL;
   mv_MultiVectorPtr eigenvectors_h   = NULL;
   mv_MultiVectorPtr tmp_eigenvectors = NULL;
   mv_InterfaceInterpreter* interpreter;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (myid==0)
   {
      printf("=============================================================\n" );
      printf("PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
      printf("Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
      printf("=============================================================\n" );
   }

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   /* Default problem parameters */
   n = 100;
   max_levels = 4;
   num_eigens = 10;
   /* AMG第一层矩阵是原来的1/2, 之后都是1/4, 我们要求H空间的维数是所求特征值个数的8倍 */
   block_size = 3;

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
	 else if ( strcmp(argv[arg_index], "-num_eigens") == 0 )
	 {
	    arg_index++;
	    num_eigens = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-tol") == 0 )
	 {
	    arg_index++;
	    tolerance = atof(argv[arg_index++]);
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
	 printf("  -n <n>           : problem size in each direction (default: 100)\n");
	 printf("  -block_size <n>  : block size of auxiliary space  (default: 10)\n");
	 printf("  -num_eigens <n>  : number of eigenvalues   (default: 39)\n");
	 printf("  -max_levels <n>  : max levels of AMG (default: 4)\n");
	 printf("  -tol        <n>  : max tolerance of cycle (default: 1.e-8)\n");
	 printf("\n");
      }

      if (print_usage)
      {
	 MPI_Finalize();
	 return (0);
      }
   }

   /* 多算more个特征值 */
   more = num_eigens;
   num_eigens = (num_eigens / block_size + 2) * block_size; 
   more = num_eigens - more;

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /*----------------------- Laplace精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, num_eigens);
   tmp_eigenvalues = hypre_CTAlloc (HYPRE_Real, num_eigens);
   {
      int tmp_nn = (int) sqrt(num_eigens) + 3;
      exact_eigenvalues = hypre_CTAlloc (HYPRE_Real, tmp_nn*tmp_nn);
      for (i = 0; i < tmp_nn; ++i) 
      {
	 for (k = 0; k < tmp_nn; ++k) 
	 {
	    /* exact_eigenvalues[i*tmp_nn+k] = M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2)); */
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
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetInterpType(amg_solver, 0 );
   HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0 );
   /* hypre_BoomerAMGSetup中有 */
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
   HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);  /* maximum number of levels */

   /* Now setup */
   HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, par_b, par_x);

   /* Get A_array, P_array, F_array and U_array of AMG */
   amg_data = (hypre_ParAMGData*) amg_solver;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   if (myid==0)
   {
      printf ( "The number of levels = %d\n", num_levels );
   }

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
   if (myid==0)
   {
      printf ( "The dim of the coarsest space is %d.\n", N_H );
   }

   /* -----------------------特征向量存储成MultiVector--------------------- */

   /* define an interpreter for the HYPRE ParCSR interface */
   interpreter_H = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter_H);
   HYPRE_ParCSRSetupMatvec(&matvec_fn_H);

   /* define an interpreter for the PASE ParCSR interface */
   interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   PASE_ParCSRSetupInterpreter(interpreter_Hh);
   PASE_ParCSRSetupMatvec(&matvec_fn_Hh);

   /*TODO: 多余, 直接用interpreter_H就行, 当然这里特指HYPRE的PCG的解释器 */
   interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter);

   /* Create PCG solver */
   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   /* TODO:在进行pcg进行线性方程组求解时是否可以用到得到的precond, 至少level==0时可以, 
    * 是否可以不用precondition, 因为迭代8次都达到残差tol了 */
   HYPRE_PCGSetMaxIter(pcg_solver, 10); /* max iterations */
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

   /* TODO: 可以有Precondition */
   /* 首次在粗网格上进行特征值问题的计算 */
   {
      /* 指针指向最粗层 */
      par_x_H = U_array[num_levels-1];
      par_b_H = F_array[num_levels-1];
      parcsr_A_H = A_array[num_levels-1];
      parcsr_B_H = B_array[num_levels-1];

      /* eigenvectors - create a multivector */
      {
	 eigenvectors_H = 
	    mv_MultiVectorCreateFromSampleVector(interpreter_H, num_eigens, par_x_H);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H);
	 pvx_H = (HYPRE_ParVector*)(tmp -> vector);
      }
      /* LOBPCG for HYPRE */
      HYPRE_LOBPCGCreate(interpreter_H, &matvec_fn_H, &lobpcg_solver);

      /* 最粗层特征值问题的参数选取 */
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 100);
      /* TODO: 搞清楚这是什么意思, 即是否可以以pvx_Hh为初值进行迭代 */
      /* use rhs as initial guess for inner pcg iterations */
      HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);
      HYPRE_LOBPCGSetTol(lobpcg_solver, 1.e-8);
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, 0);

      mv_MultiVectorSetRandom (eigenvectors_H, 775);
      HYPRE_LOBPCGSetPrecond(lobpcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
	    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

      /* 这里(HYPRE_Matrix)的存在意义是变成一个空结构 */
      HYPRE_LOBPCGSetup (lobpcg_solver, (HYPRE_Matrix)parcsr_A_H, (HYPRE_Vector)par_b_H, (HYPRE_Vector)par_x_H);
      HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B_H, (HYPRE_Vector)par_x_H);
      HYPRE_LOBPCGSolve (lobpcg_solver, constraints_H, eigenvectors_H, eigenvalues );
      HYPRE_LOBPCGDestroy(lobpcg_solver);
   }

   for ( idx_eig=0; idx_eig < num_eigens; ++idx_eig)
   {
      printf ( "%d : eig = %1.16e, error = %e\n", idx_eig, 
	    eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2 );
   }

   {
      eigenvectors = 
	 mv_MultiVectorCreateFromSampleVector(interpreter, num_eigens, U_array[num_levels-2]);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      pvx = (HYPRE_ParVector*)(tmp -> vector);
   }

   /* 将特征向量插值到更细层num_levels-2 */
   for (idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[num_levels-2], pvx_H[idx_eig], 0.0, pvx[idx_eig] );
   }

   /* 定义最粗+aux的pase向量 */
   {
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(F_array[num_levels-1]);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_x_Hh);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_b_Hh);
   }

   /* 在循环外创建par_x_Hh, par_b_Hh */
   /* 从粗到细, 细解问题, 最粗特征值 */
   for ( level = num_levels-2; level >= 0; --level )
   {
      /*------------------------Create a preconditioner and solve the linear system-------------*/
      if (myid==0)
      {
	 printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh\n" );
      }
      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(pcg_solver, A_array[level], F_array[level], U_array[level]);
      for (idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[level], pvx[idx_eig], 0.0, F_array[level] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[level], F_array[level], pvx[idx_eig]);
      }

      /* pvx_Hh (特征向量)  pvx(当前层解问题向量)  pvx_h(是更新后的特征向量并插值到了更细层) */
      if (myid==0)
      {
	 printf ( "Current level = %d\n", level );
	 printf ( "Set A_Hh and B_Hh\n" );
      }
      /* 最粗层且第一次迭代时, 生成Hh矩阵, 其它都是重置 */

      if (level == 0)
      {
	 eigenvectors_h = 
	    mv_MultiVectorCreateFromSampleVector(interpreter, num_eigens, U_array[0]);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
	 pvx_h = (HYPRE_ParVector*)(tmp -> vector);
      }
      else
      {
	 eigenvectors_h = 
	    mv_MultiVectorCreateFromSampleVector(interpreter, num_eigens, U_array[level-1]);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
	 pvx_h = (HYPRE_ParVector*)(tmp -> vector);
      }


      for ( idx_block = 0; idx_block < num_eigens; idx_block += block_size)
      {
	 if ( level == num_levels-2 && idx_block == 0)
	 {
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[level],
		  A_array[level], &pvx[idx_block], &parcsr_A_Hh, U_array[num_levels-1], U_array[level] );
	    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[level],
		  B_array[level], &pvx[idx_block], &parcsr_B_Hh, U_array[num_levels-1], U_array[level] );
	    /* eigenvectors - create a multivector */
	    {
	       eigenvectors_Hh = 
		  mv_MultiVectorCreateFromSampleVector(interpreter_Hh, num_eigens, par_x_Hh);
	       mv_TempMultiVector* tmp = 
		  (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	       pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	    }
	    /* LOBPCG for PASE */
	    HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
	    /* 最粗+aux特征值问题的参数选取 */
	    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, 50);
	    /* TODO: 搞清楚这是什么意思, 即是否可以以pvx_Hh为初值进行迭代 */
	    /* use rhs as initial guess for inner pcg iterations */
	    HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);
	    HYPRE_LOBPCGSetTol(lobpcg_solver, 1.e-8);
	    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, 0);
	 } 
	 else
	 {
	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[level],
		  A_array[level], &pvx[idx_block], 0, U_array[num_levels-1], U_array[level] );
	    PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[level],
		  B_array[level], &pvx[idx_block], 0, U_array[num_levels-1], U_array[level] );
	 }

	 /* 对特征向量赋予初值 */
	 for ( idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
	 {
	    HYPRE_ParCSRMatrixMatvecT ( 1.0, P_array[num_levels-2], pvx[idx_eig], 0.0, pvx_Hh[idx_eig]->b_H );
	    hypre_SeqVectorSetConstantValues( pvx_Hh[idx_eig]->aux_h, 0.0 );
	 }
	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorSetConstantValues(pvx_Hh[idx_block+idx_eig], 0.0);
	    pvx_Hh[idx_block+idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }
	 /* LOBPCG eigensolver */
	 if (myid==0)
	 {
	    printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
	 }
	 PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
	 PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);

	 ((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh))->numVectors = idx_block + block_size;

	 HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, tmp_eigenvalues);

	 /* 定义更细层的特征向量 */
	 if (level == 0)
	 {
	    /* 判断所求得到的特征向量哪个是AUX里的 */
	    for ( idx_eig=0; idx_eig < block_size; ++idx_eig)
	    {
	       idx = idx_block+idx_eig;
	       for (k = (idx_block-block_size)>0?(idx_block-block_size):0; k < idx_block+block_size; ++k)
	       {
		  if ( fabs(pvx_Hh[k]->aux_h->data[idx_eig]) > fabs(pvx_Hh[idx]->aux_h->data[idx_eig]) )
		  {
		     idx = k;
		  }
	       }
	       eigenvalues[idx_block+idx_eig] = tmp_eigenvalues[idx];
	       /* 将pvx_Hh插值到0层, 然后再插值到更细层 */
	       PASE_ParVectorGetParVector( Q_array[0], block_size, &pvx[idx_block], pvx_Hh[idx], pvx_h[idx_block+idx_eig] );
	       printf ( "idx_block = %d, idx_eig = %d, idx = %d\n", idx_block, idx_eig, idx );
	    }
	 }
	 else
	 {
	    /* 判断所求得到的特征向量哪个是AUX里的 */
	    for ( idx_eig=0; idx_eig < block_size; ++idx_eig)
	    {
	       idx = idx_block+idx_eig;
//	       printf ( "%d: %f\n", idx_eig, tmp_eigenvalues[idx_eig]/h2 );
	       for (k = (idx_block-block_size)>0?(idx_block-block_size):0; k < idx_block+block_size; ++k)
	       {
//		  printf ( "(pvx_Hh[%d]->aux_h->data[%d]) = %f\n", k, idx_eig, (pvx_Hh[k]->aux_h->data[idx_eig]) );
		  if ( fabs(pvx_Hh[k]->aux_h->data[idx_eig]) > fabs(pvx_Hh[idx]->aux_h->data[idx_eig]) )
		  {
		     idx = k;
		  }
	       }
	       eigenvalues[idx_block+idx_eig] = tmp_eigenvalues[idx];
	       /* 将pvx_Hh插值到level层, 然后再插值到更细层 */
	       PASE_ParVectorGetParVector( Q_array[level], block_size, &pvx[idx_block], pvx_Hh[idx], U_array[level] );
	       /* 生成当前网格下的特征向量 */
	       HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level-1], U_array[level], 0.0, pvx_h[idx_block+idx_eig] );
	       printf ( "idx_block = %d, idx_eig = %d, idx = %d\n", idx_block, idx_eig, idx );
	    }
	 }

	 for ( idx_eig=0; idx_eig < block_size; ++idx_eig)
	 {
	    printf ( "%d : eig = %1.16e, error = %e\n", idx_eig+idx_block, 
		  eigenvalues[idx_block+idx_eig]/h2, (eigenvalues[idx_block+idx_eig]-exact_eigenvalues[idx_block+idx_eig])/h2 );
	 }
      }

      /* eigenvectors - create a multivector and get a pointer */
      /* 释放这一层解问题向量的空间 */
      for (idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx[idx_eig]);
      }
      hypre_TFree(pvx);
      free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
      hypre_TFree(eigenvectors);
      /* pvx_h是当前层更新的特征向量 */
      pvx = pvx_h;
      eigenvectors = eigenvectors_h;
   }

   {
      eigenvectors_h = 
	 mv_MultiVectorCreateFromSampleVector(interpreter, num_eigens, U_array[0]);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
      pvx_h = (HYPRE_ParVector*)(tmp -> vector);
   }

   HYPRE_ParCSRPCGSetup(pcg_solver, A_array[0], F_array[0], U_array[0]);

   printf ( "num_eigens = %d, more = %d, block_size = %d\n", num_eigens, more, block_size );

   /* pvx是上次迭代的特征向量 */
   while (iter < max_its && num_conv < num_eigens-more)
   {
      for ( idx_block = 0; idx_block < num_eigens; idx_block += block_size)
      {
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[0],
	       A_array[0], &pvx[idx_block], 0, U_array[num_levels-1], U_array[0] );
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[0],
	       B_array[0], &pvx[idx_block], 0, U_array[num_levels-1], U_array[0] );

	 for ( idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
	 {
	    HYPRE_ParCSRMatrixMatvecT ( 1.0, P_array[num_levels-2], pvx[idx_eig], 0.0, pvx_Hh[idx_eig]->b_H );
	    hypre_SeqVectorSetConstantValues( pvx_Hh[idx_eig]->aux_h, 0.0 );
	 }
	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorSetConstantValues(pvx_Hh[idx_block+idx_eig], 0.0);
	    pvx_Hh[idx_block+idx_eig]->aux_h->data[idx_eig] = 1.0;
	 }

	 /* LOBPCG eigensolver */
	 if (myid==0)
	 {
	    printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
	 }
	 PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
	 PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);

	 ((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh))->numVectors = idx_block + block_size;

	 HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, tmp_eigenvalues);

	 /* 判断所求得到的特征向量哪个是AUX里的 */
	 for ( idx_eig=0; idx_eig < block_size; ++idx_eig)
	 {
	    idx = idx_block+idx_eig;
	    for (k = (idx_block-block_size)>0?(idx_block-block_size):0; k < idx_block+block_size; ++k)
	    {
	       printf ( "(pvx_Hh[%d]->aux_h->data[%d]) = %f\n", k, idx_eig, (pvx_Hh[k]->aux_h->data[idx_eig]) );
	       if ( fabs(pvx_Hh[k]->aux_h->data[idx_eig]) > fabs(pvx_Hh[idx]->aux_h->data[idx_eig]) )
	       {
		  idx = k;
	       }
	    }
	    eigenvalues[idx_block+idx_eig] = tmp_eigenvalues[idx];
	    /* 将pvx_Hh插值到0层 */
	    PASE_ParVectorGetParVector( Q_array[0], block_size, &pvx[idx_block], pvx_Hh[idx], pvx_h[idx_block+idx_eig] );
	    printf ( "idx_block = %d, idx_eig = %d, idx = %d\n", idx_block, idx_eig, idx );
	 }

	 for ( idx_eig=0; idx_eig < block_size; ++idx_eig)
	 {
	    printf ( "%d : eig = %1.16e, error = %e\n", idx_eig+idx_block, 
		  eigenvalues[idx_block+idx_eig]/h2, (eigenvalues[idx_block+idx_eig]-exact_eigenvalues[idx_block+idx_eig])/h2 );
	 }
      }

      for (idx_eig = num_conv; idx_eig < num_eigens; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParCSRPCGSolve(pcg_solver, A_array[0], F_array[0], pvx_h[idx_eig]);
      }

      if (myid==0)
      {
	 printf ( "compute residual and update eigenvalues\n" );
      }
      for (idx_eig = num_conv; idx_eig < num_eigens-more; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx_h[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], pvx_h[idx_eig], &tmp_double);
	 HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx_h[idx_eig], -eigenvalues[idx_eig], F_array[0] );
	 HYPRE_ParVectorInnerProd (F_array[0], F_array[0], &residual);
	 residual = sqrt(residual/tmp_double);
	 /* TODO:收敛准则的科学性 */
	 if (residual < tolerance+tolerance*eigenvalues[idx_eig])
	 {
	    ++num_conv;
	 }
//	 printf ( "residual = %e\n", residual );
      }
      if (myid==0)
      {
	 printf ( "num_conv = %d\n", num_conv );
      }
      ++iter;

      tmp_pvx = pvx;
      pvx = pvx_h;
      pvx_h = tmp_pvx;

      tmp_eigenvectors = eigenvectors;
      eigenvectors = eigenvectors_h;
      eigenvectors_h = tmp_eigenvectors;
   }

   if (myid==0)
   {
      printf ( "iter = %d\n", iter );
   }

   hypre_EndTiming(global_time_index);

   sum_error = 0;
   for (idx_eig = 0; idx_eig < num_eigens-more; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
      HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], pvx[idx_eig], &tmp_double);
      eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;

      HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], -eigenvalues[idx_eig], F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], F_array[0], &residual);
      residual = sqrt(residual/tmp_double);

      if (myid == 0)
      {
	 printf ( "eig = %1.16e, error = %e, residual = %e\n", 
	       eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2, residual );
	 sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
      }
   }
   if (myid == 0)
   {
      printf ( "the sum of error for eigenvalues = %e\n", sum_error ); 
      printf ( "The dim of the refinest space is %d.\n", N );
      printf ( "The dim of the coarsest space is %d.\n", N_H );
      printf ( "The number of levels = %d\n", num_levels );
      printf ( "The number of iterations = %d\n", iter );
   }
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();

   /* Destroy PCG solver and preconditioner */
   HYPRE_ParCSRPCGDestroy(pcg_solver);
   HYPRE_BoomerAMGDestroy(precond);
   HYPRE_LOBPCGDestroy(lobpcg_solver);

   /* 销毁level==num_levels-2层的特征向量 */
   for ( idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_H[idx_eig]);
   }
   hypre_TFree(pvx_H);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H));
   hypre_TFree(eigenvectors_H);
   hypre_TFree(interpreter_H);

   /* 销毁PASE向量和解释器 */
   for ( idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
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


   /* 销毁Hh空间的特征向量(当只有一层时也对) */
   for ( idx_eig = 0; idx_eig < num_eigens; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx[idx_eig]);
      HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
   }
   hypre_TFree(pvx);
   hypre_TFree(pvx_h);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h));
   hypre_TFree(eigenvectors_h);
   hypre_TFree(eigenvectors);

   hypre_TFree(interpreter);

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

   hypre_TFree(tmp_eigenvalues);
   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Destroy amg_solver */
   HYPRE_BoomerAMGDestroy(amg_solver);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
