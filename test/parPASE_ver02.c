/*
   parPASE_ver02

   Interface:    Linear-Algebraic (IJ)

   Compile with: make parPASE_ver02

   Sample run:   parPASE_ver01 -block_size 20 -n 100 -max_levels 6 -tol 1.e-8

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is PASE (Parallels Auxiliary Space Eigen-solver)
                 with LOBPCG and AMG preconditioner.

		 首先通过最粗+aux求特征值问题与细空间求解问题, 得到特征向量的好初值
		 然后从最细到最粗, 最细求解线性问题, 最粗+aux求解特征问题
   
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
      if ( fabs(array[pre_start]-array[start]) > fabs(array[pre_start])*min_gap )
      {
	 part[ nparts ] = start;
	 pre_start = start;
	 ++nparts;
      }
   }
   part[ nparts ] = size;
   return nparts;
}

static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }

int main (int argc, char *argv[])
{
   int i, k, idx_eig, begin_idx;
   int N, N_H;
   int myid, num_procs;

   int ilower, iupper;
   int local_size, extra;

   int level, num_levels;
   int global_time_index;

   double h, h2;
   double sum_error, tmp_double;
   double start_time, end_time; 

   /* 算法的各个参数 */
   int block_size = 20;        /* 特征值求解个数 */
   int max_levels = 5;         /* 最大网格层数 */
   int n          = 100;       /* 单维剖分个数 */
   int more       = 0;         /* 多算的特征值数 */
   int iter       = 0;         /* 迭代次数 */
   int num_conv   = 0;         /* 收敛个数 */
   int max_its    = 20;        /* 最大迭代次数 */
   int num_iters  = 0;         /* 迭代次数 */
   double residual    = 1.0;   /* 残量 */
   double tolerance   = 1E-8;  /* 最小残量 */
   double tol_lobpcg  = 1E-8;  /* 最小残量 */
   double tol_pcg     = 1E-8;  /* 最小残量 */
   double tol_amg     = 1E-11; /* 最小残量 */
   double min_gap     = 1E-3;  /* 不同特征值的相对最小距离 */

   /* ------------------------矩阵向量, 求解器---------------------- */ 

   /* 最细矩阵 */
   HYPRE_IJMatrix     A,            B;
   HYPRE_IJVector     x,            b;
   HYPRE_ParCSRMatrix parcsr_A,     parcsr_B;
   HYPRE_ParVector    par_x,        par_b;
   /* 最粗矩阵向量 */
   HYPRE_ParCSRMatrix parcsr_A_H,   parcsr_B_H;
   HYPRE_ParVector    par_x_H,      par_b_H;
   /* 最粗+辅助矩阵向量 */
   PASE_ParCSRMatrix  parcsr_A_Hh,  parcsr_B_Hh;
   PASE_ParVector     par_x_Hh,     par_b_Hh;
   /* 特征值求解, 存储多向量 */
   HYPRE_ParVector   *pvx_H;
   PASE_ParVector    *pvx_Hh;
   /* 插值到细空间求解线性问题, 存储多向量 */
   HYPRE_ParVector   *pvx_pre,     *pvx,    *tmp_pvx;
   /* 特征值 */
   HYPRE_Real        *eigenvalues, *exact_eigenvalues;
   /* AMG求解器, 生成多重网格 */ 
   HYPRE_Solver       amg; 
   /* AMG求解器求解线性问题, 特征值求解器求解特征值问题 */
   HYPRE_Solver       amg_solver,   lobpcg_solver;
   /* HYPRE_Solver的AMG预条件子 */
   HYPRE_Solver       precond;
   /* PASE_Solver lobpcg的pcg预条件子 */
   PASE_Solver        pcg_solver;

   /* ------------------------AMG各层矩阵向量---------------------- */ 

   /* Using AMG to get multilevel matrix */
   hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* 若共4层: Q0 = P0P1P2  Q1 = P1P2  Q2 = P2 */
   hypre_ParCSRMatrix **Q_array;
   /* rhs and x */
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /* ------------------------特征值问题相关的多向量以及矩阵向量操作解释器---------------------- */ 

   /* 最粗空间中的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_H = NULL;
   mv_MultiVectorPtr constraints_H  = NULL;
   mv_InterfaceInterpreter *interpreter_H;
   HYPRE_MatvecFunctions    matvec_fn_H;

   /* PASE 特征值问题的特征向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors_Hh = NULL;
   mv_MultiVectorPtr constraints_Hh  = NULL;
   mv_InterfaceInterpreter *interpreter_Hh;
   HYPRE_MatvecFunctions    matvec_fn_Hh;

   /* HYPRE 线性方程组的解向量和矩阵向量解释器 */
   mv_MultiVectorPtr eigenvectors     = NULL;
   mv_MultiVectorPtr eigenvectors_h   = NULL;
   mv_MultiVectorPtr tmp_eigenvectors = NULL;
   mv_InterfaceInterpreter *interpreter;

   /* -------------------------------------------------------------------- */

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
	 printf("  -block_size <n>  : eigenproblem block size (default: 10)\n");
	 printf("  -max_levels <n>  : max levels of AMG (default: 5)\n");
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
   more = (block_size<10)?(block_size):(10);
   block_size += more;

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /*----------------------- Laplace精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   /* 得到精确特征值 */
   {
      int tmp_nn = (int) sqrt(block_size) + 3;
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

   /* --------------------------------------------------------------- */

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   /* -------------------------- 利用AMG生成各个层的矩阵------------------ */
   start_time = MPI_Wtime();

   /* Create solver */
   HYPRE_BoomerAMGCreate(&amg);
   /* hypre_BoomerAMGSetup详细介绍各种参数表示的意义 */
   HYPRE_BoomerAMGSetPrintLevel (amg, 1);
   HYPRE_BoomerAMGSetInterpType (amg, 0);
   HYPRE_BoomerAMGSetPMaxElmts  (amg, 0);
   HYPRE_BoomerAMGSetCoarsenType(amg, 6);
   HYPRE_BoomerAMGSetMaxLevels  (amg, max_levels);
   /* 一般地, AMG第一层矩阵是原来的1/2, 之后都是1/4 */
   HYPRE_BoomerAMGSetup(amg, parcsr_A, par_b, par_x);
   /* 得到AMGData */
   amg_data = (hypre_ParAMGData*) amg;
   /* Get A_array, P_array, F_array and U_array of AMG */
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   /* 基于特征值的个数, 选择最粗层, 最粗层自由度>10倍特征值数 */
   for (i = num_levels-1; i >=0; --i)
   {
      N_H = hypre_ParCSRMatrixGlobalNumRows(A_array[i]);
      if ( N_H > block_size*10 )
      {
	 num_levels = i+1;
	 break;
      }
   }
   if (myid==0)
   {
      printf ( "The number of levels = %d\n", num_levels );
      printf ( "The dim of the coarsest space is %d.\n", N_H );
   }

   /* 生成广义特征值问题的B和各个层到最粗层的投影矩阵Q */
   B_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   Q_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
   B_array[0] = parcsr_B;
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrix  *tmp_parcsr_mat;
      tmp_parcsr_mat = hypre_ParMatmul (B_array[level-1], P_array[level-1] ); 
      B_array[level] = hypre_ParTMatmul(P_array[level-1], tmp_parcsr_mat ); 
      /* 参考AMGSetup中分层矩阵的计算, 需要Owns的作用是释放空间时不要重复释放 */
      hypre_ParCSRMatrixRowStarts(B_array[level]) = hypre_ParCSRMatrixColStarts(B_array[level]);
      hypre_ParCSRMatrixOwnsRowStarts(B_array[level]) = 0;
      hypre_ParCSRMatrixOwnsColStarts(B_array[level]) = 0;
      if (num_procs > 1) hypre_MatvecCommPkgCreate(B_array[level]); 
      hypre_ParCSRMatrixDestroy(tmp_parcsr_mat);
   }
   /* P0P1P2  P1P2  P2 */
   Q_array[num_levels-2] = P_array[num_levels-2];
   for ( level = num_levels-3; level >= 0; --level )
   {
      Q_array[level] = hypre_ParMatmul(P_array[level], Q_array[level+1]); 
   }

   /* -----------------------MultiVector存储特征向量--------------------- */

   /* define an interpreter for the HYPRE ParCSR interface */
   interpreter_H = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter_H);
   HYPRE_ParCSRSetupMatvec(&matvec_fn_H);
   /* define an interpreter for the PASE  ParCSR interface */
   interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   PASE_ParCSRSetupInterpreter(interpreter_Hh);
   PASE_ParCSRSetupMatvec(&matvec_fn_Hh);
   /* 这里特指HYPRE的线性问题解释器 */
   interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter);

   /* -----------------------PASE LOBPCG 预条件子---------------------- */

   PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
   HYPRE_PCGSetPrintLevel(pcg_solver, 0);       /* print solve info */
   HYPRE_PCGSetTwoNorm   (pcg_solver, 1);       /* use the two norm as the stopping criteria */
   HYPRE_PCGSetLogging   (pcg_solver, 1);       /* needed to get run info later */
   HYPRE_PCGSetMaxIter   (pcg_solver, 5);       /* max iterations */
   HYPRE_PCGSetTol       (pcg_solver, tol_pcg); /* conv. tolerance */

   /* -----------------------HYPRE 线性求解器---------------------- */

   HYPRE_BoomerAMGCreate(&amg_solver);
   HYPRE_BoomerAMGSetPrintLevel (amg_solver, 0);       /* print solve info + parameters */
   HYPRE_BoomerAMGSetInterpType (amg_solver, 0);
   HYPRE_BoomerAMGSetPMaxElmts  (amg_solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
   HYPRE_BoomerAMGSetRelaxType  (amg_solver, 3);       /* G-S/Jacobi hybrid relaxation */
   HYPRE_BoomerAMGSetRelaxOrder (amg_solver, 1);       /* uses C/F relaxation */
   HYPRE_BoomerAMGSetNumSweeps  (amg_solver, 1);       /* Sweeeps on each level */
   HYPRE_BoomerAMGSetMaxIter    (amg_solver, 10);
   HYPRE_BoomerAMGSetTol        (amg_solver, tol_amg); /*  conv. tolerance */

   /* -----------------------HYPRE LOBPCG 预条件子---------------------- */

   HYPRE_BoomerAMGCreate(&precond);
   HYPRE_BoomerAMGSetPrintLevel (precond, 0);   /* print amg solution info */
   HYPRE_BoomerAMGSetInterpType (precond, 0);
   HYPRE_BoomerAMGSetPMaxElmts  (precond, 0);
   HYPRE_BoomerAMGSetCoarsenType(precond, 6);
   HYPRE_BoomerAMGSetRelaxType  (precond, 6);   /* Sym G.S./Jacobi hybrid */
   HYPRE_BoomerAMGSetNumSweeps  (precond, 2);
   HYPRE_BoomerAMGSetTol        (precond, 0.0); /* conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter    (precond, 1);   /* do only one iteration! */

   end_time = MPI_Wtime();
   if (myid==0)
   {
      printf ( "Initialization time %f \n", end_time-start_time );
   }

   /* ----------------------------首次在粗网格上进行特征值问题的计算------------------------------ */
   start_time = MPI_Wtime();

   {
      /* 指针指向最粗层 */
      parcsr_A_H = A_array[num_levels-1];
      parcsr_B_H = B_array[num_levels-1];
      par_x_H    = U_array[num_levels-1];
      par_b_H    = F_array[num_levels-1];
      /* eigenvectors - create a multivector */
      {
	 eigenvectors_H = 
	    mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, par_x_H);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H);
	 pvx_H = (HYPRE_ParVector*)(tmp -> vector);
	 mv_MultiVectorSetRandom (eigenvectors_H, 775);
      }
      /* LOBPCG for HYPRE */
      HYPRE_LOBPCGCreate(interpreter_H, &matvec_fn_H, &lobpcg_solver);
      HYPRE_LOBPCGSetMaxIter         (lobpcg_solver, 50); /* 最粗层特征值问题的参数选取 */
      HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);  /* 求解残量方程的初值, 用残量本身 */
      HYPRE_LOBPCGSetPrintLevel      (lobpcg_solver, 0);
      HYPRE_LOBPCGSetTol             (lobpcg_solver, tol_lobpcg);
      HYPRE_LOBPCGSetRTol            (lobpcg_solver, tol_lobpcg);

      HYPRE_LOBPCGSetPrecond(lobpcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
	    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
      /* 这里(HYPRE_Matrix)的存在意义是变成一个空结构 */
      HYPRE_LOBPCGSetup (lobpcg_solver, (HYPRE_Matrix)parcsr_A_H, (HYPRE_Vector)par_b_H, (HYPRE_Vector)par_x_H);
      HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B_H, (HYPRE_Vector)par_x_H);
      HYPRE_LOBPCGSolve (0, lobpcg_solver, constraints_H, eigenvectors_H, eigenvalues );
      if (myid==0)
      {
	 printf ( "num_iters = %d for lobpcg in the coarsest mesh.\n", HYPRE_LOBPCGIterations(lobpcg_solver) );
      }
      HYPRE_LOBPCGDestroy(lobpcg_solver);
   }

   end_time = MPI_Wtime();
   if (myid==0)
   {
      printf ( "LOBPCG for the coarsest mesh time %f.\n", end_time-start_time );
   }

   /* ---------------------Full MultiGrid iterations------------------------------------------------------- */
   start_time = MPI_Wtime();

   {
      eigenvectors = 
	 mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[num_levels-2]);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      pvx = (HYPRE_ParVector*)(tmp -> vector);
   }
   /* 将特征向量插值到更细层num_levels-2 */
   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[num_levels-2], pvx_H[idx_eig], 0.0, pvx[idx_eig] );
   }
   /* 循环外创建par_x_Hh, par_b_Hh, 定义最粗+aux的pase向量 */
   {
      HYPRE_Int *partitioning;
      partitioning = hypre_ParVectorPartitioning(F_array[num_levels-1]);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_x_Hh);
      PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL,  partitioning, &par_b_Hh);
   }
   /* 从粗到细, 细解问题, 最粗特征值 */
   for ( level = num_levels-2; level >= 0; --level )
   {
      /* pvx_Hh (特征向量)  pvx(当前层解问题向量)  pvx_pre(是更新后的特征向量并插值到了更细层) */
      if (myid==0)
      {
	 printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh\n" );
      }
      HYPRE_BoomerAMGSetup(amg_solver, A_array[level], F_array[level], U_array[level]);
      for (idx_eig = 0; idx_eig < block_size-more; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec (eigenvalues[idx_eig], B_array[level], pvx[idx_eig], 0.0, F_array[level]);
	 HYPRE_BoomerAMGSolve     (amg_solver, A_array[level], F_array[level], pvx[idx_eig]);
	 HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iters);
	 if (myid==0)
	 {
	    printf ( "num_iters = %d for amg for %d eigenvale in level %d.\n", num_iters, idx_eig, level );
	 }
      }
      if (myid==0)
      {
	 printf ( "Update A_Hh B_Hh x_Hh\n" );
      }
      /* 最粗层且第一次迭代时, 生成Hh矩阵, 其它都是重置 */
      if ( level == num_levels-2 )
      {
	 PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_A_H, Q_array[level],
	       A_array[level], pvx, &parcsr_A_Hh, U_array[num_levels-1], U_array[level] );
	 PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, parcsr_B_H, Q_array[level],
	       B_array[level], pvx, &parcsr_B_Hh, U_array[num_levels-1], U_array[level] );
	 {
	    eigenvectors_Hh = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	    pvx_Hh = (PASE_ParVector*)(tmp -> vector);
	 }
	 /* LOBPCG for PASE */
	 HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
	 HYPRE_LOBPCGSetPrintLevel      (lobpcg_solver, 0);
	 HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, 1);
	 HYPRE_LOBPCGSetMaxIter         (lobpcg_solver, 10);
	 HYPRE_LOBPCGSetTol             (lobpcg_solver, tol_lobpcg);
	 HYPRE_LOBPCGSetRTol            (lobpcg_solver, tol_lobpcg);
	 /* LOBPCG 设置pcg为求解残量方程的线性求解器 */
	 PASE_LOBPCGSetPrecond(lobpcg_solver, (PASE_PtrToSolverFcn) PASE_ParCSRPCGSolve,
	       (PASE_PtrToSolverFcn) PASE_ParCSRPCGSetup, pcg_solver);
	 PASE_LOBPCGSetup (lobpcg_solver, parcsr_A_Hh, par_b_Hh, par_x_Hh);
	 PASE_LOBPCGSetupB(lobpcg_solver, parcsr_B_Hh, par_x_Hh);
      } 
      else
      {
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_A_Hh, block_size, Q_array[level],
	       A_array[level], pvx, 0, U_array[num_levels-1], U_array[level] );
	 PASE_ParCSRMatrixSetAuxSpace( MPI_COMM_WORLD, parcsr_B_Hh, block_size, Q_array[level],
	       B_array[level], pvx, 0, U_array[num_levels-1], U_array[level] );
      }
      /* 为特征向量赋予初值 */
      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	 pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
      }
      if (myid==0)
      {
	 printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      }
      PASE_LOBPCGSolve(0, lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
      /* 定义更细层的特征向量 */
      if (level == 0)
      {
	 eigenvectors_h = 
	    mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[0]);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
	 pvx_pre = (HYPRE_ParVector*)(tmp -> vector);
	 /* 将pvx_Hh插值到0层, 然后再插值到更细层 */
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorGetParVector( Q_array[0], block_size, pvx, pvx_Hh[idx_eig], pvx_pre[idx_eig] );
	 }
      }
      else
      {
	 eigenvectors_h = 
	    mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[level-1]);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
	 pvx_pre = (HYPRE_ParVector*)(tmp -> vector);
	 /* 将pvx_Hh插值到level层, 然后再插值到更细层 */
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    PASE_ParVectorGetParVector( Q_array[level], block_size, pvx, pvx_Hh[idx_eig], U_array[level] );
	    HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level-1], U_array[level], 0.0, pvx_pre[idx_eig] );
	 }
      }

      /* 释放这一层解问题向量的空间 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_ParVectorDestroy(pvx[idx_eig]);
      }
      hypre_TFree(pvx);
      free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
      hypre_TFree(eigenvectors);
      /* pvx_pre是当前层更新的特征向量 */
      pvx = pvx_pre;
      eigenvectors = eigenvectors_h;
   }

   /* 瑞利商 */
   sum_error = 0;
   for (idx_eig = 0; idx_eig < block_size-more; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec(1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd(F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
      HYPRE_ParCSRMatrixMatvec(1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd(F_array[0], pvx[idx_eig], &tmp_double);
      eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;

      HYPRE_ParCSRMatrixMatvec(1.0, A_array[0], pvx[idx_eig], -eigenvalues[idx_eig], F_array[0] );
      HYPRE_ParVectorInnerProd(F_array[0], F_array[0], &residual);
      residual = sqrt(residual/tmp_double);

      HYPRE_ParVectorScale(1/sqrt(tmp_double), pvx[idx_eig]);

      if (myid == 0)
      {
	 printf ( "%d : eig = %1.16e, error = %e, residual = %e\n", 
	       idx_eig, eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2, residual );
	 sum_error += fabs(eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2;
      }
   }
   if (myid == 0)
   {
      printf ( "the sum of error for eigenvalues = %e\n", sum_error ); 
   }

   end_time = MPI_Wtime();
   if (myid==0)
   {
      printf ( "Full MC time %f \n", end_time-start_time );
   }

   /* ---------------------------------Two Grid iterations---------------------------------------------- */
   start_time = MPI_Wtime();

   {
      eigenvectors_h = 
	 mv_MultiVectorCreateFromSampleVector(interpreter, block_size, U_array[0]);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_h);
      pvx_pre = (HYPRE_ParVector*)(tmp -> vector);
   }

   num_conv  = 0;
   begin_idx = num_conv;
   while (iter < max_its && num_conv < block_size-more)
   {
      /* 只从不收敛的特征值开始求解线性问题 */
      if (myid==0)
      {
	 printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh from %d\n", num_conv );
      }
      for (idx_eig = num_conv; idx_eig < block_size; ++idx_eig)
      {
	 /* 生成右端项 y = alpha*A*x + beta*y */
	 HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[0], pvx[idx_eig], 0.0, F_array[0] );
	 HYPRE_BoomerAMGSolve(amg_solver, A_array[0], F_array[0], pvx[idx_eig]);
	 HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iters);
	 if (myid==0)
	 {
	    printf ( "num_iters = %d for amg for %d eigenvale.\n", num_iters, idx_eig );
	 }
      }
      /* 这里设置了begin_idx这个参数, 使得特征子空间已整体收敛 */
      if (myid==0)
      {
	 printf ( "Update Aux Space from %d\n", begin_idx );
      }
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
      if (myid==0)
      {
	 printf ( "LOBPCG solve A_Hh U_Hh = lambda_Hh B_Hh U_Hh\n" );
      }
      PASE_LOBPCGSolve(num_conv, lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
      /* 将pvx_Hh插值到0层 */
      for (idx_eig = num_conv; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorGetParVector( Q_array[0], block_size, pvx, pvx_Hh[idx_eig], pvx_pre[idx_eig] );
      }
      if (myid==0)
      {
	 printf ( "compute residual and update eigenvalues\n" );
      }

      /* TODO:是否需要计算每个特征值的残量, 再利用activeMark进行标记, 使得对应的特征值再进行光滑 */
      begin_idx = num_conv;
      for (idx_eig = num_conv; idx_eig < block_size-more; ++idx_eig)
      {
	 HYPRE_ParCSRMatrixMatvec(1.0, B_array[0], pvx_pre[idx_eig], 0.0, F_array[0] );
	 HYPRE_ParVectorInnerProd(F_array[0], pvx_pre[idx_eig], &tmp_double);
	 HYPRE_ParCSRMatrixMatvec(1.0, A_array[0], pvx_pre[idx_eig], -eigenvalues[idx_eig], F_array[0] );
	 HYPRE_ParVectorInnerProd(F_array[0], F_array[0], &residual);
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
      if (num_conv > 0 && num_conv < block_size-more)
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
      if (myid == 0)
      {
	 printf ( "begin_idx = %d, num_conv = %d\n", begin_idx, num_conv );
      }

      ++iter;

      /* 交换特征向量指针 */
      tmp_pvx = pvx;
      pvx = pvx_pre;
      pvx_pre = tmp_pvx;

      tmp_eigenvectors = eigenvectors;
      eigenvectors = eigenvectors_h;
      eigenvectors_h = tmp_eigenvectors;
   }

   end_time = MPI_Wtime();
   if (myid==0)
   {
      printf ( "Two levels MC time %f \n", end_time-start_time );
      printf ( "Two levels MC num iter = %d\n", iter );
   }

   hypre_EndTiming(global_time_index);
   /* --------------------------------------------------------------------------------------- */

   sum_error = 0;
   for (idx_eig = 0; idx_eig < block_size-more; ++idx_eig)
   {
      HYPRE_ParCSRMatrixMatvec(1.0, A_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd(F_array[0], pvx[idx_eig], &eigenvalues[idx_eig]);
      HYPRE_ParCSRMatrixMatvec(1.0, B_array[0], pvx[idx_eig], 0.0, F_array[0] );
      HYPRE_ParVectorInnerProd(F_array[0], pvx[idx_eig], &tmp_double);
      eigenvalues[idx_eig] = eigenvalues[idx_eig] / tmp_double;

      HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[0], pvx[idx_eig], -eigenvalues[idx_eig], F_array[0] );
      HYPRE_ParVectorInnerProd (F_array[0], F_array[0], &residual);
      residual = sqrt(residual/tmp_double);

      HYPRE_ParVectorScale(1/sqrt(tmp_double), pvx[idx_eig]);

      if (myid == 0)
      {
	 printf ( "%d : eig = %1.16e, error = %e, residual = %e\n", 
	       idx_eig, eigenvalues[idx_eig]/h2, (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig])/h2, residual );
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

   if (myid==0)
   {
      int nparts, *part;
      part = hypre_CTAlloc (HYPRE_Int, block_size);
      nparts = partition(part, eigenvalues, block_size, min_gap);
      for (i = 0; i < block_size; ++i)
      {
	 printf ( "%d\n", part[i] );
	 if (part[i] == block_size)
	    break;
      }
      for (k = 0; k < nparts; ++k)
      {
	 for (i = part[k]; i < part[k+1]; ++i)
	 {
	    printf ( "%lf\t", eigenvalues[i]/h2 );
	 }
	 printf ( "\n" );
      }
      hypre_TFree(part);
   }
   
   /* --------------------------销毁各类------------------------- */
   /* Destroy solver */
   PASE_ParCSRPCGDestroy(pcg_solver);
   HYPRE_BoomerAMGDestroy(amg_solver);
   HYPRE_BoomerAMGDestroy(precond);
   HYPRE_LOBPCGDestroy(lobpcg_solver);

   /* 销毁level==num_levels-2层的特征向量 */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx_H[idx_eig]);
   }
   hypre_TFree(pvx_H);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_H));
   hypre_TFree(eigenvectors_H);
   hypre_TFree(interpreter_H);

   /* 销毁PASE向量和解释器 */
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

   /* 销毁Hh空间的特征向量(当只有一层时也对) */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx[idx_eig]);
      HYPRE_ParVectorDestroy(pvx_pre[idx_eig]);
   }
   hypre_TFree(pvx);
   hypre_TFree(pvx_pre);
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

   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Destroy amg */
   HYPRE_BoomerAMGDestroy(amg);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}

