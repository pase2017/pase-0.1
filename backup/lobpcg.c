/*
 * =====================================================================================
 *
 *       Filename:  lobpcg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年09月14日 13时39分07秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>

#include "pase.h"


int main (int argc, char *argv[])
{
   int i, k;
   int myid, num_procs;
   int N, n;
   int blockSize, maxLevels;

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
   /* 线性方程组求解 */
   HYPRE_ParVector* pvx;

   /* 最粗矩阵A_Hh */
   HYPRE_IJMatrix A_Hh;
   HYPRE_ParCSRMatrix parcsr_A_Hh;
   HYPRE_IJMatrix B_Hh;
   HYPRE_ParCSRMatrix parcsr_B_Hh;
   HYPRE_IJVector x_Hh;
   HYPRE_ParVector par_b_Hh;
   HYPRE_IJVector b_Hh;
   HYPRE_ParVector par_x_Hh;
   /* 特征值求解 */
   HYPRE_ParVector* pvx_Hh;
   /* 插值到细空间 */
   HYPRE_ParVector* pvx_h;

   /* -------------------------求解器声明---------------------- */ 
   HYPRE_Solver amg_solver, lobpcg_solver;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   printf("=============================================================\n" );
   printf("PASE (Parallels Auxiliary Space Eigen-solver), serial version\n"); 
   printf("Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
   printf("=============================================================\n" );

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   /* Default problem parameters */
   n = 33;
   blockSize = 3;
   maxLevels = 2;

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
	 else if ( strcmp(argv[arg_index], "-blockSize") == 0 )
	 {
	    arg_index++;
	    blockSize = atoi(argv[arg_index++]);
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
	 printf("  -blockSize <n>      : eigenproblem block size (default: 3)\n");
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
   /* Using AMG to get multilevel matrix */
   hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* rhs and x */
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /* Create solver */
   HYPRE_BoomerAMGCreate(&amg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetOldDefault(amg_solver);            /* Falgout coarsening with modified classical interpolaiton */
   HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);          /* G-S/Jacobi hybrid relaxation */
   HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);         /* uses C/F relaxation */
   HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);          /* Sweeeps on each level */
   HYPRE_BoomerAMGSetMaxLevels(amg_solver, maxLevels);  /* maximum number of levels */
   HYPRE_BoomerAMGSetTol(amg_solver, 1e-7);             /* conv. tolerance */

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

   /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
   B_array[0] = parcsr_B;
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrix  *tmp_parcsr_mat;
      tmp_parcsr_mat = hypre_ParTMatmul(P_array[level-1], B_array[level-1]); 
      B_array[level] = hypre_ParMatmul (tmp_parcsr_mat, P_array[level-1]); 
      hypre_ParCSRMatrixDestroy(tmp_parcsr_mat);
   }
   
   /* ---------------------创建A_Hh B_Hh-------------------------- */
   int N_H = hypre_ParCSRMatrixGlobalNumRows(A_array[num_levels-1]);

   int idx_eig; 
   /* eigenvalues - allocate space */
   double *eigenvalues = (double*) calloc( blockSize, sizeof(double) );
   /* 辅助空间 */
   double **aux;
   aux = calloc(blockSize, sizeof(double*));
   for (i = 0; i < blockSize; ++i)
   {
      aux[i] = calloc(blockSize, sizeof(double));
   }

   /* Laplace精确特征值 */
   int nn = (int) sqrt(blockSize) + 1;
   double *exact_eigenvalues = (double*) calloc(nn*nn, sizeof(double));
   for (i = 0; i < nn; ++i) 
   {
      for (k = 0; k < nn; ++k) 
      {
	 exact_eigenvalues[i*nn+k] = M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2));
      }
   }
   qsort(exact_eigenvalues, nn*nn, sizeof(double), cmp);

   /* -----------------------特征向量存储成MultiVector--------------------- */
   mv_MultiVectorPtr eigenvectors_Hh = NULL;
   mv_MultiVectorPtr constraints_Hh = NULL;
   mv_InterfaceInterpreter* interpreter_Hh;
   HYPRE_MatvecFunctions matvec_fn;
   /* define an interpreter for the ParCSR interface */
   interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
   HYPRE_ParCSRSetupInterpreter(interpreter_Hh);
   HYPRE_ParCSRSetupMatvec(&matvec_fn);


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

      if (level == num_levels-3)
      {
	 for ( idx_eig = 0; idx_eig < blockSize; ++idx_eig)
	 {
	    HYPRE_ParVectorDestroy(pvx_Hh[idx_eig]);
	 }
      }
      /* 最开始par_x_Hh是最粗空间的大小, 之后变成Hh空间的大小 */
      if (level >= num_levels-3)
      {
	 /* eigenvectors - create a multivector */
	 eigenvectors_Hh = 
	    mv_MultiVectorCreateFromSampleVector(interpreter_Hh, blockSize, par_x_Hh);
	 mv_TempMultiVector* tmp = 
	    (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	 pvx_Hh = (HYPRE_ParVector*)(tmp -> vector);
	 mv_MultiVectorSetRandom (eigenvectors_Hh, lobpcgSeed);
      }

      HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn, &lobpcg_solver);
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
      //	 HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
      HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

      /* use a preconditioner */
      //	 HYPRE_LOBPCGSetPrecond(lobpcg_solver,
      //	       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
      //	       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
      //	       precond);

      HYPRE_LOBPCGSetup(lobpcg_solver, (HYPRE_Matrix)parcsr_A_Hh,
	    (HYPRE_Vector)par_b_Hh, (HYPRE_Vector)par_x_Hh);
      HYPRE_LOBPCGSetupB(lobpcg_solver, (HYPRE_Matrix)parcsr_B_Hh,
	    (HYPRE_Vector)par_x_Hh);

      //	 time_index = hypre_InitializeTiming("LOBPCG Solve");
      //	 hypre_BeginTiming(time_index);

      HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );

      /* 
	 for ( idx_eig = 0; idx_eig < blockSize; ++idx_eig)
	 {
	 printf ( "eig = %lf, error = %lf\n", 
	 eigenvalues[idx_eig]/h2, fabs(eigenvalues[idx_eig]/h2-exact_eigenvalues[idx_eig]) );

	 }
	 */

      //	 hypre_EndTiming(time_index);
      //	 hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      //	 hypre_FinalizeTiming(time_index);
      //	 hypre_ClearTiming();

      /* clean-up */
      HYPRE_BoomerAMGDestroy(precond);
      HYPRE_LOBPCGDestroy(lobpcg_solver);
   }


   /* Clean up */
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrixDestroy(B_array[level]);
   }
   hypre_TFree(B_array);

   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);
   for (i = 0; i < blockSize; ++i)
   {
      hypre_TFree(aux[i]);
   }
   hypre_TFree(aux);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   HYPRE_IJMatrixDestroy(A_Hh);
   HYPRE_IJMatrixDestroy(B_Hh);
   HYPRE_IJVectorDestroy(b_Hh);
   HYPRE_IJVectorDestroy(x_Hh);

   /* Destroy amg_solver */
   HYPRE_BoomerAMGDestroy(amg_solver);
   hypre_TFree(interpreter_Hh);
   
   hypre_EndTiming(global_time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();


   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}

