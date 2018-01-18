#include "pase.h"
#include "laplace.h"

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

int main (int argc, char *argv[])
{
   int idx_eig;
   int myid, num_procs;
   int n, block_size, max_levels;
   double tolerance;

   double start_time, end_time; 
   int global_time_index;


   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix     A,           B;
   HYPRE_IJVector     x,           b;
   HYPRE_ParCSRMatrix parcsr_A,    parcsr_B;
   HYPRE_ParVector    par_x,       par_b;

   /* 插值到细空间求解线性问题, 存储多向量 */
   HYPRE_ParVector   *eigenvectors;
   HYPRE_Real        *eigenvalues, *exact_eigenvalues;

   /* 求解器声明 */ 
   HYPRE_Solver pcg_solver, precond, pase_solver;

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

   /* Default problem parameters */
   n = 10;
   max_levels = 3;
   block_size = 1;
   tolerance  = 1E-8;

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
	 printf("  -n <n>           : problem size in each direction (default: 200)\n");
	 printf("  -block_size <n>  : eigenproblem block size (default: 10)\n");
	 printf("  -max_levels <n>  : max levels of AMG (default: 6)\n");
	 printf("  -tol        <n>  : max tolerance of cycle (default: 1.e-8)\n");
	 printf("\n");
      }

      if (print_usage)
      {
	 MPI_Finalize();
	 return (0);
      }
   }

   /*----------------------- Laplace矩阵和精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   n = LaplaceAB(MPI_COMM_WORLD, n, &A, &B, &x, &b, &parcsr_A, &parcsr_B, &par_x, &par_b);
   LaplaceEigenvalues(&exact_eigenvalues, block_size, n);

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   start_time = MPI_Wtime();

   eigenvectors = hypre_CTAlloc (HYPRE_ParVector, block_size);
//   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
//   {
//      eigenvectors[idx_eig] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)par_x);
//   }
   /* Create solver */
   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);

   HYPRE_PASECreate(&pase_solver);

   HYPRE_PASESetTol(pase_solver, tolerance);
   HYPRE_PASESetMaxLevels(pase_solver, max_levels);
   HYPRE_PASESetLinearSolver(pase_solver, (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPCGSolve, 
	 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPCGSetup, pcg_solver);

   HYPRE_ParCSRMatrixPrint(parcsr_B, "B");

   HYPRE_PASESetup(pase_solver, parcsr_A, parcsr_B, par_b, par_x);

   end_time = MPI_Wtime();

   start_time = MPI_Wtime();

   HYPRE_PASESolve( pase_solver, block_size, NULL, eigenvectors, eigenvalues);
//   HYPRE_PASESolve( pase_solver, block_size, NULL, eigenvectors, exact_eigenvalues);

   end_time = MPI_Wtime();


   double h, h2;
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      if (myid == 0)
      {
	 printf ( "%d : eig = %1.16e, error = %e\n", 
	       idx_eig, eigenvalues[idx_eig], (eigenvalues[idx_eig]-exact_eigenvalues[idx_eig]));
      }
   }




   HYPRE_ParCSRPCGDestroy(pcg_solver);
   HYPRE_PASEDestroy(pase_solver);

   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(eigenvectors[idx_eig]);
   }
   hypre_TFree(eigenvectors);

   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);

   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   hypre_EndTiming(global_time_index);
   hypre_PrintTiming("Solve phase times",  MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();

   /* Destroy amg */

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
