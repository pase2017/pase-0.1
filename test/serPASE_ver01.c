/*
   serPASE_ver01

   Interface:    Linear-Algebraic (IJ)

   Compile with: make serPASE_ver01

   Sample run:   serPASE_ver01 -block_size 20 -n 100 -max_levels 6 

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is PASE (Parallels Auxiliary Space Eigen-solver)
                 with LOBPCG and AMG preconditioner.
   
   Created:      2017.08.26

   Author:       Li Yu (liyu@lsec.cc.ac.cn).
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

/* lobpcg stuff */
#include "HYPRE_lobpcg.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"


#include "_hypre_parcsr_ls.h"


static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }




int main (int argc, char *argv[])
{
   int i, k;
   int myid, num_procs;
   int N, n;
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
   HYPRE_Solver amg_solver, precond, lobpcg_solver, pcg_solver;

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
   /* P0P1P2  P1P2  P2 */
   hypre_ParCSRMatrix **Q_array;
   /* (P0P1P2)^T*A0  (P1P2)^T*A1  (P2)^T*A2 */
   hypre_ParCSRMatrix **S_array;
   /* (P0P1P2)^T*M0  (P1P2)^T*M1  (P2)^T*M2 */
   hypre_ParCSRMatrix **T_array;
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
   HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);  /* maximum number of levels */
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
   Q_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   S_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   T_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);

   /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
   B_array[0] = parcsr_B;
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrix  *tmp_parcsr_mat;
      tmp_parcsr_mat = hypre_ParTMatmul(P_array[level-1], B_array[level-1]); 
      B_array[level] = hypre_ParMatmul (tmp_parcsr_mat, P_array[level-1]); 
      hypre_ParCSRMatrixDestroy(tmp_parcsr_mat);
   }
   /* P0P1P2  P1P2  P2    */
   Q_array[num_levels-2] = P_array[num_levels-2];
   for ( level = num_levels-3; level >= 0; --level )
   {
      Q_array[level] = hypre_ParMatmul(P_array[level], Q_array[level+1]); 
   }
   for ( level = 0; level < num_levels-1; ++level )
   {
      S_array[level] = hypre_ParTMatmul(Q_array[level], A_array[level]); 
      T_array[level] = hypre_ParTMatmul(Q_array[level], B_array[level]); 
   }
   
   /* ---------------------创建A_Hh B_Hh-------------------------- */
   int N_H = hypre_ParCSRMatrixGlobalNumRows(A_array[num_levels-1]);
   N = N_H+block_size;

   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_Hh);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &B_Hh);
   HYPRE_IJMatrixSetObjectType(A_Hh, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(B_Hh, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A_Hh);
   HYPRE_IJMatrixInitialize(B_Hh);

   /* Initialize A_Hh and B_Hh */
   {
      int nnz;
      double *values = calloc(N, sizeof(double));
      int *cols = calloc(N, sizeof(int));

      hypre_CSRMatrix* local_diag = hypre_ParCSRMatrixDiag(A_array[num_levels-1]);

      HYPRE_Complex *matrix_data = hypre_CSRMatrixData(local_diag);
      HYPRE_Int     *matrix_i    = hypre_CSRMatrixI(local_diag);
      HYPRE_Int     *matrix_j    = hypre_CSRMatrixJ(local_diag);
      HYPRE_Int      num_rows    = hypre_CSRMatrixNumRows(local_diag);

      /* 将A_H赋给A_Hh, 将B_H赋给B_Hh */
      for (i = 0; i < num_rows; ++i)
      {
	 nnz = matrix_i[i+1]-matrix_i[i];
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &i, &matrix_j[ matrix_i[i] ], &matrix_data[ matrix_i[i] ]);

	 nnz = block_size;
	 for (k = 0; k < nnz; ++k)
	 {
	    cols[k] = num_rows+k;
	    values[k] = 0.0;
	 }
	 HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &i, cols, values);
      }
      for (i = num_rows; i < N; ++i)
      {
	 nnz = N;
	 for (k = 0; k < N; ++k)
	 {
	    cols[k] = k;
	    values[k] = 0.0;
	 }
	 HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &i, cols, values);
      }
      hypre_TFree(values);
      hypre_TFree(cols);
   }
   {
      int nnz;
      double *values = calloc(N, sizeof(double));
      int *cols = calloc(N, sizeof(int));

      HYPRE_ParCSRMatrix B_H = B_array[num_levels-1];
      hypre_CSRMatrix* local_diag = hypre_ParCSRMatrixDiag(B_H);

      HYPRE_Complex *matrix_data = hypre_CSRMatrixData(local_diag);
      HYPRE_Int     *matrix_i    = hypre_CSRMatrixI(local_diag);
      HYPRE_Int     *matrix_j    = hypre_CSRMatrixJ(local_diag);
      HYPRE_Int      num_rows    = hypre_CSRMatrixNumRows(local_diag);

      for (i = 0; i < num_rows; ++i)
      {
	 nnz = matrix_i[i+1]-matrix_i[i];
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &i, &matrix_j[ matrix_i[i] ], &matrix_data[ matrix_i[i] ]);

	 nnz = block_size;
	 for (k = 0; k < nnz; ++k)
	 {
	    cols[k] = num_rows+k;
	    values[k] = 0.0;
	 }
	 HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &i, cols, values);
      }
      for (i = num_rows; i < N; ++i)
      {
	 nnz = N;
	 for (k = 0; k < N; ++k)
	 {
	    cols[k] = k;
	    values[k] = 0.0;
	 }
	 HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &i, cols, values);
      }
      hypre_TFree(values);
      hypre_TFree(cols);
   }
   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A_Hh);
   HYPRE_IJMatrixAssemble(B_Hh);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x_Hh);
   HYPRE_IJVectorSetObjectType(x_Hh, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x_Hh);
   HYPRE_IJVectorAssemble(x_Hh);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b_Hh);
   HYPRE_IJVectorSetObjectType(b_Hh, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b_Hh);
   HYPRE_IJVectorAssemble(b_Hh);

   int idx_eig; 
   /* eigenvalues - allocate space */
   double *eigenvalues = (double*) calloc( block_size, sizeof(double) );
   /* 辅助空间 */
   double **aux;
   aux = calloc(block_size, sizeof(double*));
   for (i = 0; i < block_size; ++i)
   {
      aux[i] = calloc(block_size, sizeof(double));
   }

   /* Laplace精确特征值 */
   int nn = (int) sqrt(block_size) + 1;
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




   /* 从粗到细, 最粗特征值, 细解问题  */
   for ( level = num_levels-2; level >= 0; --level )
   {
      /* pvx_Hh (特征向量)  pvx_h(上一层解问题向量)  pvx(当前层解问题向量)  */
      printf ( "Current level = %d\n", level );

      /* 最粗层时, 直接就是最粗矩阵, 否则生成Hh矩阵 */
      printf ( "Set A_Hh and B_Hh\n" );
      if ( level == num_levels-2 )
      {
	 par_x_Hh = U_array[num_levels-1];
	 par_b_Hh = F_array[num_levels-1];
	 parcsr_A_Hh = A_array[num_levels-1];
	 parcsr_B_Hh = B_array[num_levels-1];
      }
      else {
	 HYPRE_IJVectorGetObject(x_Hh, (void **) &par_x_Hh);
	 HYPRE_IJVectorGetObject(b_Hh, (void **) &par_b_Hh);

	 HYPRE_ParVector* tmp;
	 tmp = pvx_h; 
	 /* pvx_h是level+1层的特征向量 */
	 pvx_h = pvx;
	 /* 释放上一层解问题向量的空间 */
	 if (level < num_levels-3)
	 {
	    for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       HYPRE_ParVectorDestroy(tmp[idx_eig]);
	    }
	    hypre_TFree(tmp);
	 }
	 
	 /* N == A_Hh的大小 */
	 int nnz;
	 int row;
	 double *values;
	 int *cols = calloc(N, sizeof(int));

	 /* 更新A_Hh中的辅助空间 */
	 for (i = 0; i < N_H; ++i)
	 {
	    cols[i] = i;
	 }
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 将上一层解问题的向量乘以A, 再投影到最粗层 y = alpha*A*x + beta*y */
	    HYPRE_ParCSRMatrixMatvec ( 1.0, S_array[level+1], pvx_h[idx_eig], 0.0, U_array[num_levels-1] );
	    /* 重置给A_Hh对应的行和列 */
	    nnz = N_H; 
	    row = N_H+idx_eig;
	    values = hypre_VectorData( hypre_ParVectorLocalVector(U_array[num_levels-1]) );
	    HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &row, cols, values);

	    nnz = 1;
	    for (i = 0; i < N_H; ++i)
	    {
	       HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &i, &row, &values[i]);
	    }
	 }

	 for (i = 0; i < block_size; ++i)
	 {
	    cols[i] = N_H+i;
	 }
	 for (i = 0; i < block_size; ++i)
	 {
	    for (k = 0; k < block_size; ++k)
	    {
	       /* pvx_h A_array[level+1] pvx_h  不记得是aux[i][k] 还是反过来 */
	       HYPRE_ParCSRMatrixMatvec ( 1.0, A_array[level+1], pvx_h[k], 0.0, F_array[level+1] );
	       HYPRE_ParVectorInnerProd (pvx_h[i], F_array[level+1], &aux[i][k]);
	    }
	    /* 重置给A_Hh */
	    nnz = block_size; 
	    row = N_H+i;
	    HYPRE_IJMatrixSetValues(A_Hh, 1, &nnz, &row, cols, aux[i]);
	 }

	 /* 更新B_Hh中的辅助空间 */
	 for (i = 0; i < N_H; ++i)
	 {
	    cols[i] = i;
	 }
	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 将上一层解问题的向量乘以M, 再投影到最粗层 y = alpha*A*x + beta*y */
	    HYPRE_ParCSRMatrixMatvec ( 1.0, T_array[level+1], pvx_h[idx_eig], 0.0, U_array[num_levels-1] );
	    /* 重置给B_Hh */
	    nnz = N_H; 
	    row = N_H+idx_eig;
	    values = hypre_VectorData( hypre_ParVectorLocalVector(U_array[num_levels-1]) );
	    HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &row, cols, values);

	    nnz = 1;
	    for (i = 0; i < N_H; ++i)
	    {
	       HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &i, &row, &values[i]);
	    }
	 }

	 for (i = 0; i < block_size; ++i)
	 {
	    cols[i] = N_H+i;
	 }
	 for (i = 0; i < block_size; ++i)
	 {
	    for (k = 0; k < block_size; ++k)
	    {
	       /* pvx_h B_array[level+1] pvx_h  不记得是aux[i][k] 还是反过来 */
	       HYPRE_ParCSRMatrixMatvec ( 1.0, B_array[level+1], pvx_h[k], 0.0, F_array[level+1] );
	       HYPRE_ParVectorInnerProd (pvx_h[i], F_array[level+1], &aux[i][k]);
	    }
	    /* 重置给B_Hh */
	    nnz = block_size; 
	    row = N_H+i;
	    HYPRE_IJMatrixSetValues(B_Hh, 1, &nnz, &row, cols, aux[i]);
	 }

	 HYPRE_IJMatrixAssemble(A_Hh);
	 HYPRE_IJMatrixAssemble(B_Hh);
	 /* Get the parcsr matrix object to use */
	 HYPRE_IJMatrixGetObject(A_Hh, (void**) &parcsr_A_Hh);
	 HYPRE_IJMatrixGetObject(B_Hh, (void**) &parcsr_B_Hh);
	 hypre_TFree(cols);
      }

      /*------------------------Create a preconditioner and solve the eigenproblem-------------*/

      /* AMG preconditioner */
      {
	 HYPRE_BoomerAMGCreate(&precond);
	 HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
	 HYPRE_BoomerAMGSetNumSweeps(precond, 2); /* 2 sweeps of smoothing */
	 HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
	 HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
      }

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

	 /* 销毁level==num_levels-2层的特征向量 */
	 if (level == num_levels-3)
	 {
	    for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
	    {
	       HYPRE_ParVectorDestroy(pvx_Hh[idx_eig]);
	    }
	    hypre_TFree(pvx_Hh);
	    free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
	    hypre_TFree(eigenvectors_Hh);
	 }
	 /* 最开始par_x_Hh是最粗空间的大小, 之后变成Hh空间的大小 */
	 if (level >= num_levels-3)
	 {
	    /* eigenvectors - create a multivector */
	    eigenvectors_Hh = 
	       mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, par_x_Hh);
	    mv_TempMultiVector* tmp = 
	       (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
	    pvx_Hh = (HYPRE_ParVector*)(tmp -> vector);
	    mv_MultiVectorSetRandom (eigenvectors_Hh, lobpcgSeed);
	 }
	 
	 HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn, &lobpcg_solver);
	 HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
	 HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
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
	 for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
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


      /*------------------------Create a preconditioner and solve the linear system-------------*/

      printf ( "PCG solve A_h U = lambda_Hh B_h U_Hh\n" );
      /* PCG with AMG preconditioner */
      {
	 mv_MultiVectorPtr eigenvectors = NULL;
	 mv_InterfaceInterpreter* interpreter;
	 interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);
	 HYPRE_ParCSRSetupInterpreter(interpreter);

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
	       HYPRE_ParCSRMatrixMatvec ( 1.0, Q_array[level], pvx_Hh[idx_eig], 0.0, pvx[idx_eig] );
	    }
	    else {
	       double *v_H;
	       double *v_Hh;

	       /* 利用最粗层的U将Hh空间中的H部分的向量投影到上一层 */
	       v_H  = hypre_VectorData( hypre_ParVectorLocalVector(U_array[num_levels-1]) );

	       /* Q pvx_Hh[0:N_H-1] + alpha[0] pvx_h[0] + ... + alpha[block_size-1] pvx_h[block_size-1] */
	       v_Hh = hypre_VectorData( hypre_ParVectorLocalVector(pvx_Hh[idx_eig]) );
	       hypre_VectorData( hypre_ParVectorLocalVector(U_array[num_levels-1]) ) = v_Hh;
	       HYPRE_ParCSRMatrixMatvec ( 1.0, Q_array[level+1], U_array[num_levels-1], 0.0, U_array[level+1] );
	       for (k = 0; k < block_size; ++k)
	       {
		  /* y = a x + y */
		  HYPRE_ParVectorAxpy(v_Hh[N_H+k], pvx_h[k], U_array[level+1]);
	       }
	       /* 生成当前网格下的特征向量 */
	       HYPRE_ParCSRMatrixMatvec ( 1.0, P_array[level], U_array[level+1], 0.0, pvx[idx_eig] );

	       /* 还原U */
	       hypre_VectorData( hypre_ParVectorLocalVector(U_array[num_levels-1]) ) = v_H;
	    }
	 }

	 /* Create solver */
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
	 HYPRE_BoomerAMGSetCoarsenType(precond, 6);
	 HYPRE_BoomerAMGSetOldDefault(precond); 
	 HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
	 HYPRE_BoomerAMGSetNumSweeps(precond, 1);
	 HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
	 HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

	 /* Set the PCG preconditioner */
	 HYPRE_PCGSetPrecond(pcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
	       (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

	 /* Now setup and solve! */
	 HYPRE_ParCSRPCGSetup(pcg_solver, A_array[level], F_array[level], U_array[level]);

//	 time_index = hypre_InitializeTiming("PCG Solve");
//	 hypre_BeginTiming(time_index);

	 for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
	 {
	    /* 生成右端项 y = alpha*A*x + beta*y */
	    HYPRE_ParCSRMatrixMatvec ( eigenvalues[idx_eig], B_array[level], pvx[idx_eig], 0.0, F_array[level] );
	    HYPRE_ParCSRPCGSolve(pcg_solver, A_array[level], F_array[level], pvx[idx_eig]);
	 }

//	 hypre_EndTiming(time_index);
//	 hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
//	 hypre_FinalizeTiming(time_index);
//	 hypre_ClearTiming();

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
	       printf ( "eig = %f, error = %f\n", 
		     eigenvalues[idx_eig]/h2, fabs(eigenvalues[idx_eig]/h2-exact_eigenvalues[idx_eig]) );
	    }
	 }

	 /* Destroy solver and preconditioner */
	 HYPRE_ParCSRPCGDestroy(pcg_solver);
	 HYPRE_BoomerAMGDestroy(precond);

	 hypre_TFree(interpreter);
	 free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors));
	 hypre_TFree(eigenvectors);
      }
   }

   /* 销毁Hh空间的特征向量(当只有一层时也对) */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(pvx[idx_eig]);
      HYPRE_ParVectorDestroy(pvx_Hh[idx_eig]);
      if ( num_levels > 2 )
      {
	 HYPRE_ParVectorDestroy(pvx_h[idx_eig]);
      }
   }
   if ( num_levels > 2 )
   {
      hypre_TFree(pvx_h);
   }
   hypre_TFree(pvx);
   hypre_TFree(pvx_Hh);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
   hypre_TFree(eigenvectors_Hh);



   /* Clean up */
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrixDestroy(B_array[level]);
      hypre_ParCSRMatrixDestroy(S_array[level-1]);
      hypre_ParCSRMatrixDestroy(T_array[level-1]);
   }
   for ( level = 0; level < num_levels-2; ++level )
   {
      hypre_ParCSRMatrixDestroy(Q_array[level]);
   }
   hypre_TFree(B_array);
   hypre_TFree(Q_array);
   hypre_TFree(S_array);
   hypre_TFree(T_array);

   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);
   for (i = 0; i < block_size; ++i)
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
