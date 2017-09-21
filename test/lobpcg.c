/*
   Example lobpcg

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex11

   Sample run:   mpirun -np 4 ex11

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is LOBPCG with AMG preconditioner.
*/

#include "pase.h"

int main (int argc, char *argv[])
{
   int i, k;
   int myid, num_procs;
   int N, n;
   int blockSize;

   int ilower, iupper;
   int local_size, extra;

   double h, h2;

   HYPRE_IJMatrix A, B;
   HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;
   PASE_ParVector* pvx;

   HYPRE_Solver lobpcg_solver;
   mv_InterfaceInterpreter* interpreter;
   HYPRE_MatvecFunctions matvec_fn;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Default problem parameters */
   n = 6;
   blockSize = 2;

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
         printf("  -blockSize <n>      : eigenproblem block size (default: 10)\n");
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
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorInitialize(x);
   {
      double *x_values;
      double *b_values;
      int    *rows;

      x_values =  (double*) calloc(local_size, sizeof(double));
      b_values =  (double*) calloc(local_size, sizeof(double));
      rows = (int*) calloc(local_size, sizeof(int));

      for (i=0; i<local_size; i++)
      {
         x_values[i] = 0.0;
         b_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      HYPRE_IJVectorSetValues(b, local_size, rows, b_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(b_values);
      free(x_values);
      free(rows);
   }
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   /* -------------------------- 利用AMG生成各个层的矩阵------------------ */
   HYPRE_Int    num_levels, level;
   HYPRE_Int    maxLevels = 2;
   HYPRE_Solver amg_solver;
   /* Using AMG to get multilevel matrix */
   hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   /* Create solver */
   HYPRE_BoomerAMGCreate(&amg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetOldDefault(amg_solver);            /* Falgout coarsening with modified classical interpolaiton */
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 0);
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


   /*---------------------------- TEST PASE Matrix and Vector ----------------------------*/

   HYPRE_Int block_size = 1;
   HYPRE_Int N_H = A_array[1]->global_num_rows;

   HYPRE_ParCSRMatrix A_h = A_array[0];
   HYPRE_ParCSRMatrix A_H = A_array[1];
   HYPRE_ParCSRMatrix B_h = B_array[0];
   HYPRE_ParCSRMatrix B_H = B_array[1];
   HYPRE_ParCSRMatrix P   = P_array[0];
   HYPRE_ParVector workspace_h = F_array[0];
   HYPRE_ParVector workspace_H = F_array[1];

   HYPRE_ParVector* x_h = &par_x;
   HYPRE_ParVector  x_H = U_array[1];
   HYPRE_ParCSRMatrixMatvecT(1.0, P, x_h[0], 0.0, x_H);

//   HYPRE_ParCSRMatrixPrint(A_h, "A_h");
//   HYPRE_ParCSRMatrixPrint(A_H, "A_H");
//   HYPRE_ParCSRMatrixPrint(P, "P");
//   HYPRE_ParVectorPrint(x_h[0], "x_h");

   HYPRE_ParVector* b_h = &par_b;
   HYPRE_ParVector  b_H = F_array[1];
   HYPRE_ParCSRMatrixMatvecT(1.0, P, b_h[0], 0.0, b_H);


   HYPRE_Real *data_x;
   data_x = hypre_CTAlloc(HYPRE_Real, block_size);
   data_x[0] = 1.0;
   HYPRE_Real *data_b;
   data_b = hypre_CTAlloc(HYPRE_Real, block_size);
   data_b[0] = 1.0;
   
   PASE_ParCSRMatrix A_Hh;
   PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, A_H, P, A_h, x_h, &A_Hh, workspace_H, workspace_h );
   (A_Hh->aux_hh->data)[0] = 1000.0;

   PASE_ParCSRMatrix B_Hh;
   PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, B_H, P, B_h, x_h, &B_Hh, workspace_H, workspace_h );
   (B_Hh->aux_hh->data)[0] = 1.0;

   PASE_ParVector    x_Hh;
   PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, x_H, NULL, &x_Hh );
   (x_Hh->aux_h->data)[0] = 1.0;

   PASE_ParVector    b_Hh;
   PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, b_H, NULL, &b_Hh );
   (b_Hh->aux_h->data)[0] = 1.0;

//   PASE_ParCSRMatrixMatvec(1.0, A_Hh, x_Hh, 0.0, b_Hh);

//   PASE_ParCSRMatrixPrint(A_Hh, "A_Hh");
//   PASE_ParCSRMatrixPrint(B_Hh, "B_Hh");
//   PASE_ParVectorPrint(x_Hh, "x_Hh");
//   PASE_ParVectorPrint(b_Hh, "b_Hh");

//   PASE_ParVectorAxpy( 2.0, x_Hh , b_Hh );
//   PASE_ParVectorPrint(b_Hh, "b_Hh");

   /*---------------------------- TEST PASE LOBPCG without preocondition ----------------------------*/

   printf ( "TEST LOBPCG\n" );

   /* LOBPCG eigensolver */
   {
      int time_index;

      int maxIterations = 100; /* maximum number of iterations */
      int pcgMode = 1;         /* use rhs as initial guess for inner pcg iterations */
      int verbosity = 2;       /* print iterations info */
      double tol = 1.e-8;      /* absolute tolerance (all eigenvalues) */
      int lobpcgSeed = 775;    /* random seed */

      mv_MultiVectorPtr eigenvectors = NULL;
      mv_MultiVectorPtr constraints = NULL;
      double *eigenvalues = NULL;

      if (myid != 0)
         verbosity = 0;

      /* define an interpreter for the ParCSR interface */
      interpreter = hypre_CTAlloc(mv_InterfaceInterpreter,1);
      HYPRE_ParCSRSetupInterpreter(interpreter);
      PASE_ParCSRSetupInterpreter(interpreter);
      PASE_ParCSRSetupMatvec(&matvec_fn);

      /* eigenvectors - create a multivector */
      eigenvectors =
         mv_MultiVectorCreateFromSampleVector(interpreter, blockSize, x_Hh);
      mv_MultiVectorSetRandom (eigenvectors, lobpcgSeed);

      /* eigenvectors - get a pointer */
      {
	 mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
	 pvx = (PASE_ParVector*)(tmp -> vector);
      }

      /* eigenvalues - allocate space */
      eigenvalues = (double*) calloc( blockSize, sizeof(double) );

      /* 
       * LOBPCG只要改变interpreter和matvec_fn就可以完全适用于任何形式的矩阵向量
       * 前者用于自己算法的函数, 后者应该是用于内部pcg的函数 
       */
      HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
      HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
      HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

      PASE_LOBPCGSetup(lobpcg_solver, A_Hh, b_Hh, x_Hh);
      PASE_LOBPCGSetupB(lobpcg_solver, B_Hh, x_Hh);

      time_index = hypre_InitializeTiming("LOBPCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_LOBPCGSolve(lobpcg_solver, constraints, eigenvectors, eigenvalues );

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();


//      PASE_ParVectorPrint(pvx[0], "pvx");
      if (myid==0)
      {
	 for (k = 0; k < blockSize; ++k)
	 {
	    printf ( "eigenvalues = %f\n", eigenvalues[k]/h2 );
	 }
      }
      for (k = 0; k < blockSize; ++k)
      {
//	 PASE_ParVectorCopy(pvx[k], b_Hh);
	 PASE_ParCSRMatrixMatvec( eigenvalues[k], B_Hh, pvx[k], 0.0, b_Hh );
	 PASE_ParCSRMatrixMatvec( 1.0, A_Hh, pvx[k], -eigenvalues[k], b_Hh );
	 PASE_ParVectorInnerProd(b_Hh, b_Hh, &eigenvalues[k]);
      }
      if (myid==0)
      {
	 for (k = 0; k < blockSize; ++k)
	 {
	    printf ( "error = %f\n", eigenvalues[k] );
	 }
      }

      /* clean-up */
      mv_MultiVectorDestroy(eigenvectors);
      HYPRE_LOBPCGDestroy(lobpcg_solver);
      hypre_TFree(eigenvalues);
      hypre_TFree(interpreter);
   }

   /* Clean up */
   for ( level = 1; level < num_levels; ++level )
   {
      hypre_ParCSRMatrixDestroy(B_array[level]);
   }
   hypre_TFree(B_array);
   PASE_ParCSRMatrixDestroy( A_Hh );
   PASE_ParCSRMatrixDestroy( B_Hh );
   PASE_ParVectorDestroy( b_Hh );
   PASE_ParVectorDestroy( x_Hh );
   hypre_TFree(data_b);
   hypre_TFree(data_x);

   /* Destroy amg_solver */
   HYPRE_BoomerAMGDestroy(amg_solver);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
