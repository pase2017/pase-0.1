
#include "laplace.h"


static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }


PASE_Int LaplaceEigenvalues(PASE_Real **eigenvalues, PASE_Int block_size, PASE_Int n)
{
   int i, k;
   int num = (int) sqrt(block_size) + 3;
   double  h, h2;
   h = 1.0/(n+1); /*  mesh size*/
   h2 = h*h;
   (*eigenvalues) = hypre_CTAlloc (PASE_Real, num*num);
   for (i = 0; i < num; ++i) 
   {
      for (k = 0; k < num; ++k) 
      {
	 /* eigenvalues[i*num+k] = M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2)); */
	 (*eigenvalues)[i*num+k] = ( 4*sin( (i+1)*M_PI/(2*(n+1)) )*sin( (i+1)*M_PI/(2*(n+1)) ) 
	       + 4*sin( (k+1)*M_PI/(2*(n+1)) )*sin( (k+1)*M_PI/(2*(n+1)) ) ) / h2;
      }
   }
   qsort(*eigenvalues, num*num, sizeof(double), cmp);
   return num*num;
}

/* Create the matrix.
   Note that this is a square matrix, so we indicate the row partition
   size twice (since number of rows = number of cols) */
PASE_Int LaplaceAB( hypre_MPI_Comm          comm, 
                     PASE_Int                n, 
                     HYPRE_IJMatrix*         A,
                     HYPRE_IJMatrix*         B, 
                     HYPRE_IJVector*         b, 
                     HYPRE_IJVector*         x,
	             HYPRE_ParCSRMatrix*     parcsr_A,
	             HYPRE_ParCSRMatrix*     parcsr_B,
	             HYPRE_ParVector*        par_b,
                     HYPRE_ParVector*        par_x
                     )
{
   int i;
   int myid, num_procs;
   int ilower, iupper;
   int local_size, extra;
   int N;
   double h, h2;

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /*  global number of rows */
   h = 1.0/(n+1); /*  mesh size*/
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

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, A);
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, B);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(*A, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(*B, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(*A);
   HYPRE_IJMatrixInitialize(*B);

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
	 HYPRE_IJMatrixSetValues(*A, 1, &nnz, &i, cols, values);
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
	 values[0] = h2;
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(*B, 1, &nnz, &i, cols, values);
      }
   }
   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(*A);
   HYPRE_IJMatrixAssemble(*B);
   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(*A, (void**) parcsr_A);
   HYPRE_IJMatrixGetObject(*B, (void**) parcsr_B);

   /* Create sample rhs and solution vectors */
   HYPRE_IJVectorCreate(comm, ilower, iupper,b);
   HYPRE_IJVectorSetObjectType(*b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(*b);
   HYPRE_IJVectorAssemble(*b);
   HYPRE_IJVectorGetObject(*b, (void **) par_b);

   HYPRE_IJVectorCreate(comm, ilower, iupper,x);
   HYPRE_IJVectorSetObjectType(*x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(*x);
   HYPRE_IJVectorAssemble(*x);
   HYPRE_IJVectorGetObject(*x, (void **) par_x);

   return n;
}
