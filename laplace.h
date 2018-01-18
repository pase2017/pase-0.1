
#ifndef _laplace_h_
#define _laplace_h_


#include "pase_hypre.h"

#ifdef __cplusplus
extern "C" {
#endif

PASE_Int LaplaceEigenvalues(PASE_Real **eigenvalues, PASE_Int block_size, PASE_Int n);

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
                    );


#ifdef __cplusplus
}
#endif


#endif
