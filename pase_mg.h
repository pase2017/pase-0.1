#ifndef _pase_mv_h_
#define _pase_mv_h_

#include "pase_hypre.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef struct pase_MultiGrid_struct 
{
   
   MPI_Comm            comm;
   HYPRE_Solver        amg_solver;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* P0P1P2  P1P2  P2 */
   hypre_ParCSRMatrix **Q_array;
   /* rhs and x */
   hypre_ParVector    **U_array;
   hypre_ParVector    **F_array;

   
   
} pase_MultiGrid;
typedef struct pase_MultiGrid_struct *PASE_MultiGrid;

HYPRE_Int PASE_MultiGridCreate(PASE_MultiGrid* multi_grid, 
   HYPRE_ParCSRMatrix parcsr_A, HYPRE_ParCSRMatrix parcsr_B,
   HYPRE_ParVector par_x, HYPRE_ParVector par_b);

HYPRE_Int PASE_MultiGridDestroy(PASE_MultiGrid multi_grid);

#ifdef __cplusplus
}
#endif

#endif
