#ifndef _pase_mg_h_
#define _pase_mg_h_

#include "pase_hypre.h"



#ifdef __cplusplus
extern "C" {
#endif

#define pase_MultiGridDataAArray(data)  ((data)->A_array)
#define pase_MultiGridDataBArray(data)  ((data)->B_array)
#define pase_MultiGridDataPArray(data)  ((data)->P_array)
#define pase_MultiGridDataQArray(data)  ((data)->Q_array)
#define pase_MultiGridDataUArray(data)  ((data)->U_array)
#define pase_MultiGridDataFArray(data)  ((data)->F_array)
#define pase_MultiGridDataAMG(data)     ((data)->amg_solver)

#define pase_MultiGridDataNumLevels(data) (hypre_ParAMGDataNumLevels((hypre_ParAMGData*)(data)->amg_solver))
      
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

PASE_Int PASE_MultiGridCreate(PASE_MultiGrid* multi_grid, PASE_Int max_levels, 
   HYPRE_ParCSRMatrix parcsr_A, HYPRE_ParCSRMatrix parcsr_B,
   HYPRE_ParVector par_x, HYPRE_ParVector par_b);

PASE_Int PASE_MultiGridDestroy(PASE_MultiGrid multi_grid);

#ifdef __cplusplus
}
#endif

#endif
