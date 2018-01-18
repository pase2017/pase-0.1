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
#define pase_MultiGridDataAMG(data)     ((data)->amg)

#define pase_MultiGridDataNumLevels(data) ((data)->num_levels)
      
typedef struct pase_MultiGrid_struct 
{
   
   MPI_Comm             comm;
   HYPRE_Solver         amg;

   HYPRE_Int            num_levels;
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


PASE_Int PASE_MultiGridFromItoJ(PASE_MultiGrid multi_grid, PASE_Int level_i, PASE_Int level_j, 
      PASE_Int num, HYPRE_ParVector* pvx_i, HYPRE_ParVector* pvx_j);


#ifdef __cplusplus
}
#endif

#endif
