#include <stdlib.h>
#include <math.h>
#include "pase_mg.h"

PASE_Int PASE_MultiGridCreate(PASE_MultiGrid* multi_grid, PASE_Int max_levels,
      HYPRE_ParCSRMatrix parcsr_A, HYPRE_ParCSRMatrix parcsr_B,
      HYPRE_ParVector par_x, HYPRE_ParVector par_b)
{
   PASE_Int num_levels, level, num_procs;

   *multi_grid = hypre_CTAlloc(pase_MultiGrid, 1);

   MPI_Comm_size(parcsr_A->comm,  &num_procs);

   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* P0P1P2  P1P2  P2 */
   hypre_ParCSRMatrix **Q_array;

   /* -------------------------- 利用AMG生成各个层的矩阵------------------ */

   /* Create solver */
   HYPRE_BoomerAMGCreate(&(*multi_grid)->amg);
   HYPRE_Solver      amg      = (*multi_grid)->amg;
   hypre_ParAMGData* amg_data = (hypre_ParAMGData*) amg;

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(amg, 0);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetInterpType(amg, 0 );
   HYPRE_BoomerAMGSetPMaxElmts(amg, 0 );
   HYPRE_BoomerAMGSetCoarsenType(amg, 6);
   HYPRE_BoomerAMGSetMaxLevels(amg, max_levels);  /* maximum number of levels */

   /* Now setup */
   HYPRE_BoomerAMGSetup(amg, parcsr_A, par_b, par_x);
   (*multi_grid)->num_levels = hypre_ParAMGDataNumLevels(amg_data);
   num_levels = (*multi_grid)->num_levels;

   printf ( "The number of levels = %d\n", num_levels );

   /* Get A_array, P_array, F_array and U_array of AMG */
   (*multi_grid)->A_array = hypre_ParAMGDataAArray(amg_data);
   (*multi_grid)->P_array = hypre_ParAMGDataPArray(amg_data);
   (*multi_grid)->U_array = hypre_ParAMGDataUArray(amg_data);
   (*multi_grid)->F_array = hypre_ParAMGDataFArray(amg_data);
   (*multi_grid)->B_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   (*multi_grid)->Q_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);

   P_array  = pase_MultiGridDataPArray(*multi_grid);
   B_array  = pase_MultiGridDataBArray(*multi_grid);
   Q_array  = pase_MultiGridDataQArray(*multi_grid);

   /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
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
   /* P0P1P2  P1P2  P2    */
   Q_array[num_levels-2] = P_array[num_levels-2];
   for ( level = num_levels-3; level >= 0; --level )
   {
      Q_array[level] = hypre_ParMatmul(P_array[level], Q_array[level+1]); 
   }

   return 0;
}


PASE_Int PASE_MultiGridDestroy(PASE_MultiGrid multi_grid)
{
   PASE_Int level, num_levels;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **Q_array;
   HYPRE_Solver amg = multi_grid->amg;  

   B_array  = pase_MultiGridDataBArray(multi_grid);
   Q_array  = pase_MultiGridDataQArray(multi_grid);

   num_levels = pase_MultiGridDataNumLevels(multi_grid);
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

   HYPRE_BoomerAMGDestroy(amg);	

   hypre_TFree(multi_grid);
   return 0;
}

/**
 * @brief 0 1 2 3 越來越粗
 *
 * Q[0] = P0P1P2  Q[1] = P1P2  Q[2] = P2 
 * if level_i == num_levels-1  pvx_j = Q  [level_j] pvx_i
 * if level_j == num_levels-1  pvx_j = Q^T[level_i] pvx_i
 *
 * if level_i == level_j + 1  从粗到细
 *    pvx_j = P  [level_j] pvx_i 
 * if level_i == level_j - 1  从细到粗
 *    pvx_j = P^T[level_j] pvx_i 
 *
 *
 * @param multi_grid
 * @param level_i
 * @param level_j
 * @param num
 * @param pvx_i
 * @param pvx_j
 */
PASE_Int PASE_MultiGridFromItoJ(PASE_MultiGrid multi_grid, PASE_Int level_i, PASE_Int level_j, 
      PASE_Int num, HYPRE_ParVector* pvx_i, HYPRE_ParVector* pvx_j)
{
   PASE_Int idx, num_levels;

   num_levels = pase_MultiGridDataNumLevels(multi_grid);
   hypre_ParCSRMatrix** P_array;
   hypre_ParCSRMatrix** Q_array;

   P_array = pase_MultiGridDataPArray(multi_grid);
   Q_array = pase_MultiGridDataQArray(multi_grid);


   if (level_i == num_levels-1)
   {
      for (idx = 0; idx < num; ++idx)
      {
	 HYPRE_ParCSRMatrixMatvec (1.0, Q_array[level_j], pvx_i[idx], 0.0, pvx_j[idx]);
      }
   }
   else if (level_j == num_levels-1)
   {
      for (idx = 0; idx < num; ++idx)
      {
	 HYPRE_ParCSRMatrixMatvecT(1.0, Q_array[level_i], pvx_i[idx], 0.0, pvx_j[idx]);
      }
   }
   else if (level_i-level_j == 1)
   {
      for (idx = 0; idx < num; ++idx)
      {
	 HYPRE_ParCSRMatrixMatvec (1.0, P_array[level_j], pvx_i[idx], 0.0, pvx_j[idx]);
      }
   } 
   else if (level_i-level_j == -1)
   {
      for (idx = 0; idx < num; ++idx)
      {
	 HYPRE_ParCSRMatrixMatvecT(1.0, P_array[level_i], pvx_i[idx], 0.0, pvx_j[idx]);
      }
   } 
   else
   {
      printf ( "Can not be from %d to %d.\n", level_i, level_j );
   }

   return 0;
}
