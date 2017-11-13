HYPRE_Int PASE_MultiGridCreate(PASE_MultiGrid* multi_grid, HYPRE_Int max_levels,
   HYPRE_ParCSRMatrix parcsr_A, HYPRE_ParCSRMatrix parcsr_B,
   HYPRE_ParVector par_x, HYPRE_ParVector par_b)
{
   (*multi_grid) = hypre_CTAlloc(pase_MultiGrid, 1);
   HYPRE_Solver amg_solver = multi_grid->amg_solver;
   /* Using AMG to get multilevel matrix */
   //hypre_ParAMGData   *amg_data;

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **P_array;
   /* P0P1P2  P1P2  P2 */
   hypre_ParCSRMatrix **Q_array;
   /* rhs and x */
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   
  

  /* -------------------------- 利用AMG生成各个层的矩阵------------------ */

   /* Create solver */
   HYPRE_BoomerAMGCreate(&amg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0);         /* print solve info + parameters */
   HYPRE_BoomerAMGSetInterpType(amg_solver, 0 );
   HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0 );
   HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
   HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);  /* maximum number of levels */
   //   HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);          /* G-S/Jacobi hybrid relaxation */
   //   HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);         /* uses C/F relaxation */
   //   HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);          /* Sweeeps on each level */
   //   HYPRE_BoomerAMGSetTol(amg_solver, 1e-7);             /* conv. tolerance */

   /* Now setup */
   HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, par_b, par_x);

   /* Get A_array, P_array, F_array and U_array of AMG */

   //amg_data = pase_MultiGridDataAMG   (multi_grid);
   /* 这是将指针赋予对应的指针 */   
   A_array  = pase_MultiGridDataAArray(multi_grid);
   P_array  = pase_MultiGridDataPArray(multi_grid);
   F_array  = pase_MultiGridDataFArray(multi_grid);
   U_array  = pase_MultiGridDataUArray(multi_grid);
   
   B_array  = pase_MultiGridDataBArray(multi_grid);
   Q_array  = pase_MultiGridDataQArray(multi_grid);
   
   
   // amg_data = (hypre_ParAMGData*) amg_solver;  
   // A_array = hypre_ParAMGDataAArray(amg_data);
   // P_array = hypre_ParAMGDataPArray(amg_data);
   // F_array = hypre_ParAMGDataFArray(amg_data);
   // U_array = hypre_ParAMGDataUArray(amg_data);

   //num_levels = hypre_ParAMGDataNumLevels(amg_data);
   num_levels = pase_MultiGridDataNumLevels(multi_grid);
   printf ( "The number of levels = %d\n", num_levels );

   B_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);
   Q_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  num_levels);

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


HYPRE_Int PASE_MultiGridDestroy(PASE_MultiGrid multi_grid)
{
   HYPRE_Int level, num_levels;
   hypre_ParCSRMatrix **B_array;
   hypre_ParCSRMatrix **Q_array;
   HYPRE_Solver amg_solver = multi_grid->amg_solver;  
   
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
	
   HYPRE_BoomerAMGDestroy(amg_solver);	
   
   return 0;
	
}