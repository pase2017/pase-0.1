HYPRE_Int PASE_LinearSolverCreate(PASE_Solver* linear_solver, PASE_Solver* precond, 
   MPI_Comm comm)
{
   HYPRE_Solver* pcg_solver = linear_solver;	
   /* Create PCG solver for PASE */
   PASE_ParCSRPCGCreate(comm, pcg_solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_PCGSetMaxIter(*pcg_solver, 1000); /* max iterations */
   HYPRE_PCGSetTol(*pcg_solver, 1e-8); /* conv. tolerance */
   HYPRE_PCGSetTwoNorm(*pcg_solver, 1); /* use the two norm as the stopping criteria */
   HYPRE_PCGSetPrintLevel(*pcg_solver, 0); /* print solve info */
   HYPRE_PCGSetLogging(*pcg_solver, 1); /* needed to get run info later */

   /* Now set up the AMG preconditioner and specify any parameters */
   HYPRE_BoomerAMGCreate(precond);
   HYPRE_BoomerAMGSetPrintLevel(*precond, 0); /* print amg solution info */
   HYPRE_BoomerAMGSetInterpType( *precond, 0 );
   HYPRE_BoomerAMGSetPMaxElmts( *precond, 0 );
   HYPRE_BoomerAMGSetCoarsenType(*precond, 6);
   HYPRE_BoomerAMGSetRelaxType(*precond, 6); /* Sym G.S./Jacobi hybrid */
   HYPRE_BoomerAMGSetNumSweeps(*precond, 1);
   HYPRE_BoomerAMGSetTol(*precond, 0.0); /* conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter(*precond, 1); /* do only one iteration! */
   /* Set the PCG preconditioner */
   HYPRE_PCGSetPrecond(*pcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
	 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
   return 0;
}  




HYPRE_Int PASE_LinearSolverSetup(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
PASE_ParVector F, PASE_ParVector U)
{
	hypre_PCGSetup(linear_solver, A, F, U);
	return 0;
}

HYPRE_Int PASE_LinearSolverSolve(PASE_Solver linear_solver, PASE_ParCSRMatrix A, 
PASE_ParVector F, PASE_ParVector U)
{
	hypre_PCGSolve(linear_solver, A, F, U);	
	return 0;	
	// HYPRE_PCGGetNumIterations(linear_solver, &num_iterations);
	// HYPRE_PCGGetFinalRelativeResidualNorm(linear_solver, &final_res_norm);
}
HYPRE_Int PASE_LinearSolverDestroy(PASE_Solver linear_solver, PASE_Solver precond)
{
	PASE_ParCSRPCGDestroy(linear_solver);
	HYPRE_BoomerAMGDestroy(precond);
	return 0;
}