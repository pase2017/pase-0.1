/* mv_InterfaceInterpreter 

void*  (*CreateVector)  ( void *vector ); 
HYPRE_Int    (*DestroyVector) ( void *vector );

HYPRE_Real   (*InnerProd)     ( void *x,  void *y );
HYPRE_Int    (*CopyVector)    ( void *x,  void *y );
HYPRE_Int    (*ClearVector)   ( void *x );
HYPRE_Int    (*SetRandomValues)   ( void *x,  HYPRE_Int seed );
HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha,  void *x );
HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha,  void *x,  void *y );
HYPRE_Int    (*VectorSize)    (void * vector);

void*  (*CreateMultiVector)  ( void*,  HYPRE_Int n,  void *vector );
void*  (*CopyCreateMultiVector)  ( void *x,  HYPRE_Int );
void    (*DestroyMultiVector) ( void *x );

HYPRE_Int    (*Width)  ( void *x );
HYPRE_Int    (*Height) ( void *x );

void   (*SetMask) ( void *x,  HYPRE_Int *mask );

void   (*CopyMultiVector)    ( void *x,  void *y );
void   (*ClearMultiVector)   ( void *x );
void   (*SetRandomVectors)   ( void *x,  HYPRE_Int seed );
void   (*MultiInnerProd)     ( void *x,  void *y,  HYPRE_Int,  HYPRE_Int,  HYPRE_Int,  HYPRE_Real* );
void   (*MultiInnerProdDiag) ( void *x,  void *y,  HYPRE_Int*,  HYPRE_Int,  HYPRE_Real* );
void   (*MultiVecMat)        ( void *x,  HYPRE_Int,  HYPRE_Int,  HYPRE_Int,  HYPRE_Complex*,  void *y );
void   (*MultiVecMatDiag)    ( void *x,  HYPRE_Int*,  HYPRE_Int,  HYPRE_Complex*,  void *y );
void   (*MultiAxpy)          ( HYPRE_Complex alpha,  void *x,  void *y );

void   (*MultiXapy)          ( void *x,  HYPRE_Int,  HYPRE_Int,  HYPRE_Int,  HYPRE_Complex*,  void *y );
void   (*Eval)               ( void (*f)( void*,  void*,  void* ),  void*,  void *x,  void *y );
 
*/

/* using interpreter and matvecFunctions to implement GCG
 * 1. Random initialize X_pre
 * 2. Orthogonalize X_pre 
 * 3. Compute Ritz A in (X_pre)
 * 4. Compute Rayleigh Ritz problem to get X and A_sub = x'Ax
 * 5. Compute P using linear solver 
 * 6. Orthogonalize P in (X)
 * 7. Compute Ritz A in (X, P) using A_sub
 * 8. Compute Rayleigh Ritz problem to get x_pre, X_pre and p_pre
 * 9. W = NULL and initialize unlock
 *10. do while 
 *   *  Set zero for the part of X in p_pre
 *   *  Orthogonalize (x_pre, p_pre)
 *   *  Compute P_pre = (X, P, W)*p_pre, whose basis is (X, P, W)
 *      Compute A_sub = (x_pre, p_pre)'A(x_pre, p_pre) 
 *   *  Compute W using linear solver using X_pre
 *      Orthogonalize W in (X_pre, P_pre) 
 *      Compute Ritz A in (X_pre, P_pre, W) using A_sub 
 *      Compute Rayleigh Ritz problem to get x, X and p
 *      Check convergence to upate unlock
 *      X switch X_pre
 *      P switch P_pre
 *      p switch p_pre
 *11. end while
 *12. Rayleigh Quation 
 *
 *remark: Orthogonalization remove almost zero vectors
 *
 * ComputeA(A, order, start, vecs);
 * size = MatrixOrthogonalize(mat, fixed_vecs,  fixed_size, vecs,   size);
 * size = L2Orthogonalize(fixed_array, fixed_size, arrays, size);
 * LinearSolver(mat, rhs, sol);
 *
 * */

typedef struct
{
   gcg_Tolerance                 tolerance;
   HYPRE_Int                     maxIterations;
   HYPRE_Int                     verbosityLevel;
   HYPRE_Int                     precondUsageMode;
   HYPRE_Int                     iterationNumber;
   utilities_FortranMatrix*      eigenvaluesHistory;
   utilities_FortranMatrix*      residualNorms;
   utilities_FortranMatrix*      residualNormsHistory;

} gcg_Data;


typedef struct
{

   gcg_Data                      gcgData;

   mv_InterfaceInterpreter*      interpreter;

   void*                         A;
   void*                         matvecData;
   void*                         precondData;

   void*                         B;
   void*                         matvecDataB;
   void*                         T;
   void*                         matvecDataT;

   hypre_LOBPCGPrecond           precondFunctions;

   HYPRE_MatvecFunctions*        matvecFunctions;

} hypre_GCGData; 



HYPRE_LOBPCGCreate
HYPRE_Int HYPRE_GCGCreate (mv_InterfaceInterpreter *interpreter, HYPRE_MatvecFunctions *mvfunctions, HYPRE_Solver *solver);
HYPRE_Int HYPRE_GCGDestroy(HYPRE_Solver solver);
HYPRE_Int HYPRE_GCGSetup  (HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,  HYPRE_ParVector x);
HYPRE_Int HYPRE_GCGSetupB (HYPRE_Solver solver, HYPRE_ParCSRMatrix B, HYPRE_ParVector x);
HYPRE_Int HYPRE_GCGSolve  (HYPRE_Solver solver, mv_MultiVectorPtr  y, mv_MultiVectorPtr x, HYPRE_Real *lambda, HYPRE_Int num_lock);

HYPRE_Int
gcg_initialize( gcg_Data* data )
{
   (data->tolerance).absolute    = 1.0e-06;
   (data->tolerance).relative    = 1.0e-06;
   (data->maxIterations)         = 500;
   (data->precondUsageMode)      = 0;
   (data->verbosityLevel)        = 0;
   (data->eigenvaluesHistory)    = utilities_FortranMatrixCreate();
   (data->residualNorms)         = utilities_FortranMatrixCreate();
   (data->residualNormsHistory)  = utilities_FortranMatrixCreate();

   return 0;
}

   HYPRE_Int
gcg_clean( gcg_Data* data )
{
   utilities_FortranMatrixDestroy( data->eigenvaluesHistory );
   utilities_FortranMatrixDestroy( data->residualNorms );
   utilities_FortranMatrixDestroy( data->residualNormsHistory );

   return 0;
}

   HYPRE_Int
hypre_GCGDestroy( void *gcg_vdata )
{
   hypre_GCGData      *gcg_data      = (hypre_GCGData*)gcg_vdata;

   if (gcg_data) {
      HYPRE_MatvecFunctions * mv = gcg_data->matvecFunctions;
      if ( gcg_data->matvecData != NULL ) {
	 (*(mv->MatvecDestroy))(gcg_data->matvecData);
	 gcg_data->matvecData = NULL;
      }
      if ( gcg_data->matvecDataB != NULL ) {
	 (*(mv->MatvecDestroy))(gcg_data->matvecDataB);
	 gcg_data->matvecDataB = NULL;
      }
      if ( gcg_data->matvecDataT != NULL ) {
	 (*(mv->MatvecDestroy))(gcg_data->matvecDataT);
	 gcg_data->matvecDataT = NULL;
      }

      gcg_clean( &(gcg_data->gcgData) );

      hypre_TFree( gcg_vdata );
   }

   return hypre_error_flag;
}

   HYPRE_Int
hypre_GCGSetup( void *gcg_vdata, void *A, void *b, void *x )
{
   hypre_GCGData *gcg_data = (hypre_GCGData*)gcg_vdata;
   HYPRE_MatvecFunctions * mv = gcg_data->matvecFunctions;
   HYPRE_Int  (*precond_setup)(void*,void*,void*,void*) = (gcg_data->precondFunctions).PrecondSetup;
   void *precond_data = (gcg_data->precondData);

   (gcg_data->A) = A;

   if ( gcg_data->matvecData != NULL )
      (*(mv->MatvecDestroy))(gcg_data->matvecData);
   (gcg_data->matvecData) = (*(mv->MatvecCreate))(A, x);

   if ( precond_setup != NULL ) {
      if ( gcg_data->T == NULL )
	 precond_setup(precond_data, A, b, x);
      else
	 precond_setup(precond_data, gcg_data->T, b, x);
   }

   return hypre_error_flag;
}

   HYPRE_Int
hypre_GCGSetupB( void *gcg_vdata, void *B, void *x )
{
   hypre_GCGData *gcg_data = (hypre_GCGData*)gcg_vdata;
   HYPRE_MatvecFunctions * mv = gcg_data->matvecFunctions;

   (gcg_data->B) = B;

   if ( gcg_data->matvecDataB != NULL )
      (*(mv->MatvecDestroy))(gcg_data -> matvecDataB);
   (gcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
   if ( B != NULL )
      (gcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
   else
      (gcg_data->matvecDataB) = NULL;

   return hypre_error_flag;
}

   HYPRE_Int
hypre_GCGSetPrecond( void  *gcg_vdata,
      HYPRE_Int  (*precond)(void*,void*,void*,void*),
      HYPRE_Int  (*precond_setup)(void*,void*,void*,void*),
      void  *precond_data )
{
   hypre_GCGData* gcg_data = (hypre_GCGData*)gcg_vdata;

   (gcg_data->precondFunctions).Precond      = precond;
   (gcg_data->precondFunctions).PrecondSetup = precond_setup;
   (gcg_data->precondData)                   = precond_data;

   return hypre_error_flag;
}


   HYPRE_Int
HYPRE_GCGCreate( mv_InterfaceInterpreter* ii, HYPRE_MatvecFunctions* mv, HYPRE_Solver* solver )
{
   hypre_GCGData *gcg_data;

   gcg_data = hypre_CTAlloc(hypre_GCGData,1);

   (gcg_data->precondFunctions).Precond = NULL;
   (gcg_data->precondFunctions).PrecondSetup = NULL;

   /* set defaults */

   (gcg_data->interpreter)               = ii;
   gcg_data->matvecFunctions             = mv;

   (gcg_data->matvecData)	       	= NULL;
   (gcg_data->B)	       		= NULL;
   (gcg_data->matvecDataB)	       	= NULL;
   (gcg_data->T)	       		= NULL;
   (gcg_data->matvecDataT)	       	= NULL;
   (gcg_data->precondData)	       	= NULL;

   gcg_initialize( &(gcg_data->gcgData) );

   *solver = (HYPRE_Solver)gcg_data;

   return hypre_error_flag;
}

   HYPRE_Int 
HYPRE_GCGDestroy( HYPRE_Solver solver )
{
   return( hypre_GCGDestroy( (void *) solver ) );
}

   HYPRE_Int 
HYPRE_GCGSetup( HYPRE_Solver solver,
      HYPRE_Matrix A,
      HYPRE_Vector b,
      HYPRE_Vector x      )
{
   return( hypre_GCGSetup( solver, A, b, x ) );
}

   HYPRE_Int 
HYPRE_GCGSetupB( HYPRE_Solver solver,
      HYPRE_Matrix B,
      HYPRE_Vector x      )
{
   return( hypre_GCGSetupB( solver, B, x ) );
}

   HYPRE_Int
HYPRE_GCGSetPrecond( HYPRE_Solver         solver,
      HYPRE_PtrToSolverFcn precond,
      HYPRE_PtrToSolverFcn precond_setup,
      HYPRE_Solver         precond_solver )
{
   return( hypre_GCGSetPrecond( (void *) solver,
	    (HYPRE_Int (*)(void*, void*, void*, void*))precond,
	    (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
	    (void *) precond_solver ) );
}
