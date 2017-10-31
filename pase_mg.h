/*
 * =====================================================================================
 *
 *       Filename:  pash_mg.h
 *
 *    Description:  后期可以考虑将行参类型都变成void *, 以方便修改和在不同计算机上调试
 *                  一般而言, 可以让用户调用的函数以PASE_开头, 内部函数以pase_开头
 *
 *        Version:  1.0
 *        Created:  2017年09月11日 
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  HONGQICHEN   
 *   Organization:  LSEC
 *   Organization:  LSEC
 *   Organization:  LSEC
 *
 * =====================================================================================
 */

#ifndef _pase_mg_h_
#define _pase_mg_h_

#include "pase.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   char *       (*CAlloc)        ( size_t count, size_t elt_size );
   PASE_Int    (*Free)          ( char *ptr );
   PASE_Int    (*CommInfo)      ( void  *A, PASE_Int   *my_id,
                                   PASE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   PASE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   /* y = alpha Ax + beta y */
   PASE_Int    (*Matvec)        ( void *matvec_data, PASE_Complex alpha, void *A, void *x, PASE_Complex beta, void *y );
   PASE_Int    (*MatMultiVec)   ( PASE_Int block_size, PASE_Real alpha, void *A,
                                   void *x, PASE_Real beta, void *y);
   PASE_Int    (*MatvecDestroy) ( void *matvec_data );
   PASE_Real   (*InnerProd)     ( void *x, void *y );
   PASE_Int    (*CopyVector)    ( void *x, void *y );
   PASE_Int    (*ClearVector)   ( void *x );
   PASE_Int    (*ScaleVector)   ( PASE_Real alpha, void *x );
   /* y = alpha x + y  */
   PASE_Int    (*Axpy)          ( PASE_Real alpha, void *x, void *y );

   //PASE_Int    (*Precond)(void *vdata , void *A , void *b , void *x);
   //PASE_Int    (*PrecondSetup)(void *vdata , void *A , void *b , void *x);

   PASE_Int    (*DirSolver)(PASE_Solver solver);
   PASE_Int    (*PreSmooth)(PASE_Solver solver);
   PASE_Int    (*PostSmooth)(PASE_Solver solver);
   PASE_Int    (*PreSmoothInAux)(PASE_Solver solver);
   PASE_Int    (*PostSmoothInAux)(PASE_Solver solver);

} pase_MGFunctions;

typedef struct
{
   PASE_Int 	block_size;
   PASE_Int     pre_iter;
   PASE_Int     post_iter;
   PASE_Int 	max_iter;
   PASE_Int     max_level;
   PASE_Int     cur_level;
   PASE_Real   	rtol;
   PASE_Real   	atol;
   PASE_Real    r_norm;
   void    	**A;
   void    	**M;
   void    	**P;
   void    	**Ap;
   void    	**Mp;
   void    	***u;
   PASE_Complex **eigenvalues;
   pase_MGFunctions *functions;

   PASE_Complex *exact_eigenvalues;

   PASE_Int     num_converged;

   PASE_Int     num_iter;
   PASE_Int     print_level; /* printing when print_level>0 */
   //PASE_Int     logging;  /* extra computations for logging when logging>0 */

} pase_MGData;

pase_MGFunctions *pase_MGFunctionsCreate(
   char *       (*CAlloc)        ( size_t count, size_t elt_size ),
   PASE_Int    (*Free)          ( char *ptr ),
   PASE_Int    (*CommInfo)      ( void  *A, PASE_Int   *my_id,
                                   PASE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   PASE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   PASE_Int    (*Matvec)        ( void *matvec_data, PASE_Complex alpha, void *A, void *x, PASE_Complex beta, void *y ),
   PASE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   PASE_Int    (*CopyVector)    ( void *x, void *y ),
   PASE_Int    (*ClearVector)   ( void *x ),
   PASE_Int    (*ScaleVector)   ( PASE_Complex alpha, void *x ),
   PASE_Int    (*Axpy)          ( PASE_Complex alpha, void *x, void *y ),
   PASE_Int    (*DirSolver)    ( PASE_Solver solver ),
   PASE_Int    (*PreSmooth)     ( PASE_Solver solver ),
   PASE_Int    (*PostSmooth)    ( PASE_Solver solver ),
   PASE_Int    (*PreSmoothInAux)( PASE_Solver solver ),
   PASE_Int    (*PostSmoothInAux)( PASE_Solver solver )
   );
void *pase_MGCreate( pase_MGFunctions* mg_functions);
PASE_Int PASE_ParCSRMGCreate( PASE_Solver* solver);
//PASE_Int PASE_ParCSRMGSetup( PASE_Solver solver);
PASE_Int PASE_MGAddLevel( PASE_Solver solver,
	                 HYPRE_ParCSRMatrix* A,
			 HYPRE_ParCSRMatrix* M,
			 HYPRE_ParCSRMatrix* P,
			 PASE_Int n,
			 PASE_Int flag);
PASE_Int PASE_ParCSRMGInit( MPI_Comm comm,
			    PASE_Solver solver,
			    HYPRE_ParVector* u,
			    PASE_Int block_size,
			    PASE_Int* seed);
PASE_Int PASE_ParCSRMGSolve( PASE_Solver solver);
PASE_Int pase_ParCSRMGIteration( PASE_Solver solver);
PASE_Int pase_ParCSRMGPreSmooth( PASE_Solver solver);
PASE_Int pase_ParCSRMGPostSmooth( PASE_Solver solver);
PASE_Int pase_ParCSRMGAuxMatrixCreate( PASE_Solver solver);
PASE_Int pase_MGAuxMatrixCreate(PASE_Solver solver); 
PASE_Int pase_ParCSRMGPostCorrection( PASE_Solver solver);
PASE_Int pase_ParCSRMGRestart( PASE_Solver solver);
PASE_Int PASE_ParCSRMGDestroy( PASE_Solver solver);
PASE_Int PASE_MGSetMaxIter( PASE_Solver solver, PASE_Int max_iter);
PASE_Int PASE_MGSetPreIter( PASE_Solver solver, PASE_Int pre_iter);
PASE_Int PASE_MGSetPostIter( PASE_Solver solver, PASE_Int post_iter);
PASE_Int PASE_MGSetBlockSize( PASE_Solver solver, PASE_Int block_size);
PASE_Int PASE_MGSetPrintLevel( PASE_Solver solver, PASE_Int print_level);
PASE_Int PASE_MGSetATol( PASE_Solver solver, PASE_Real atol);
PASE_Int PASE_MGSetRTol( PASE_Solver solver, PASE_Real rtol);
PASE_Int PASE_MGSetExactEigenvalues( PASE_Solver solver, PASE_Complex* exact_eigenvalues);
PASE_Int pase_ParCSRMGErrorEstimate( PASE_Solver solver);

PASE_Int pase_ParCSRMGDirSolver(PASE_Solver solver);
PASE_Int pase_ParCSRMGSmootherCG(PASE_Solver solver);
PASE_Int pase_ParCSRMGInAuxSmootherCG(PASE_Solver solver);
void PASE_Orth(PASE_Solver solver);
void PASE_Get_initial_vector(PASE_Solver solver);
PASE_Int PASE_Cg(PASE_Solver solver);
PASE_Int PASE_Vector_inner_production_general_hypre(HYPRE_ParCSRMatrix A, HYPRE_ParVector x, HYPRE_ParVector y, PASE_Real *prod);
PASE_Int PASE_Vector_inner_production_general(PASE_ParCSRMatrix A, PASE_ParVector x, PASE_ParVector y, PASE_Real *prod);
#ifdef __cplusplus
}
#endif

#endif
