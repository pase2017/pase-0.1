/*
 * =====================================================================================
 *
 *       Filename:  pase_pcg.c
 *
 *    Description:  PASE_ParCSRMatrix下PCG求解线性方程组
 *
 *        Version:  1.0
 *        Created:  2017年09月08日 15时41分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_pcg.h"

PASE_Int
pase_ParKrylovCommInfo( void   *A, PASE_Int *my_id, PASE_Int *num_procs)
{
   MPI_Comm comm = ((pase_ParCSRMatrix *)A)->comm;
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}
void *
pase_ParKrylovCreateVector( void *vvector )
{

   pase_ParVector *vector = (pase_ParVector *) vvector;
   pase_ParVector *x_Hh;
   PASE_ParVectorCreate( vector->comm, vector->N_H, vector->block_size,	
	NULL, vector->b_H->partitioning, &x_Hh );

   return ( (void *) x_Hh );
}
PASE_Int
pase_ParKrylovDestroyVector( void *vvector )
{
   pase_ParVector *vector = (pase_ParVector *) vvector;

   return( PASE_ParVectorDestroy( vector ) );
}
PASE_Int
pase_ParKrylovMatvec( void   *matvec_data,
                      HYPRE_Complex  alpha,
                      void   *A,
                      void   *x,
                      HYPRE_Complex  beta,
                      void   *y           )
{
   return ( PASE_ParCSRMatrixMatvec ( alpha,
                                      (pase_ParCSRMatrix *) A,
                                      (pase_ParVector *) x,
                                      beta,
                                      (pase_ParVector *) y ) );
}
PASE_Real
pase_ParKrylovInnerProd( void *x, void *y )
{
   PASE_Real prod;
   PASE_ParVectorInnerProd( (pase_ParVector *) x, (pase_ParVector *) y, &prod ); 
   return prod;
}
PASE_Int
pase_ParKrylovCopyVector( void *x, void *y )
{
   return ( PASE_ParVectorCopy( (pase_ParVector *) x, (pase_ParVector *) y ) );
}
PASE_Int
pase_ParKrylovClearVector( void *x )
{
   return ( PASE_ParVectorSetConstantValues( (pase_ParVector *) x, 0.0 ) );
}
PASE_Int
pase_ParKrylovScaleVector( HYPRE_Complex  alpha, void *x )
{
   return ( PASE_ParVectorScale( alpha, (pase_ParVector *) x ) );
}
PASE_Int
pase_ParKrylovAxpy( HYPRE_Complex alpha, void *x, void *y )
{
   return ( PASE_ParVectorAxpy( alpha, (pase_ParVector *) x, (pase_ParVector *) y ) );
}
PASE_Int
pase_ParKrylovIdentity( void *vdata, void *A, void *b, void *x )
{
   return( pase_ParKrylovCopyVector( b, x ) );
}
PASE_Int
pase_ParSetRandomValues( void* v, PASE_Int seed ) 
{

  PASE_ParVectorSetRandomValues( (pase_ParVector *)v, seed );
  return 0;
}



/* 这里是否需要都将函数名封装成pase呢? */
PASE_Int
PASE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, pase_ParKrylovCommInfo,
         pase_ParKrylovCreateVector, pase_ParKrylovDestroyVector, 
	 hypre_ParKrylovMatvecCreate,
         pase_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         pase_ParKrylovInnerProd, pase_ParKrylovCopyVector,
         pase_ParKrylovClearVector,
         pase_ParKrylovScaleVector, pase_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, pase_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/* TODO:貌似直接调用HYPRE的PCG销毁就行 */
PASE_Int
PASE_ParCSRPCGDestroy( HYPRE_Solver solver )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;

   if (pcg_data)
   {
      hypre_PCGFunctions *pcg_functions = pcg_data->functions;
      if ( (pcg_data -> norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> norms, pcg_functions );
         pcg_data -> norms = NULL;
      } 
      if ( (pcg_data -> rel_norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
         pcg_data -> rel_norms = NULL;
      }
      if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
      {
         (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
         pcg_data -> matvec_data = NULL;
      }
      if ( pcg_data -> p != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> p);
         pcg_data -> p = NULL;
      }
      if ( pcg_data -> s != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> s);
         pcg_data -> s = NULL;
      }
      if ( pcg_data -> r != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r);
         pcg_data -> r = NULL;
      }
      hypre_TFreeF( pcg_data, pcg_functions );
      hypre_TFreeF( pcg_functions, pcg_functions );
   }

   return(hypre_error_flag);
}



PASE_Int PASE_ParCSRSetupInterpreter( mv_InterfaceInterpreter* i)
{
  /* Vector part */

  i->CreateVector = pase_ParKrylovCreateVector;
  i->DestroyVector = pase_ParKrylovDestroyVector; 
  i->InnerProd = pase_ParKrylovInnerProd; 
  i->CopyVector = pase_ParKrylovCopyVector;
  i->ClearVector = pase_ParKrylovClearVector;
  i->SetRandomValues = pase_ParSetRandomValues;
  i->ScaleVector = pase_ParKrylovScaleVector;
  i->Axpy = pase_ParKrylovAxpy;

  /* Multivector part */

  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}
PASE_Int PASE_ParCSRSetupMatvec( HYPRE_MatvecFunctions* mv)
{
  mv->MatvecCreate = hypre_ParKrylovMatvecCreate;
  mv->Matvec = pase_ParKrylovMatvec;
  mv->MatvecDestroy = hypre_ParKrylovMatvecDestroy;

  mv->MatMultiVecCreate = NULL;
  mv->MatMultiVec = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}

