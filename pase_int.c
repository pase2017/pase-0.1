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
#include "pase_int.h"

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
                      PASE_Real  alpha,
                      void   *A,
                      void   *x,
                      PASE_Real  beta,
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
pase_ParKrylovScaleVector( PASE_Real  alpha, void *x )
{
   return ( PASE_ParVectorScale( alpha, (pase_ParVector *) x ) );
}
PASE_Int
pase_ParKrylovAxpy( PASE_Real alpha, void *x, void *y )
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

