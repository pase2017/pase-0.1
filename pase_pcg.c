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
