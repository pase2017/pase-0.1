/*
 * =====================================================================================
 *
 *       Filename:  pase_diag.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年01月18日 21时04分15秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include "pase_diag.h"


/* 
 * (S' AA S) S^-1  u  = lambda (S' BB S) S^-1 u
 *
 * Z = A^-1 X
 *
 * S = E   -Z
 *     0    E
 *
 * A    X     A    O
 * X'   a --> O'   a-X'Z
 *
 * B    Y     B          Y-BZ
 * Y'   b --> (Y-BZ)'    b-Y'Z-Z'Y+Z'BZ      
 *
 *
 * S^-1 u = v, u = S v
 *
 * E  -Z |  v1   =   v1 - Z v2 = u1
 * 0   E |  v2       v2          u2
 *
 * */
HYPRE_Int
hypre_PASEDiagCreate(hypre_PASEDiag* diag_data,
      PASE_ParCSRMatrix parcsr_A, PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector    par_b,    PASE_ParVector    par_x, 
      HYPRE_ParVector  *Z ,       HYPRE_ParVector sample)
{
   HYPRE_Int row, col, block_size;
   diag_data->block_size = parcsr_A->block_size;
   block_size = diag_data->block_size;

   hypre_CSRMatrix *XZ, *YZ, *ZBZ;
   diag_data->XZ  = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   XZ  = diag_data->XZ;
   hypre_CSRMatrixInitialize( XZ );

   HYPRE_Int*  matrix_i = XZ->i;
   HYPRE_Int*  matrix_j = XZ->j;
   for (row = 0; row < block_size+1; ++row)
   {
      matrix_i[row] = row*block_size;
   }
   for ( row = 0; row < block_size; ++row)
   {
      for ( col = 0; col < block_size; ++col)
      {
	 matrix_j[col+row*block_size] = col;
      }
   }

   if (Z!=NULL)
   {
      diag_data->Z = Z;
      diag_data->owns_data = 0;
   }
   else if (sample!=NULL)
   {
      diag_data->Z = hypre_CTAlloc (HYPRE_ParVector, block_size);
      for (row = 0; row < block_size; ++row)
      {
	 diag_data->Z[row] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)sample);
      }
      diag_data->owns_data = 1;
   }
   else
   {
      printf ( "Can not create diag_data Z\n" );
   }

   diag_data->YZ  = NULL;
   diag_data->ZBZ = NULL;
   diag_data->BZ  = NULL;
   diag_data->ZF = NULL;

   if (parcsr_B!=NULL)
   {
      diag_data->YZ  = hypre_CSRMatrixCreate(block_size, block_size, block_size*block_size);
      diag_data->ZBZ = hypre_CSRMatrixCreate(block_size, block_size, block_size*block_size);
      YZ  = diag_data->YZ;
      ZBZ = diag_data->ZBZ;

      hypre_CSRMatrixInitialize( YZ );
      hypre_CSRMatrixInitialize( ZBZ);
      hypre_CSRMatrixCopy(XZ, YZ,  0);
      hypre_CSRMatrixCopy(XZ, ZBZ, 0);

      if (sample!=NULL)
      {
	 diag_data->BZ = hypre_CTAlloc (HYPRE_ParVector, block_size);
	 for (row = 0; row < block_size; ++row)
	 {
	    diag_data->BZ[row] = (HYPRE_ParVector)hypre_ParKrylovCreateVector((void*)sample);
	 }
      }
      else
      {
	 printf ( "Can not create diag_data BZ\n" );
      }
   }
   if (par_x!=NULL || par_b!=NULL)
   {
      diag_data->ZF = hypre_SeqVectorCreate(block_size);
      hypre_SeqVectorInitialize(diag_data->ZF);

   }



   HYPRE_BoomerAMGCreate(&(diag_data->precond));

   HYPRE_Solver precond = diag_data->precond;

   HYPRE_BoomerAMGSetOldDefault(precond);
   HYPRE_BoomerAMGSetRelaxType (precond,  3);    /* G-S/Jacobi hybrid relaxation */
   HYPRE_BoomerAMGSetRelaxOrder(precond,  1);    /* uses C/F relaxation */
   HYPRE_BoomerAMGSetNumSweeps (precond,  1);
   HYPRE_BoomerAMGSetTol       (precond,  1E-14);/* conv. tolerance zero */
   HYPRE_BoomerAMGSetMaxIter   (precond,  20);   /* do only one iteration! */
   HYPRE_BoomerAMGSetPrintLevel(precond,  0);    /* print amg solution info */

   HYPRE_BoomerAMGSetup(precond, parcsr_A->A_H, parcsr_A->aux_hH[0], Z[0]);


//   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &(diag_data->precond));
//   HYPRE_Solver precond = diag_data->precond;
//
//   HYPRE_PCGSetMaxIter   (precond, 50);     /*  max iterations */
//   HYPRE_PCGSetTol       (precond, 1E-14);  /*  conv. tolerance */
//   HYPRE_PCGSetTwoNorm   (precond, 1);      /*  use the two norm as the stopping criteria */
//   HYPRE_PCGSetLogging   (precond, 1); 
//   HYPRE_PCGSetPrintLevel(precond, 2);
//
//   HYPRE_ParCSRPCGSetup(precond, parcsr_A->A_H, parcsr_A->aux_hH[0], Z[0]);


   return 0;
}

HYPRE_Int
hypre_PASEDiagChange(hypre_PASEDiag*  diag_data, 
      PASE_ParCSRMatrix parcsr_A,     PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector*   eigenvectors, 
      PASE_ParVector    par_b,        PASE_ParVector    par_x)
{

   HYPRE_Real* matrix_data;
   hypre_CSRMatrix *XZ, *YZ, *ZBZ;
   hypre_Vector    *ZF;
   HYPRE_ParVector *Z, *BZ;

   XZ  = diag_data->XZ;
   YZ  = diag_data->YZ;
   ZBZ = diag_data->ZBZ;
   Z   = diag_data->Z;
   ZF  = diag_data->ZF;
   BZ  = diag_data->BZ;

   HYPRE_Solver precond = diag_data->precond;

   HYPRE_Int row, col, block_size;
   block_size = diag_data->block_size;

   /* Z = A^-1 X 求解线性方程组 */
   for (row = 0; row < block_size; ++row)
   {
      HYPRE_BoomerAMGSolve(precond, parcsr_A->A_H, parcsr_A->aux_hH[row], Z[row]);
//      HYPRE_ParCSRPCGSolve(precond, parcsr_A->A_H, parcsr_A->aux_hH[row], Z[row]);
   }
   /* XZ */
   matrix_data = XZ->data;
   for (row = 0; row < block_size; ++row)
   {
      for (col = row; col < block_size; ++col)
      {
	 HYPRE_ParVectorInnerProd(parcsr_A->aux_hH[row], Z[col], &matrix_data[row*block_size+col]);
	 parcsr_A->aux_hh->data[row*block_size+col] -= matrix_data[row*block_size+col];
      }
   }
   for (row = 0; row < block_size; ++row)
   {
      for (col = 0; col < row; ++col)
      {
	 parcsr_A->aux_hh->data[row*block_size+col] = parcsr_A->aux_hh->data[col*block_size+row];
	 matrix_data[row*block_size+col] = matrix_data[col*block_size+row];
      }
   }
   parcsr_A->diag = 1;

   if (parcsr_B!=NULL)
   {
      /* YZ */
      matrix_data = YZ->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorInnerProd(parcsr_B->aux_hH[row], Z[col], &matrix_data[row*block_size+col]);
	    parcsr_B->aux_hh->data[row*block_size+col] -= matrix_data[row*block_size+col];
	 }
      }
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    parcsr_B->aux_hh->data[row*block_size+col] -= matrix_data[col*block_size+row];
	 }
      }
      /* BZ */
      for (row = 0; row < block_size; ++row)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, parcsr_B->A_H, Z[row], 0.0, BZ[row] );
	 HYPRE_ParVectorAxpy(-1.0, BZ[row], parcsr_B->aux_hH[row]);
      }
      /* ZBZ */
      matrix_data = ZBZ->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = row; col < block_size; ++col)
	 {
	    HYPRE_ParVectorInnerProd(BZ[row], Z[col], &matrix_data[row*block_size+col]);
	    parcsr_B->aux_hh->data[row*block_size+col] += matrix_data[row*block_size+col];
	 }
      }
      for (row = 0; row < block_size; ++row)
      {
	 for (col = 0; col < row; ++col)
	 {
	    parcsr_B->aux_hh->data[row*block_size+col] = parcsr_B->aux_hh->data[col*block_size+row];
	    matrix_data[row*block_size+col] = matrix_data[col*block_size+row];
	 }
      }

   }
   /* 对于特征值问题, 初始特征向量进行变换 */
   if (eigenvectors!=NULL)
   {
      for ( row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorAxpy(eigenvectors[row]->aux_h->data[col], Z[col], eigenvectors[row]->b_H); 
	 }
      }
   }
   if (par_x!=NULL)
   {
      for ( col = 0; col < block_size; ++col)
      {
	 HYPRE_ParVectorAxpy(par_x->aux_h->data[col], Z[col], par_x->b_H); 
      }
   }
   if (par_b!=NULL)
   {
      for ( col = 0; col < block_size; ++col)
      {
	 HYPRE_ParVectorInnerProd(Z[col], par_b->b_H, &(ZF->data[col]));
      }
      hypre_SeqVectorAxpy(-1.0, ZF, par_b->aux_h);
   }

   return 0;
}

HYPRE_Int
hypre_PASEDiagBack(hypre_PASEDiag* diag_data, 
      PASE_ParCSRMatrix parcsr_A,     PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector*   eigenvectors, 
      PASE_ParVector    par_b,        PASE_ParVector    par_x)
{
   HYPRE_Int block_size, row, col;
   block_size = diag_data->block_size;

   HYPRE_Real*     matrix_data;
   hypre_CSRMatrix *XZ, *YZ, *ZBZ;
   hypre_Vector    *ZF;
   HYPRE_ParVector *Z, *BZ;

   XZ  = diag_data->XZ;
   YZ  = diag_data->YZ;
   ZBZ = diag_data->ZBZ;
   Z   = diag_data->Z;
   ZF  = diag_data->ZF;
   BZ  = diag_data->BZ;


   if (eigenvectors!=NULL)
   {
      for (row = 0; row < block_size; ++row)
      {
	 for (col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorAxpy(-(eigenvectors[row]->aux_h->data[col]), Z[col], eigenvectors[row]->b_H);
	 }
      }
   }
   if (par_x!=NULL)
   {
      for (col = 0; col < block_size; ++col)
      {
	 HYPRE_ParVectorAxpy(-(par_x->aux_h->data[col]), Z[col], par_x->b_H);
      }
   }
   if (par_b!=NULL)
   {
      hypre_SeqVectorAxpy(1.0, ZF, par_b->aux_h);
   }
   if (parcsr_A!=NULL)
   {
      /* XZ */
      matrix_data = XZ->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    parcsr_A->aux_hh->data[row*block_size+col] += matrix_data[row*block_size+col];
	 }
      }
      parcsr_A->diag = 0;
   }
   if (parcsr_B!=NULL)
   {
      /* +BZ */
      for (row = 0; row < block_size; ++row)
      {
	 HYPRE_ParVectorAxpy(1.0, BZ[row], parcsr_B->aux_hH[row]);
      }
      /* +YZ+ZY-ZBZ */
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    parcsr_B->aux_hh->data[row*block_size+col] += 
	       (YZ->data)[row*block_size+col]+(YZ->data)[col*block_size+row]-(ZBZ->data)[row*block_size+col];
	 }
      }
   }

   return 0;
}

HYPRE_Int
hypre_PASEDiagDestroy(hypre_PASEDiag* diag_data)
{
   HYPRE_Int block_size, row;
   block_size = diag_data->block_size;

   hypre_CSRMatrixDestroy(diag_data->XZ);
   if (diag_data->owns_data==1)
   {
      for ( row = 0; row < block_size; ++row)
      {
	 HYPRE_ParVectorDestroy(diag_data->Z[row]);
      }
      hypre_TFree(diag_data->Z);
   }
   else
   {
      diag_data->Z = NULL;
   }
   if (diag_data->BZ!=NULL)
   {
      for ( row = 0; row < block_size; ++row)
      {
	 HYPRE_ParVectorDestroy(diag_data->BZ[row]);
      }
      hypre_TFree(diag_data->BZ);

      hypre_CSRMatrixDestroy(diag_data->YZ);
      hypre_CSRMatrixDestroy(diag_data->ZBZ);
   }
   if (diag_data->ZF!=NULL)
   {
      hypre_SeqVectorDestroy(diag_data->ZF);
   }
   HYPRE_BoomerAMGDestroy(diag_data->precond);
//   HYPRE_ParCSRPCGDestroy(diag_data->precond);

   return 0;
}
