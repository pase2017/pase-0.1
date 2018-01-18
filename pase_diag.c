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


typedef struct
{
   hypre_CSRMatrix *bTy;
   hypre_CSRMatrix *cTy; 
   hypre_CSRMatrix *yTBy;
   
} hypre_PASEDiag;

/* 
 * (S^T A_Hh S) S^-1  x_Hh  = lambda (S^T B S) S^-1 x_Hh
 *
 * S = E   -y
 *     0    E
 *
 * y = A^-1 b
 *
 * A    b      A    0
 * b^T  a  --> 0^T  a-b^T y
 *
 * B    c      B          c-By
 * c^T  f  --> (c-By)^T   f-c^T y - y^T c + y^T B y      
 *
 *
 * S x = x_Hh
 *
 * E  -y   z   =   z - ym
 * 0   E   m       m
 *
 * */
hypre_PASEDiagCreate()
{
   hypre_CSRMatrix *bTy, *cTy, *yTBy;
   bTy  = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   cTy  = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   yTBy = hypre_CSRMatrixCreate(block_size,  block_size,  block_size*block_size);
   hypre_CSRMatrixInitialize( bTy );
   hypre_CSRMatrixInitialize( cTy );
   hypre_CSRMatrixInitialize( yTBy);
   HYPRE_Int*  matrix_i = bTy->i;
   HYPRE_Int*  matrix_j = bTy->j;
   HYPRE_Real* matrix_data;
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
   hypre_CSRMatrixCopy(bTy, cTy,  0);
   hypre_CSRMatrixCopy(bTy, yTBy, 0);

   HYPRE_ParVector   *B_pvx_H;
   mv_MultiVectorPtr B_eigenvectors_H = NULL;
   {
      B_eigenvectors_H = 
	 mv_MultiVectorCreateFromSampleVector(interpreter_H, block_size, par_x_H);
      mv_TempMultiVector* tmp = 
	 (mv_TempMultiVector*) mv_MultiVectorGetData(B_eigenvectors_H);
      B_pvx_H = (HYPRE_ParVector*)(tmp -> vector);
   }
   HYPRE_BoomerAMGSetup(precond, parcsr_A_H, par_x_H, par_b_H);
}


hypre_PASEDiagChange()
{
   /* 对角化 */
   {
      /* 求解线性方程组 */
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 HYPRE_BoomerAMGSolve(precond, parcsr_A_Hh->A_H, parcsr_A_Hh->aux_hH[idx_eig], pvx_H[idx_eig]);
      }

      /* bTy */
      matrix_data = bTy->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorInnerProd(parcsr_A_Hh->aux_hH[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	    parcsr_A_Hh->aux_hh->data[row*block_size+col] =
	       parcsr_A_Hh->aux_hh->data[row*block_size+col] - matrix_data[row*block_size+col];
	 }
      }
      /* cTy */
      matrix_data = cTy->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorInnerProd(parcsr_B_Hh->aux_hH[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	    parcsr_B_Hh->aux_hh->data[row*block_size+col] =
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] - matrix_data[row*block_size+col];
	 }
      }
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    parcsr_B_Hh->aux_hh->data[row*block_size+col] =
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] - matrix_data[col*block_size+row];
	 }
      }
      /* By */
      for (row = 0; row < block_size; ++row)
      {
	 HYPRE_ParCSRMatrixMatvec ( 1.0, parcsr_B_Hh->A_H, pvx_H[row], 0.0, B_pvx_H[row] );
	 HYPRE_ParVectorAxpy(-1.0, B_pvx_H[row], parcsr_B_Hh->aux_hH[row]);
      }
      /* yTBy */
      matrix_data = yTBy->data;
      for (row = 0; row < block_size; ++row)
      {
	 for ( col = 0; col < block_size; ++col)
	 {
	    HYPRE_ParVectorInnerProd(B_pvx_H[row],  pvx_H[col], &matrix_data[row*block_size+col]);
	    parcsr_B_Hh->aux_hh->data[row*block_size+col] =
	       parcsr_B_Hh->aux_hh->data[row*block_size+col] + matrix_data[row*block_size+col];
	 }
      }

      for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 PASE_ParVectorSetConstantValues(pvx_Hh[idx_eig], 0.0);
	 HYPRE_ParVectorCopy(pvx_H[idx_eig], pvx_Hh[idx_eig]->b_H); 
	 pvx_Hh[idx_eig]->aux_h->data[idx_eig] = 1.0;
      }
   }
   /* 对角化 */


}
hypre_PASEDiagBack
{
   /* 对角化 */
   {
      for (idx_eig = 0; idx_eig < block_size; ++idx_eig)
      {
	 for (row = 0; row < block_size; ++row)
	 {
	    HYPRE_ParVectorAxpy(-(pvx_Hh[idx_eig]->aux_h->data[row]), pvx_H[row], pvx_Hh[idx_eig]->b_H);
	 }
      }
   }
   /* 对角化 */

}

hypre_PASEDiagDestroy()
{
   /* 对角化 */
   for ( idx_eig = 0; idx_eig < block_size; ++idx_eig)
   {
      HYPRE_ParVectorDestroy(B_pvx_H[idx_eig]);
   }
   hypre_TFree(B_pvx_H);
   free((mv_TempMultiVector*) mv_MultiVectorGetData(B_eigenvectors_H));
   hypre_TFree(B_eigenvectors_H);
   hypre_CSRMatrixDestroy(bTy);
   hypre_CSRMatrixDestroy(cTy);
   hypre_CSRMatrixDestroy(yTBy);

}


