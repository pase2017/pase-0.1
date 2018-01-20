
#ifndef _pase_diag_h_
#define _pase_diag_h_


#include "pase_int.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
   HYPRE_Int       block_size;
   HYPRE_Solver    precond;

   hypre_CSRMatrix *XZ;
   hypre_CSRMatrix *YZ; 
   hypre_CSRMatrix *ZBZ;

   hypre_Vector    *ZF;

   HYPRE_ParVector *BZ;
   HYPRE_ParVector *Z;

   HYPRE_Int        owns_data;     
   
} hypre_PASEDiag;


HYPRE_Int
hypre_PASEDiagCreate(hypre_PASEDiag* diag_data, 
      PASE_ParCSRMatrix parcsr_A, PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector    par_b,    PASE_ParVector    par_x, 
      HYPRE_ParVector  *Z ,       HYPRE_ParVector sample);

HYPRE_Int
hypre_PASEDiagChange(hypre_PASEDiag*  diag_data, 
      PASE_ParCSRMatrix parcsr_A,     PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector*   eigenvectors, 
      PASE_ParVector    par_b,        PASE_ParVector    par_x);

HYPRE_Int
hypre_PASEDiagBack(hypre_PASEDiag* diag_data, 
      PASE_ParCSRMatrix parcsr_A,     PASE_ParCSRMatrix parcsr_B, 
      PASE_ParVector*   eigenvectors, 
      PASE_ParVector    par_b,        PASE_ParVector    par_x);

HYPRE_Int
hypre_PASEDiagDestroy(hypre_PASEDiag* diag_data);

#ifdef __cplusplus
}
#endif


#endif
