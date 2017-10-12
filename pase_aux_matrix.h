#ifndef __PASE_AUX_MATRIX_H__
#define __PASE_AUX_MATRIX_H__

#include "pase_matrix.h"

//typedef struct PASE_AUX_MATRIX_OPERATOR_PRIVATE_ {
//
//} PASE_AUX_MATRIX_OPERATOR_PRIVATE; 
//typedef PASE_AUX_MATRIX_OPERATOR_PRIVATE * PASE_AUX_MATRIX_OPERATOR;

/*
 * aux matrix = [mat   vec    ]
 *              [vecT  block2d]
 */
typedef struct PASE_AUX_MATRIX_PRIVATE_ {
  PASE_MATRIX *mat;
  PASE_VECTOR *vec;
  PASE_VECTOR *vecT;
  PASE_SCALAR *block2d;
  //PASE_AUX_MATRIX_OPERATOR ops;
  PASE_MATRIX_OPERATOR ops;
} PASE_AUX_MATRIX_PRIVATE;
typedef PASE_AUX_MATRIX_PRIVATE * PASE_AUX_MATRIX;

PASE_AUX_MATRIX PASE_Create_aux_matrix(PASE_MATRIX A, PASE_PARAMETER param);
void PASE_Destroy_aux_matrix(PASE_AUX_MATRIX aux_A);
//void PASE_Matrix_multiply_matrix_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C);
//void PASE_Matrix_multiply_matrix_matrix_hypre(PASE_MATRIX A, PASE_MATRIX B, PASE_MATRIX C);
PASE_MATRIX *CopyPaseMatrix(PASE_MATRIX *pase_mat);

#endif
