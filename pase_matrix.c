#include "pase_matrix.h"


/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX PASE_Create_matrix(void *matrix_data, PASE_PARAMETER param, PASE_MATRIX_OPERATOR ops)
{
    PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX));
    if(NULL != ops) {
        A->ops = ops;
    } else {
        if(param.matrix_internal_type == HYPRE_MATRIX_)  {
            A->ops = PASE_Set_hypre_matrix_operators();
        }
    }
    if(param.copy_matrix > 0) {
        A->matrix_data = A->ops->copy_matrix_data(matrix_data);
        A->is_matrix_data_owner = 1;
    } else {
        A->matrix_data = matrix_data;
        A->is_matrix_data_owner = 0;
    }
    A->nrow = A->ops->get_nrow(A);
    A->ncol = A->ops->get_ncol(A);
    return A;
}

void PASE_Destroy_matrix(PASE_MATRIX A)
{
    if(A->is_matrix_data_owner > 0) {
        A->ops->free_matrix_data(A->matrix_data);
        A->matrix_data = NULL;
    }
    PASE_Free(A);
    A = NULL;
}

/**
 * @brief C = A * B
 */
void PASE_Matrix_multiply_matrix_matrix(PASE_MATRIX A, PASE_MATRIX B, PASE_MATRIX C)
{
    A->operator->multiply_matrix_matrix(A, B, C);
}

void PASE_Matrix_multiply_matrix_matrix_hypre(PASE_MATRIX A, PASE_MATRIX B, PASE_MATRIX C)
{
    if((A->ncol == B->nrow) && (A->nrow == C->nrow) && (B->ncol == C->ncol)) {
        HYPRE_Matrix_multiply_matrix_matrix(A->matrix_data, B->matrix_data, C->matrix_data);
    } else {
        printf("PASE ERROR: Matrix dimensions must be matched.\n");
        exit(-1);
    }
}

/*
void PASE_Matrix_multiply_petsc(PASE_MATRIX A, PASE_MATRIX B, PASE_MATRIX C)
{
    Petsc_Matrix_multiply(A->matrix_data, B->matrix_data, C->matrix_data);
}
*/

/* 拷贝 PASE_MATRIX 变量 */
PASE_MATRIX *CopyPaseMatrix(PASE_MATRIX *pase_mat);
