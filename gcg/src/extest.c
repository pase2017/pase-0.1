/*************************************************************************
	> File Name: extest.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Tue Jan  9 11:46:25 2018
 ************************************************************************/

#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/unistd.h>
#include <string.h>
#include <slepceps.h>
#include "SlepcCG.h"
#include "ReadWritePrint.h"
#include "SlepcGCGEigen.h"

int main(int argc, char* argv[])
{
    PetscInt  i, dim_x = 3, dim_xp = 6, dim_xpw = 9, length = dim_xpw*dim_xp;
    PetscReal *dense_mat_1 = (PetscReal*)calloc(length, sizeof(PetscReal*));
    PetscReal *dense_mat_2 = (PetscReal*)calloc(length, sizeof(PetscReal*));
    PetscReal *dense_mat_3 = (PetscReal*)calloc(length, sizeof(PetscReal*));
    PetscInt  *Ind = calloc(dim_xp, sizeof(PetscInt));
	srand((unsigned)time(NULL));
	for( i=0; i<length; i++ )
		dense_mat_1[i] = rand()%(length+1)/((double)length);
    memcpy(dense_mat_2, dense_mat_1, length*sizeof(PetscReal));
    memset(dense_mat_2+(dim_xp-1)*dim_xpw, 0.0, dim_xpw*sizeof(PetscReal));
    memcpy(dense_mat_3, dense_mat_2, (dim_xp-2)*dim_xpw*sizeof(PetscReal));
    memcpy(dense_mat_3+(dim_xp-1)*dim_xpw, dense_mat_2+(dim_xp-2)*dim_xpw, dim_xpw*sizeof(PetscReal));

    printf("mat_2:\n");
	for( i=0; i<length; i++ )
        printf("mat_2[%d] = %18.15lf\n", i, dense_mat_2[i]);
    printf("mat_3:\n");
	for( i=0; i<length; i++ )
        printf("mat_3[%d] = %18.15lf\n", i, dense_mat_3[i]);

    OrthogonalSmall(dense_mat_1, NULL, dim_xpw, dim_x, &dim_xp, Ind);
    printf("mat_1 orthogonal, dim_xp: %d\n", dim_xp);
	for( i=0; i<length; i++ )
        printf("mat_1[%d] = %18.15lf\n", i, dense_mat_1[i]);
    dim_xp = 6;

    OrthogonalSmall(dense_mat_2, NULL, dim_xpw, dim_x, &dim_xp, Ind);
    printf("mat_2 orthogonal, dim_xp: %d\n", dim_xp);
	for( i=0; i<length; i++ )
        printf("mat_2[%d] = %18.15lf\n", i, dense_mat_2[i]);
    dim_xp = 6;

    OrthogonalSmall(dense_mat_3, NULL, dim_xpw, dim_x, &dim_xp, Ind);
    printf("mat_3 orthogonal, dim_xp: %d\n", dim_xp);
	for( i=0; i<length; i++ )
        printf("mat_3[%d] = %18.15lf\n", i, dense_mat_3[i]);

    free(dense_mat_1);  dense_mat_1 = NULL;
    free(dense_mat_2);  dense_mat_2 = NULL;
    free(dense_mat_3);  dense_mat_3 = NULL;
    free(Ind);  Ind = NULL;

    return 0;
}
