/*************************************************************************
  > File Name: ex_gcg.c
  > Author: nzhang
  > Mail: zhangning114@lsec.cc.ac.cn
  > Created Time: Tue Dec 19 10:35:45 2017
 ************************************************************************/

#include <slepceps.h>
#include <petscblaslapack.h>
#include <petsctime.h>
#include <petsc/private/vecimpl.h>
#include "SlepcCG.h"
#include "ReadWritePrint.h"
#include "SlepcGCGEigen.h"

static char help[] = "Solves a generalized eigensystem Ax=kBx with matrices loaded from a file.\n"
"The COMMand line options are:\n"
"  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B.\n"
"  -evecs <filename>, output file to save computed eigenvectors.\n"
"  -ninitial <nini>, number of user-provided initial guesses.\n"
"  -finitial <filename>, binary file containing <nini> vectors.\n"
"  -nconstr <ncon>, number of user-provided constraints.\n"
"  -fconstr <filename>, binary file containing <ncon> vectors.\n\n";



int main(int argc, char* argv[])
{
    Mat            A, B;           /* problem matrix */
    Vec            *evec, vec_tmp;
    PetscErrorCode ierr;
    PetscInt       eigen_max_iter = 50, nev = 3, i, cg_max_iter = 20, if_giveninit = 0;
    PetscReal      *eval, eigen_tol = 1e-8, cg_rate = 1e-3;
    PetscLogDouble t2,t3;

    SlepcInitialize(&argc,&argv,(char*)0,help);

    ierr = GenerateFDMatrix(&A, &B);

    ierr = PetscOptionsGetInt(NULL,NULL,"-nev",&nev,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-eigen_max_iter",&eigen_max_iter,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-cg_max_iter",&cg_max_iter,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-eigen_tol",&eigen_tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-cg_rate",&cg_rate,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-if_giveninit",&if_giveninit,NULL);CHKERRQ(ierr);
    //cg_rate = eigen_tol*(1e-6);
	PetscPrintf(PETSC_COMM_WORLD, "nev: %d, eigen_max_iter: %d, cg_max_iter: %d, eigen_tol: %e, cg_rate: %e\n",
			nev, eigen_max_iter, cg_max_iter, eigen_tol, cg_rate);
    eval = (PetscReal*)calloc(nev,sizeof(PetscReal));
    //evec = (Vec*)malloc(nev*sizeof(Vec));
    //PetscPrintf(PETSC_COMM_WORLD, "line 50\n");
    ierr = MatCreateVecs(A,NULL,&vec_tmp);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD, "line 52\n");
    ierr = VecDuplicateVecs(vec_tmp,nev,&evec);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_tmp);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD, "line 54\n");
    ierr = GetRandomInitValue(evec, nev);CHKERRQ(ierr);//krylovschur怎么取的随机初值
    //PetscPrintf(PETSC_COMM_WORLD, "line 56\n");
    if(if_giveninit == 1)
    {
        ierr = ReadVec(evec, nev);CHKERRQ(ierr);
    }
    //ierr = WriteVec(V, nev, "../mat/guangji/Init_guangji_nev1.txt");CHKERRQ(ierr);

    ierr = PetscTime(&t2);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "line 61\n");
    ierr = GCG_Eigen(A, B, eval, evec, nev, eigen_tol, cg_rate, cg_max_iter, eigen_max_iter);CHKERRQ(ierr);
    ierr = PetscTime(&t3);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Total time: %lf\n", (double)(t3-t2));

    VecDestroyVecs(nev, &evec);
    /*
    for(i=0;i<nev;i++)
    {
        ierr = VecDestroy(&(evec[i]));CHKERRQ(ierr);
    }
    ierr = PetscFree(evec);CHKERRQ(ierr);
    */
    //free(evec);  evec = NULL;
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    free(eval);  eval = NULL;
    ierr = SlepcFinalize();
    return ierr;
}

PetscErrorCode GetRandomInitValue(Vec *V, PetscInt dim_x)
{
    PetscRandom    rctx;
    PetscErrorCode ierr;
    PetscInt       i;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    for( i=0; i<dim_x; i++ )
    {
        ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
