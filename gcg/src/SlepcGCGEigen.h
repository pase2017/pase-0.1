/*************************************************************************
	> File Name: SlepcGCGEigen.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:41:40 2018
 ************************************************************************/


#include <slepceps.h>
#include <petscblaslapack.h>
#include <petsctime.h>
#include <petsc/private/vecimpl.h>
#include "SlepcCG.h"

#define REORTH_TOL 0.75 

PetscErrorCode GCG_Eigen(Mat A, Mat B, PetscReal *eval, Vec *evec, PetscInt nev, PetscReal abs_tol, PetscReal cg_tol, PetscInt nsmooth, PetscInt max_iter, PetscInt MAXMIN);

//用Petsc的矩阵和向量操作构造的函数
PetscErrorCode GetRandomInitValue(Vec *V, PetscInt dim_x);
PetscErrorCode AllocateVecs(Mat A, Vec **evec, Vec **V_1, Vec **V_2, Vec **V_3, PetscInt nev, PetscInt a, PetscInt b, PetscInt c);
PetscErrorCode VecsMatrixVecsForRayleighRitz(Mat A, Vec *V, PetscReal *AA, PetscInt dim_xp, PetscInt dim_xpw, Vec tmp);
PetscErrorCode RayleighRitz(PetscInt MAXMIN, Mat A, Vec *V, PetscReal *AA, PetscReal *approx_eval, PetscReal *AA_sub, PetscReal *AA_copy, PetscInt dim_x, PetscInt start, PetscInt last_dim, PetscInt dim, Vec tmp, PetscReal *small_tmp, PetscReal *time);
PetscErrorCode GetRitzVectors(PetscReal *SmallEvec, Vec *V, Vec *RitzVec, PetscInt dim, PetscInt nev, PetscReal *time, PetscInt if_time);
void ChangeVecPointer(Vec *V_1, Vec *V_2, Vec *tmp, PetscInt size);
PetscErrorCode SumSeveralVecs(Vec *V, PetscReal *x, Vec U, PetscInt n_vec);
PetscErrorCode GCG_Orthogonal(Vec *V, Mat A, Mat M, PetscInt MAXMIN, PetscInt start, PetscInt *end, Vec *V_tmp, Vec *Zero_Vec, PetscInt *Ind, PetscReal *time);
PetscReal VecMatrixVec(Vec a, Mat Matrix, Vec b, Vec temp);

//小规模的向量或稠密矩阵操作，这些应该是串行的，所以没有改动
void OrthogonalSmall(PetscReal *V, PetscReal **B, PetscInt dim_xpw, PetscInt dim_x, PetscInt *dim_xp, PetscInt *Ind);
void DenseMatVec(PetscReal *DenseMat, PetscReal *x, PetscReal *b, PetscInt dim);
void DenseVecsMatrixVecs(PetscReal *LVecs, PetscReal *DenseMat, PetscReal *RVecs, PetscReal *ProductMat, PetscInt nl, PetscInt nr, PetscInt dim, PetscReal *tmp);
void ScalVecSmall(PetscReal alpha, PetscReal *a, PetscInt n);
PetscReal NormVecSmall(PetscReal *a, PetscInt n);
PetscReal VecDotVecSmall(PetscReal *a, PetscReal *b, PetscInt n);
void SmallAXPBY(PetscReal alpha, PetscReal *a, PetscReal beta, PetscReal *b, PetscInt n);


//Petsc CG算法
PetscErrorCode LinearSolver(Mat A, Vec b, Vec x, PetscReal cg_tol, PetscInt nsmooth);
PetscErrorCode ComputeAxbResidual(Mat A, Vec x, Vec b, Vec tmp, PetscReal *res);

PetscErrorCode GetLAPACKMatrix(Mat A, Vec *V, PetscReal *AA, PetscReal *AA_sub, PetscInt start, PetscInt last_dim, PetscInt dim, PetscReal *AA_copy, Vec tmp, PetscReal *small_tmp);
PetscErrorCode GetWinV(PetscInt start, PetscInt nunlock, PetscInt *unlock, Vec *V, PetscReal *approx_eval, Mat A, Mat B, PetscReal cg_tol, PetscInt nsmooth, Vec *V_tmp, PetscReal *time);
PetscErrorCode CheckConvergence(Mat A, Mat B, PetscInt *unlock, PetscInt *nunlock, PetscInt nev, Vec *X_tmp, PetscReal *approx_eval, PetscReal abs_tol, Vec *V_tmp, PetscInt iter, PetscReal *RRes);
PetscErrorCode GetPinV(PetscReal *AA, Vec *V, PetscInt dim_x, PetscInt *dim_xp, PetscInt dim_xpw, PetscInt nunlock, PetscInt *unlock, Vec *V_tmp, Vec *Orth_tmp, PetscInt *Ind, PetscReal *time);
void GetXinV(Vec *V, Vec *X_tmp, Vec *tmp, PetscInt dim_x);
void Updatedim_x(PetscInt start, PetscInt end, PetscInt *dim_x, PetscReal *approx_eval);
void PrintSmallEigen(PetscInt iter, PetscInt nev, PetscReal *approx_eval, PetscReal *AA, PetscInt dim, PetscReal *RRes);

PetscInt petscmax(PetscInt a, PetscInt b);
