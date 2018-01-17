/*************************************************************************
	> File Name: SlepcCG.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:39:34 2018
 ************************************************************************/

#include <slepceps.h>
#include <petsctime.h>
#define EPS 2.220446e-16

PetscErrorCode SLEPCCG(Mat Matrix, Vec b, Vec x, PetscReal accur, PetscInt Max_It, Vec *V_tmp);
PetscErrorCode SLEPCCG_2(Mat Matrix, Vec b, Vec x, PetscReal rate, PetscInt Max_It, Vec *V_tmp);
