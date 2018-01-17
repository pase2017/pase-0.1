#!/bin/bash

#=============================================================================================================================================================
SLEPC_EXE="/home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/gcg/bin/exgcg_petscbin.exe"
SLEPC_MAT1="-mat1 /home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/mat/fem5/FEM_Stiff_5.petsc.bin"
SLEPC_MAT2="-mat2 /home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/mat/fem5/FEM_Mass_5.petsc.bin"
SLEPC_INIT="-init_vec /home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/mat/fem5/randx_5.txt"
SLEPC_OUTPUT=

SLEPC_EPS="-eigen_max_iter 50"
SLEPC_ST=
SLEPC_KSP="-cg_max_iter 20"
SLEPC_EPS_VAR="-eigen_tol 1e-8 -cg_rate 1e-3" 
SLEPC_LOG=
NP="1"

IF_PRINT="-print_eval 1 -print_cg 0 -if_giveninit 1"
SLEPC_EPS_NEV="-nev 3"
EXEC=${SLEPC_EXE}" "${SLEPC_MAT1}" "${SLEPC_MAT2}" "${SLEPC_INIT}" "${SLEPC_OUTPUT}" "${SLEPC_EPS}" "${SLEPC_EPS_NEV}" "${SLEPC_EPS_VAR}" "${SLEPC_ST}" "${SLEPC_KSP}" "${IF_PRINT}" "${SLEPC_LOG}
echo ""
echo ${EXEC}
#echo "/output/guangji/extendibility/$(date +%m%d.%H%M%S).guangji.${NP}.output" 
mpiexec -n ${NP} ${EXEC} > ../output/fem5/19_fem5_nev3.${NP}.output 2>&1 


