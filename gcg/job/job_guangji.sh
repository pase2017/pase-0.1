#!/bin/bash

#=============================================================================================================================================================
SLEPC_EXE="/home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/gcg/bin/exgcg_petscbin.exe"
SLEPC_MAT1="-mat1 /home/student/JiusuoSoftCenter/guangji/guangji.stiff.mat.petsc.bin"
SLEPC_MAT2="-mat2 /home/student/JiusuoSoftCenter/guangji/guangji.mass.mat.petsc.bin"
SLEPC_INIT="-init vec /home/zhangning/slepc-3.7.3/src/eps/examples/tutorials/mat/guangji/Init_guangji_nev10_171.txt"
SLEPC_OUTPUT=

SLEPC_EPS="-eigen_max_iter 0"
SLEPC_ST=
SLEPC_KSP="-cg_max_iter 100"
SLEPC_EPS_VAR="-eigen_tol 1e-4 -cg_rate 1e-1" 
SLEPC_LOG=
NP="20"

IF_PRINT="-print_eval 1 -print_cg 1 -if_giveninit 1"
SLEPC_EPS_NEV="-nev 10"
EXEC=${SLEPC_EXE}" "${SLEPC_MAT1}" "${SLEPC_MAT2}" "${SLEPC_INIT}" "${SLEPC_OUTPUT}" "${SLEPC_EPS}" "${SLEPC_EPS_NEV}" "${SLEPC_EPS_VAR}" "${SLEPC_ST}" "${SLEPC_KSP}" "${IF_PRINT}" "${SLEPC_LOG}
echo ""
echo ${EXEC}
#echo "/output/guangji/extendibility/$(date +%m%d.%H%M%S).guangji.${NP}.output" 
mpiexec -n ${NP} ${EXEC} > ../output/guangji/guangji_nev10_testcg.${NP}.output 2>&1 


