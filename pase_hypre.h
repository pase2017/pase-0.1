/*
 * =====================================================================================
 *
 *       Filename:  pash.h
 *
 *    Description:  后期可以考虑将行参类型都变成void *, 以方便修改和在不同计算机上调试
 *                  一般而言, 可以让用户调用的函数以PASE_开头, 内部函数以pase_开头
 *
 *        Version:  1.0
 *        Created:  2017年08月29日 14时15分22秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  LIYU 
 *   Organization:  LSEC
 *
 * =====================================================================================
 */

#ifndef _pase_hypre_h_
#define _pase_hypre_h_

#include "HYPRE_seq_mv.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"


#define PASE_Int      HYPRE_Int
#define PASE_Real     HYPRE_Real
#define PASE_Solver   HYPRE_Solver


#ifdef __cplusplus
extern "C" {
#endif


HYPRE_Int hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x );

#ifdef __cplusplus
}
#endif

#endif
