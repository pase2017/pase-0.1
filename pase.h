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

#ifndef _pase_h_
#define _pase_h_

#include "pase_hypre.h"
#include "pase_mg.h"

#include "pase_mv.h"
#include "pase_int.h"
#include "pase_pcg.h"
#include "pase_lobpcg.h"

#include "pase_ls.h"
#include "pase_es.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
}
#endif

#endif
