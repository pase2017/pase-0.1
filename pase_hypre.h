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
HYPRE_Int hypre_LOBPCGSolve( HYPRE_Int num_lock, void *vdata, 
      mv_MultiVectorPtr con, mv_MultiVectorPtr vec, HYPRE_Real* val );
HYPRE_Int hypre_LOBPCGSetPrecond( void  *pcg_vdata,
      HYPRE_Int  (*precond)(void*,void*,void*,void*),
      HYPRE_Int  (*precond_setup)(void*,void*,void*,void*),
      void  *precond_data );


typedef struct
{
   HYPRE_Int    (*Precond)(void*,void*,void*,void*);
   HYPRE_Int    (*PrecondSetup)(void*,void*,void*,void*);

} hypre_LOBPCGPrecond;

typedef struct
{
   lobpcg_Tolerance              tolerance;
   HYPRE_Int                           maxIterations;
   HYPRE_Int                           verbosityLevel;
   HYPRE_Int                           precondUsageMode;
   HYPRE_Int                           iterationNumber;
   utilities_FortranMatrix*      eigenvaluesHistory;
   utilities_FortranMatrix*      residualNorms;
   utilities_FortranMatrix*      residualNormsHistory;

} lobpcg_Data;

#define lobpcg_tolerance(data)            ((data).tolerance)
#define lobpcg_absoluteTolerance(data)    ((data).tolerance.absolute)
#define lobpcg_relativeTolerance(data)    ((data).tolerance.relative)
#define lobpcg_maxIterations(data)        ((data).maxIterations)
#define lobpcg_verbosityLevel(data)       ((data).verbosityLevel)
#define lobpcg_precondUsageMode(data)     ((data).precondUsageMode)
#define lobpcg_iterationNumber(data)      ((data).iterationNumber)
#define lobpcg_eigenvaluesHistory(data)   ((data).eigenvaluesHistory)
#define lobpcg_residualNorms(data)        ((data).residualNorms)
#define lobpcg_residualNormsHistory(data) ((data).residualNormsHistory)

typedef struct
{

   lobpcg_Data                   lobpcgData;

   mv_InterfaceInterpreter*      interpreter;

   void*                         A;
   void*                         matvecData;
   void*                         precondData;

   void*                         B;
   void*                         matvecDataB;
   void*                         T;
   void*                         matvecDataT;

   hypre_LOBPCGPrecond           precondFunctions;

   HYPRE_MatvecFunctions*        matvecFunctions;

} hypre_LOBPCGData;






#ifdef __cplusplus
}
#endif

#endif
