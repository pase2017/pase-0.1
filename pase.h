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

typedef struct
{
   PASE_Int    (*PrecondSetup)(void*,void*,void*,void*);
   PASE_Int    (*PrecondSolve)(void*,void*,void*,void*);

   PASE_Int    (*LinearSetup)(void*,void*,void*,void*);
   PASE_Int    (*LinearSolve)(void*,void*,void*,void*);

   PASE_Int    (*EigenCreate)(void*,void*,void*,void*);
   PASE_Int    (*EigenSetup)(void*,void*,void*,void*);
   PASE_Int    (*EigenSolve)(void*,void*,void*,void*);
   PASE_Int    (*EigenDestroy)(void*);

   /* y = alpha A x + beta y */
   PASE_Int    (*OperatorA)(void*, void*, void*, void*);
   /* y = alpha B x + beta y */
   PASE_Int    (*OperatorB)(void*, void*, void*, void*);
   
   PASE_Int    (*FromItoJ)(void*, void*, void*, void*);

} hypre_PASEFunctions;

#define hypre_PASEDataInterpreterHYPYE(data)  (data->interpreter_hypre)
#define hypre_PASEDataInterpreterPASE(data)   (data->interpreter_pase)
#define hypre_PASEDataMatVecFnHYPRE(data)     (data->matvec_fn_hypre)
#define hypre_PASEDataMatVecFnPASE(data)      (data->matvec_fn_pase)
#define hypre_PASEDataEigenSolver(data)       (data->eigen_solver)
#define hypre_PASEDataLinearSolver(data)      (data->linear_solver)
#define hypre_PASEDataPrecond(data)           (data->precond)

typedef struct
{
   hypre_CSRMatrix *bTy;
   hypre_CSRMatrix *cTy; 
   hypre_CSRMatrix *yTBy;
   
} hypre_PASEDiag;

/* 
 * (S^T A_Hh S) S^-1  x_Hh  = lambda (S^T B S) S^-1 x_Hh
 *
 * S = E   -y
 *     0    E
 *
 * y = A^-1 b
 *
 * A    b      A    0
 * b^T  a  --> 0^T  a-b^T y
 *
 * B    c      B          c-By
 * c^T  f  --> (c-By)^T   f-c^T y - y^T c + y^T B y      
 *
 *
 * S x = x_Hh
 *
 * E  -y   z   =   z - ym
 * 0   E   m       m
 *
 * */

typedef struct
{
   
   PASE_MultiGrid              multi_grid;

   HYPRE_Solver                linear_solver;
   HYPRE_Solver                eigen_solver;
   HYPRE_Solver                precond; /* 对角化 */

   HYPRE_MatvecFunctions*      matvec_fn_hypre;
   HYPRE_MatvecFunctions*      matvec_fn_pase;

   mv_InterfaceInterpreter*    interpreter_hypre;
   mv_InterfaceInterpreter*    interpreter_pase;

   hypre_PASEFunctions         pase_functions;

   hypre_PASEDiag              pase_diag;

   PASE_ParCSRMatrix           A_pase;
   PASE_ParCSRMatrix           B_pase;
   mv_MultiVectorPtr           mv_cons_pase;
   mv_MultiVectorPtr           mv_eigs_pase;
   PASE_ParVector*             vecs_pase;
   PASE_ParVector*             cons_pase;
   PASE_ParVector              b_pase;
   PASE_ParVector              x_pase;

   /* 最粗空间求解特征值问题, 以及对角化时用到vecs_hypre */
   HYPRE_ParCSRMatrix          A_hypre;
   HYPRE_ParCSRMatrix          B_hypre;
   mv_MultiVectorPtr           mv_cons_hypre;
   mv_MultiVectorPtr           mv_eigs_hypre;
   HYPRE_ParVector*            vecs_hypre;
   HYPRE_ParVector*            cons_hypre;
   HYPRE_ParVector             b_hypre;
   HYPRE_ParVector             x_hypre;

   PASE_Int                    block_size;
   PASE_Int                    max_levels;
   PASE_Int                    max_iters;
   PASE_Int                    num_iters;
   PASE_Int                    verb_level;

   PASE_Real                   absolute_tol;
   PASE_Real                   relative_tol;
   utilities_FortranMatrix*    eigenvalues_history;
   utilities_FortranMatrix*    residual_norms;
   utilities_FortranMatrix*    residual_norms_history;

} hypre_PASEData;



PASE_Int
HYPRE_PASECreate( HYPRE_Solver* solver );

PASE_Int 
HYPRE_PASEDestroy( HYPRE_Solver solver );

PASE_Int 
HYPRE_PASESetup( HYPRE_Solver  solver,
                 HYPRE_ParCSRMatrix A,
                 HYPRE_ParCSRMatrix B,
                 HYPRE_ParVector    b,
                 HYPRE_ParVector    x);

PASE_Int
HYPRE_PASESolve( HYPRE_Solver      solver, 
                 HYPRE_Int         block_size,
                 HYPRE_ParVector*  con, 
                 HYPRE_ParVector*  vec, 
                 HYPRE_Real*       val );

HYPRE_Int 
HYPRE_PASESetTol( HYPRE_Solver solver, HYPRE_Real tol);
HYPRE_Int 
HYPRE_PASESetMaxLevels( HYPRE_Solver solver, HYPRE_Int max_levels);
HYPRE_Int 
HYPRE_PASESetAMG( HYPRE_Solver solver, HYPRE_Solver amg);

HYPRE_Int 
HYPRE_PASESetLinearSolver( HYPRE_Solver solver, 
      HYPRE_PtrToSolverFcn linear, 
      HYPRE_PtrToSolverFcn linear_setup, 
      HYPRE_Solver linear_solver);

HYPRE_Int 
hypre_PASESetLinearSolver(void* solver,
      HYPRE_Int  (*linear)(void*, void*, void*, void*), 
      HYPRE_Int  (*linear_setup)(void*, void*, void*, void*),
      void* linear_solver);


#ifdef __cplusplus
}
#endif

#endif
