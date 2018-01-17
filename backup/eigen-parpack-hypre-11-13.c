/**
 *       @file  eigen-parpack-hypre.c
 *      @brief  用PARPACK求解特征值问题
 *
 *     @author  Li Yu (), liyu@lsec.cc.ac.cn
 *
 *   @internal
 *     Created  10/17/2012
 *    Revision  $Id: doxygen.templates,v 1.3 2010/07/06 09:20:12 mehner Exp $
 *    Compiler  gcc/g++
 *     Company  LSEC CAS
 *   Copyright  Copyright (c) 2012, Li Yu
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include "template.h"

static int myid, nprocs;

typedef unsigned int LOGICAL;   /*  FORTRAN LOGICAL */ 
#define F77_True        1
#define F77_False       0

enum {SHIFT_INVERT, REGULAR_INVERT};
static const int mode_list[] = {SHIFT_INVERT, REGULAR_INVERT};
static const char *mode_name[] = {"shift-invert", "regular-invert", NULL};
static int mode_index = 0;	/* default = shift-invert */
//static bool keep_unconverged = false; /* whether keep unconverged evals */
static int mode;		/* will be set to mode_list[mode_index] */

/* 记录revers communication interface 返回的WORKD(IPNTR(1))，WORKD(IPNTR(2)) */
static PASE_ParVector va, vb;
extern void F77name(pdsaupd) (
      MPI_Comm *comm, 
      int   *ido,		/* what to do next */
      const char *bmat,	/* type of the matrix B ('I' or 'G') */
      int   *n,		/* dimension of the eigenproblem */
      const char *which,	/* 'LA' (algebraicly largest),
				 * 'SA' (algebraicly smallest),
				 * 'LM' (largest in magnitude),
				 * 'SM' (smallest in magnitude),
				 * 'BE' (half from each end of the spectrum) */
      int    *nev,	/* number of eigenvalues (Ritz values) to compute */
      double *tol,	/* relative accuracy (<=0 => use default) */
      double resid[],	/* residual vector, array of length N */
      int    *ncv,	/* number of columns of the matrix V (<= N) */
      double v[],		/* the Lanczos basis vectors, N by NCV array (out) */
      int    *ldv,	/* leading dimension of V */
      int    iparam[11],	/* parameters */
      int    ipntr[11],	/* starting locations in the WORKD and WORKL */
      double workd[],	/* work array of length 3*N */
      double workl[],	/* work array of length LWORKL */
      int    *lworkl,	/* sizeof WORKL, at least NCV**2 + 8*NCV */
      int    *info	/* Input:
			 *	0:  a randomly initial residual vector is used
			 *	!0: RESID contains the initial residual vector
			 * Output:
			 *	0: normal exit
			 *	1: maximum number of iterations taken.
			 *	... ... */
);

extern void F77name(pdseupd) (
      MPI_Comm *comm, 
      LOGICAL *rvec,	/* FALSE: Ritz values only, TRUE: Ritz vectors */
      const char *howmny,	/* 'A': all, 'S': some (specified by SELECT[]) */
      LOGICAL select[],	/* logical array of dimension NCV */
      double d[],		/* eigenvalues, array of dimension NEV (output) */
      double z[],		/* Ritz vectors, N by NEV array (may use V) */
      int    *ldz,	/* leading dimension of Z */
      double *sigma,	/* represents the shift */
      /* the following arguments are the same as for pdsaupd */
      const char *bmat,
      int    *n,
      const char *which,
      int    *nev,
      double *tol,
      double resid[],
      int    *ncv,
      double v[],
      int    *ldv,
      int    iparam[11],
      int    ipntr[11],
      double workd[],
      double workl[],
      int    *lworkl,
      int    *info);

/**
 * @brief y = inv[matB]*matA*x
 *
 * 这里solver中的矩阵和matA都是全局的，但x, y都是分布式的
 *
 * 用于reverse communication interface
 * if (solver==NULL) y = matA*x
 * if (matA==NULL) matA是单位阵
 *
 * SolverCreate已经被完成，但该函数可能更改一下地方
 * 以va为右端项，matB为矩阵，va为右端项，y存储解向量
 *
 * @param solver
 * @param matA
 * @param x
 * @param y
 *
 * @return 
 */
/* TODO: 是不是需要将通讯子 MPI_COMM_WORLD 传入*/
   static int 
pase_mat_vec (MPI_Comm MPI_COMM_WORLD, double *py, PASE_ParCSRMatrix B_Hh, HYPRE_Solver *solver, PASE_ParCSRMatrix A_Hh, double *px) 
{
   /** 利用va,vb将x,y封装成VECTOR */
   hypre_VectorData(hypre_ParVectorLocalVector(va->b_H)) = px;
   if (myid == nprocs - 1){
      hypre_VectorData(va->aux_h) = &px[va->N_H];
   }
   MPI_Bcast(hypre_VectorData(va->aux_h), va->block_size, HYPRE_Real, nprocs-1, MPI_COMM_WORLD);

   hypre_VectorData(hypre_ParVectorLocalVector(vb->b_H)) = py;
   if (myid == nprocs - 1){
      hypre_VectorData(vb->aux_h) = &px[vb->N_H];
   }
   MPI_Bcast(hypre_VectorData(vb->aux_h), vb->block_size, HYPRE_Real, nprocs-1, MPI_COMM_WORLD);

   if (A_Hh==NULL && solver==NULL)
   {
      PASE_ParVectorCopy(va, vb);
   }

   if (A_Hh!=NULL) 
   {
      /* y = alpha Ax + beta y */
      PASE_ParCSRMatrixMatvec(1.0, A_Hh, va, 0.0, vb);
   }

   if (solver != NULL) {
      /* copy vb (A*x) to va, va = vb */
      if ( A_Hh!=NULL) {
	 PASE_ParVectorCopy(vb, va);
      }

      hypre_PCGSolve(solver, B_Hh, va, vb);
   }

   hypre_VectorData(hypre_ParVectorLocalVector(va->b_H)) = NULL;
   hypre_VectorData(va->aux_h) = NULL;
   hypre_VectorData(hypre_ParVectorLocalVector(vb->b_H)) = NULL;
   hypre_VectorData(vb->aux_h) = NULL;
   return 0;
}

/**
 * @brief 利用PARPACK求解特征问题，嵌入PETSC进行并行的线性求解。
 * 不太成熟，首先，默认matA matB 是全局已知的，evec_r eval_r也是全局已知，
 * 所以这个程序只适合多重校正的小规模矩阵。
 * 也就是说，虽然每个进程在结束之后都是分布式存储的，但会通过Allreduce进行整合。
 * 
 * 调用这个程序之前要对PETSC进行初始化，结束之后要Finalize
 *
 * matA (evec_r+i*evec_i) = (eval_r+i*eval_i) matB (evec_r+i*evec_i)
 *
 * @param eigen_solver_id   SHIFT_INVERT or REGULAR_INVERT  0 or 1
 * @param matA   
 * @param matB
 * @param nevals     几个特征值
 * @param tau       the shift
 * @param is_sym    是否是对称特征值问题
 * @param ep_which  哪种类型的特征值
 * @param a         EPS_ALL时的区间左端点      
 * @param b         EPS_ALL时的区间右端点      
 * @param tol       误差限
 * @param maxits    最大迭代数
 * @param eval_r    特征值实部
 * @param eval_i    特征值虚部
 * @param evec_r    特征向量实部
 * @param evec_i    特征向量虚部
 *
 * @return 
 */
   int
HypreSolveEigenProblemPARPACK ( MPI_Comm MPI_COMM_WORLD, int eigen_solver_id, PASE_ParCSRMatrix A_Hh, PASE_ParCSRMatrix B_Hh, 
      int nevals, double tau, bool is_sym, 
      int ep_which, double a, double b, double tol, int maxits,
      double *eval, PASE_ParVector *evec)
{
   printf ( "HypreSolveEigenProblemPARPACK\n" );
   /* 还没有写不是对称的情况 */
   if (is_sym!=true) {
      printf ( "This version of FEMabc can not compute non-hermite eigenvalue problem using ARPACK.\n" );
      exit(0);
   }
   const char *bmat, *which, *howmny;
   int i, k, it, ido, n, nev, ncv, info, N;
   int ldv, iparam[11], ipntr[11], lworkl;
   double sigma, res_max, *resid, *v, *workd, *workl;
   LOGICAL rvec, *select; bool flag;
   HYPRE_Solver solver;

   /* Fortran 中这些参数数组是从1开始的 */
#define IPARAM(i)	iparam[(i) - 1]
#define IPNTR(i)	ipntr[(i) - 1]
#define WORKD(i)	workd[(i) - 1]
#define WORKL(i)	workl[(i) - 1]

   ido = 0; /* 首次调用 */
   bmat = (B_Hh == NULL ? "I" : "G");/* 是否是广义特征值问题 */

   /** 为n赋值，每个进程不同，且与PETSC中一致 */
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   /* TODO: 直接使用evec中的partitioning，不用再分配空间 */
   /*
      HYPRE_Int *partitioning;
      partitioning = malloc(sizeof(HYPRE_Int)*(nprocs+1));

      partitioning = hypre_ParCSRMatrixRowStarts(A_Hh->A_H);
      n = partitioning[myid+1] - partitioning[myid];
      */
   /* TODO: 自己修改, 需要检查, 在最后一个进程的n需要特殊设置 */
   if (myid == nprocs - 1){
      n = hypre_ParVectorPartitioning(evec[0]->b_H)[myid+1]-hypre_ParVectorPartitioning(evec[0]->b_H)[myid] + evec[0]->block_size;
   }
   else{
      n = hypre_ParVectorPartitioning(evec[0]->b_H)[myid+1]-hypre_ParVectorPartitioning(evec[0]->b_H)[myid];
   }
   N = A_Hh->N_H + A_Hh->block_size; 

   printf ( "%d, parpack n = %d, N = %d\n", myid, n, N );

   nev = nevals;/* 要求几个特征值 */
   ncv = 2*nev>N?N:2*nev;/* 矩阵v的列数，几个Lanczos基向量 */
   info = 0; /* 随机的初始残量向量 */
   ldv = n; /* leading dimension of v */

   howmny = "All";/* 计算所有Ritz向量，nev个 */
   rvec = F77_True;/* 计算Ritz值和Ritz向量 */
   sigma = tau;/* the shift */

   if (nev <= 0) {
      /** 返回0个收敛的特征值 */
      printf ( "No eigenvalue is computed\n" );
      return 0;
   }

   lworkl = ncv * (ncv + 8);/* workl的长度，就要这个值，ncv是Lanczos基向量的个数 */

   resid = calloc(n, sizeof(double));/* 残量，info==0时用随机为初始，n是矩阵大小 */
   v = malloc(n*ncv*sizeof(double));/* ncv个Lanczos基向量 */
   workd = malloc(3*n*sizeof(double)); /* 3个工作向量，用于Arnodi迭代 */
   workl = malloc(lworkl* sizeof(double));/* lworkl个工作向量 */

   select = malloc(ncv*sizeof(LOGICAL));/* howmny相关的指示，当howmny不是"ALL"时指示谁要被计算 */

   if ( (eigen_solver_id!=0) && (eigen_solver_id!=1) ) {
      printf ( "Choose Default eigen_solver_id.\n" );
      eigen_solver_id = mode_index;
   }
   mode = mode_list[eigen_solver_id];
   printf ( "Chosen mode is %s.\n", mode_name[eigen_solver_id] );
   /* VectorCreate 但不分配空间，只是一个空壳, 小向量，非全局 */
   /* TODO: 初始创建是否分配空间？*/
   /* TODO: 改成用evec创建？ */
   /*
      PASE_ParVectorCreate(MPI_COMM_WORLD, A_Hh->N_H, A_Hh->block_size, NULL, partitioning, &va);
      PASE_ParVectorCreate(MPI_COMM_WORLD, A_Hh->N_H, A_Hh->block_size, NULL, partitioning, &vb);
      */
   PASE_ParVectorCreate(MPI_COMM_WORLD, evec[0]->N_H, evec[0]->block_size, NULL, hypre_ParVectorPartitioning(evec[0]->b_H), &va);
   PASE_ParVectorCreate(MPI_COMM_WORLD, evec[0]->N_H, evec[0]->block_size, NULL, hypre_ParVectorPartitioning(evec[0]->b_H), &vb);

   /* set up initial residual vector */
   IPARAM(1) = 1;		    /* use exact ishfts */
   IPARAM(3) = maxits * nev;	    /* maxits */
   /*   ssaupd.f
    *   Mode 1:  A*x = lambda*x, A symmetric 
    * 	===> OP = A  and  B = I.
    *   Mode 2:  A*x = lambda*M*x, A symmetric, M symmetric positive definite
    * 	===> OP = inv[M]*A  and  B = M.
    * 	===> (If M can be factored see remark 3 below)
    *   Mode 3:  K*x = lambda*M*x, K symmetric positive semi-definite
    *   ===> OP = (inv[K - sigma*M])*M  and  B = M.                                                       
    *   ===> Shift-and-Invert mode
    */
   /** 不理解SHIFT_INVERT和REGULAR_INVERT
    *  前者和后者关于which是不同的设置，不知道为啥这样弄 */
   /** SHIFT_INVERT:   (A-sigma*B)*x = (lambda-sigma)*B*x，再将左边矩阵求逆 */
   /** REGULAR_INVERT: A*x = B*x，再将右边矩阵求逆 */
   if (mode == SHIFT_INVERT) {
      //	printf ( "mode == SHIFT_INVERT\n" );
      which = (ep_which > 1 ? "LM" : "SM");
      /* shift-invert mode: IPARAM(7) = 3 */
      IPARAM(7) = 3;
      /* 最好能保证，matA和matB的稀疏结构一致 */
      /* OP = (inv[A - sigma*B])*B ，将inv[A-sigma*B]存储在matA */
      //	if (sigma != 0.0)  {
      //	    printf ( "sigma != 0.0\n" );
      /** 稍后要将matA还原 */

      /* TODO: 需要PASE矩阵加减操作函数 */
      /* A = A + tau B */
      PASE_ParCSRMatrixShift(A_Hh, -1.0*sigma, B_Hh, A_Hh);
      //	}
      /* 要求解线性方程组，矩阵是matA，右端项形式上是va，在mat_vec中会具体给出 */
      PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* set some parameters (see reference manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* now setup and solve! */
      hypre_PCGSetup(solver, A_Hh, vb, va);
   }
   else {
      which = (ep_which > 1 ? "SM" : "LM");
      printf ( "mode == REGULAR_INVERT\n" );
      /* regular-invert mode: IPARAM(7) = 1 or 2 */
      if (B_Hh == NULL) {
	 //	    printf ( "matB == NULL\n" );
	 IPARAM(7) = 1;  /* regular mode (standard problem) */
      }
      else {
	 //	    printf ( "matB != NULL\n" );
	 IPARAM(7) = 2;  /* regular invert mode (generalized problem) */
	 /* OP = inv[B]*A */
	 PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

	 /* set some parameters (see reference manual for more parameters) */
	 HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
	 HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
	 HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
	 HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
	 HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

	 /* now setup and solve! */
	 hypre_PCGSetup(solver, B_Hh, vb, va);

      }
   }
   printf ( "bmat = %s, n = %d, which = %s, nev = %d\n", bmat, n, which, nev );

   /* repeatedly call ARPACK's dsaupd subroutine */
   printf("PARPACK: starting Arnoldi update iterations.\n");
   flag = true;
   for (it = 1; flag; ++it) {
      F77name(pdsaupd)(&MPI_COMM_WORLD, &ido, bmat, &n, which, &nev,
	    &tol, resid, &ncv, v, &ldv, iparam,
	    ipntr, workd, workl, &lworkl, &info);
      //      printf ( "%d, ido = %d\n", myid, ido );
      //      sleep(.01);
      /*
       *           IDO =  0: first call to the reverse communication interface
       *           IDO = -1: compute  Y = OP * X  where
       *                     IPNTR(1) is the pointer into WORKD for X,
       *                     IPNTR(2) is the pointer into WORKD for Y.
       *                     This is for the initialization phase to force the
       *                     starting vector into the range of OP.
       *           IDO =  1: compute  Y = OP * X where
       *                     IPNTR(1) is the pointer into WORKD for X,
       *                     IPNTR(2) is the pointer into WORKD for Y.
       *                     In mode 3,4 and 5, the vector B * X is already
       *                     available in WORKD(ipntr(3)).  It does not
       *                     need to be recomputed in forming OP * X.
       *           IDO =  2: compute  Y = B * X  where
       *                     IPNTR(1) is the pointer into WORKD for X,
       *                     IPNTR(2) is the pointer into WORKD for Y.
       *           IDO =  3: compute the IPARAM(8) shifts where
       *                     IPNTR(11) is the pointer into WORKL for
       *                     placing the shifts. See remark 6 below.
       *           IDO = 99: done
       */
      switch (ido) {
	 case -1: 
	    /* Perform y <-- OP(x), where OP(x) is
	       inv[A-sigma*M]*M*x for shift-invert mode, or
	       inv[M]*A*x for regular-invert mode,
	       with x = WORKD(IPNTR(1)), y = WORKD(IPNTR(2))
	       */
	    if (mode == SHIFT_INVERT) {
	       //		    printf ( "mode == SHIFT_INVERT\n" );
	       /* solver中的矩阵是[A-sigma*B]，存储在matA，右端项B*WORKD(IPNTR(1)) */
	       //	       printf ( "myid = %d, WORKD(IPNTR(1)\n", myid );
	       //	       double *tmp = &WORKD(IPNTR(1));
	       //	       VectorPrint(tmp, n);
	       pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), A_Hh, &solver, B_Hh, &WORKD(IPNTR(1)));
	       //	       sleep(1);
	    }
	    else {
	       //		    printf ( "mode == REGULAR_INVERT\n" );
	       /* solver中的矩阵是B，存储在matB，右端项A*WORKD(IPNTR(1)) */
	       if (B_Hh==NULL)
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), NULL, NULL, A_Hh, &WORKD(IPNTR(1)));
	       else {
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), NULL, NULL, A_Hh, &WORKD(IPNTR(1)));
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(1)), NULL, NULL, NULL, &WORKD(IPNTR(2)));
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), B_Hh, &solver, NULL, &WORKD(IPNTR(1)));
	       }
	    }
	    break;
	 case 1:
	    /* Perform y <-- OP(x), where OP(x) is
	       inv[A-sigma*M]*M*x for shift-invert mode, or
	       inv[M]*A*x for regular-invert mode,
	       with x = WORKD(IPNTR(1)), y = WORKD(IPNTR(2))
	       Note   M*x is already stored at WORKD(IPNTR(3))
	       and need not be recomputed for shift-invert mode
	       */
	    if (mode == SHIFT_INVERT) {
	       //		    printf ( "mode == SHIFT_INVERT\n" );
	       /* solver中的矩阵是[A-sigma*B]，右端项B*WORKD(IPNTR(1))=WORKD(IPNTR(3)) */
	       pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), A_Hh, &solver, NULL, &WORKD(IPNTR(3)));
	    }
	    else {
	       //		    printf ( "mode == REGULAR_INVERT\n" );
	       /* solver中的矩阵是B，右端项A*WORKD(IPNTR(1)) */
	       if (B_Hh==NULL)
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), NULL, NULL, A_Hh, &WORKD(IPNTR(1)));
	       else
	       {
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), NULL, NULL, A_Hh, &WORKD(IPNTR(1)));
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(1)), NULL, NULL, NULL, &WORKD(IPNTR(2)));
		  pase_mat_vec (MPI_COMM_WORLD, &WORKD(IPNTR(2)), B_Hh, &solver, NULL, &WORKD(IPNTR(1)));
	       }
	    }
	    //	    printf ( "WORKD(IPNTR(1)) = %f, %f, %f, %f\n", 
	    //		  WORKD(IPNTR(1)), WORKD(IPNTR(1)+1), WORKD(IPNTR(1)+2), WORKD(IPNTR(1)+3) );
	    //	    printf ( "WORKD(IPNTR(2)) = %f, %f, %f, %f\n", 
	    //		  WORKD(IPNTR(2)), WORKD(IPNTR(2)+1), WORKD(IPNTR(2)+2), WORKD(IPNTR(2)+3) );
	    break;
	 case 2:
	    /* Perform y <-- M*x with
	       x = WORKD(IPNTR(1)), y = WORKD(IPNTR(2)) */
	    /* WORKD(IPNTR(2)) = B*WORKD(IPNTR(1)) */
	    pase_mat_vec(MPI_COMM_WORLD, &WORKD(IPNTR(2)), NULL, NULL, B_Hh, &WORKD(IPNTR(1)));
	    //	    printf ( "WORKD(IPNTR(1)) = %f, %f, %f, %f\n", 
	    //		  WORKD(IPNTR(1)), WORKD(IPNTR(1)+1), WORKD(IPNTR(1)+2), WORKD(IPNTR(1)+3) );
	    //	    printf ( "WORKD(IPNTR(2)) = %f, %f, %f, %f\n", 
	    //		  WORKD(IPNTR(2)), WORKD(IPNTR(2)+1), WORKD(IPNTR(2)+2), WORKD(IPNTR(2)+3) );
	    break;
	 case 3:
	    printf( "IDO == 3, unexpected.\n" );
	    break;
	 default:
	    /* Terminate the loop */
	    flag = false;
      }
   }

   if (B_Hh!=NULL||mode==SHIFT_INVERT)
      PASE_ParCSRPCGDestroy(solver);

   if (mode == SHIFT_INVERT) {
      /* TODO: 需要PASE矩阵加减操作 */
      /** 将matA还原 */
      PASE_ParCSRMatrixShift(A_Hh, sigma, B_Hh, A_Hh);
   }

   if (info < 0) printf( "error in DSAUPD, info = %d\n", info);
   /* 调用ARPACK，dseupd，返回特征值eval_r，特征向量v */
   F77name(pdseupd)(&MPI_COMM_WORLD, &rvec, howmny, select, eval, v, &ldv, &sigma,
	 bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
	 iparam, ipntr, workd, workl, &lworkl, &info);
   if (info != 0) printf( "error in DSEUPD, info = %d\n", info);

   nev = IPARAM(5);		    /* number of converged eigenvalues */
   printf ( "number of converged eigenvalues = %d\n", nev );
   /** Note   v = malloc(n*ncv*sizeof(double))  */
   /* TODO: 得到PASE形式的特征向量，已经修改，需要检查 */
   if (myid == nprocs-1){
      for (k = 0; k < nev; ++k) {
	 for (i = 0; i < n-evec[k]->block_size; ++i) {
	    hypre_VectorData(hypre_ParVectorLocalVector(evec[k]->b_H))[i] = *(v+k*n+i);
	 }
      }

      for (k = 0; k < nev; ++k) {
	 for (i = 0; i < evec[k]->block_size; ++i) {
	    hypre_ParVectorLocalVector(evec[k]->aux_h)[i] = *(v+(k+1)*n-evec[k]->block_size+i);
	 }
      }
   }
   else{
      for (k = 0; k < nev; ++k) {
	 for (i = 0; i < n; ++i) {
	    hypre_VectorData(hypre_ParVectorLocalVector(evec[k]->b_H))[i] = *(v+k*n+i);
	 }  
      }
   }
   for (k = 0; k < nev; ++k){
      MPI_Bcast(hypre_VectorData(evec[k]->aux_h), evec[k]->block_size, HYPRE_Real, nprocs-1, MPI_COMM_WORLD);
   }

   //    nit = IPARAM(3);	    /* number of Arnoldi update iterations taken */
   printf ( "number of Arnoldi update iterations = %d\n", IPARAM(3) );

   /* 计算残量 */
   /*    IPNTR(8):  pointer to the NCV RITZ values of the original system.
    *    IPNTR(9):  pointer to the NCV corresponding error bounds.
    *    IPNTR(10): pointer to the NCV by NCV matrix of eigenvectors
    *          of the tridiagonal matrix T. Only referenced by sseupd if RVEC = .TRUE. */
   /* Note: IPNTR(9) points to the NCV error bounds (in workl) */
   for (i = 0, res_max = 0.; i < nev; i++)
   {
      tol = fabs(WORKL(IPNTR(8) + i));
      if (tol == 0.0) tol = WORKL(IPNTR(9) + i);
      else tol = WORKL(IPNTR(9) + i) / tol;
      //	printf( "    residual[%d] = %lg\n", i, (double)tol );
      if (res_max < tol) res_max = tol;
   }
   printf("PARPACK: %d iterations, unscaled residual = %0.4le\n", it, (double)res_max);


   free(resid); free(v); free(workd); free(workl); free(select);
   /* TODO: 先判断再释放，是不是这么个意思？*/
   if (va != NULL){
      PASE_ParVectorDestroy(va);}
   if (vb != NULL){
      PASE_ParVectorDestroy(vb);}
   /** 返回收敛特征值的个数 */
   printf ( "SolveEigenProblemPARPACK\n" );
   return nev;
}

