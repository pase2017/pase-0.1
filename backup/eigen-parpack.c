/**
 *       @file  eigen-arpack.c
 *      @brief  用ARPACK求解特征值问题
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

#if USE_PARPACK && USE_PETSC
//#define F77name(func) func ## _

static int *begin_rows;
static int *rows;
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
static VECTOR va, vb;
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
   static int 
mat_vec (double *py, SOLVER *solver, MATRIX *matA, double *px) 
{
   int *icn, len, i, k;
   double *pa;
   /** 利用va,vb将x,y封装成VECTOR */
   vb.values = py;
   if (matA!=NULL) 
   {
      for (i = 0; i < nprocs; ++i)
      {
	 rows[i] = begin_rows[i+1]-begin_rows[i];
      }
//      printf ( "rows\n" );
//      VectorPrint(rows, nprocs);
//      sleep(.1);
//      printf ( "begin_rows\n" );
//      VectorPrint(begin_rows, nprocs+1);
//      sleep(.1);
//      printf ( "myid = %d, px is\n", myid );
//      VectorPrint(px, rows[myid]);
//      sleep(1);
      MPI_Allgatherv( (void *)px, rows[myid], MPI_DOUBLE, 
	    (void *)va.values, rows, begin_rows, MPI_DOUBLE, MPI_COMM_WORLD);
//      printf ( "myid = %d, va is\n", myid );
//      VectorPrint(va.values, va.nrows);
//      sleep(1);

      /* 矩阵向量运算，vb = (1.0*matA)*va */
      /** 分布式的矩阵向量乘 */
      for (i = 0; i < rows[myid]; ++i) *(py++) = 0;

      px = va.values;
      py = vb.values;
      pa = &(matA->values[matA->begin_idxes[begin_rows[myid]]]);
      icn = &( matA->col_nums[matA->begin_idxes[begin_rows[myid]]]);
      for (i = begin_rows[myid]; i < begin_rows[myid+1]; ++i) {
	 len = matA->begin_idxes[i+1] - matA->begin_idxes[i];
	 for (k = 0; k < len; ++k) {
	    *py += *(pa++) * px[*(icn++)];
	 }
	 ++py;
      }
   }
//   MatrixMultiVector(&vb, 1.0, MAT_OP_N, matA, &va);

   /** 当matA!=NULL时，va是px的全局，vb是matA*va的局部 */
   /** 当matA==NULL时，vb是py 局部 */
   /* If solver != NULL solve with RHS = A*x and set x <-- A*x */
   if (solver != NULL) {
      /* copy vb (A*x) to va, va = vb */
      /** 分布式的矩阵向量乘 */
      if (matA!=NULL) {
	 for (i = 0; i < rows[myid]; ++i) {
	    va.values[i] = vb.values[i];
	 }
      }
      else {
	 for (i = 0; i < rows[myid]; ++i) {
	    va.values[i] = px[i];
	 }
      }
//      MatrixMultiVector(&va, 1.0, MAT_OP_N, NULL, &vb);
      /* 求解器，右端项，未知量 */
//      VectorView(&va);
//      VectorView(&vb);
//      MatrixView(solver->matrix);
      /** va vb 需要分布式存储 */
      SolverSolve(solver, &va, &vb, false, false);
      //	printf( "linear solver: nits = %d, res = %lg\n", solver->nits, solver->residual);
   }

   if (matA==NULL&&solver==NULL)
   {
      for (i = 0; i < rows[myid]; ++i) {
	 py[i] = px[i];
      }
   }
//   va.values = NULL;
   vb.values = NULL;

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
SolveEigenProblemPARPACK ( MPI_Comm comm, int eigen_solver_id, MATRIX *matA, MATRIX *matB, 
      int nevals, double tau, bool is_sym, 
      int ep_which, double a, double b, double tol, int maxits,
      double *eval_r, double *eval_i, VECTOR *evec_r, VECTOR *evec_i)
{
   printf ( "SolveEigenProblemPARPACK\n" );
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
   SOLVER solver;
   /* Fortran 中这些参数数组是从1开始的 */
#define IPARAM(i)	iparam[(i) - 1]
#define IPNTR(i)	ipntr[(i) - 1]
#define WORKD(i)	workd[(i) - 1]
#define WORKL(i)	workl[(i) - 1]

   ido = 0; /* 首次调用 */
   bmat = (matB == NULL ? "I" : "G");/* 是否是广义特征值问题 */
   N = matA->nrows;/* 特征值问题的规模 */

   /** 为n赋值，每个进程不同，且与PETSC中一致 */
   MPI_Comm_rank(comm,&myid);
   MPI_Comm_size(comm,&nprocs);
   begin_rows = malloc(sizeof(int)*(nprocs+1));
   rows = malloc(sizeof(int)*N);
   /** 得到每个进程拥有的行数，通过平均非零元 */
//   begin_rows[0] = 0; 
//   for ( n = 1; n < nprocs; ++n)
//   {
//      begin_rows[n] = begin_rows[n-1] + N/nprocs;
//   }
//   begin_rows[nprocs] = N;
   partition(matA->begin_idxes, N, begin_rows, nprocs);
   /** 对n进行赋值，对非零元素个数进行赋值 */
   n = begin_rows[myid+1] - begin_rows[myid];
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
   VectorCreate(&va, N);
   VectorCreate(&vb, n);
   VectorAllocateValues(&va);
   //    va = malloc(1*sizeof(VECTOR)); va->nrows = n; va->block = false; va->values = NULL;
   //    vb = malloc(1*sizeof(VECTOR)); vb->nrows = n; vb->block = false; vb->values = NULL;

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
      MatrixAXPBY(matA, 1.0, matA, -sigma, matB);
      //	}
      /* 要求解线性方程组，矩阵是matA，右端项形式上是va，在mat_vec中会具体给出 */
      /** PETSC CG+JACOBI */
      SolverCreate(&solver, SOLVER_PETSC, 2, 1, matA);
   }
   else {
      which = (ep_which > 1 ? "SM" : "LM");
      printf ( "mode == REGULAR_INVERT\n" );
      /* regular-invert mode: IPARAM(7) = 1 or 2 */
      if (matB == NULL) {
	 //	    printf ( "matB == NULL\n" );
	 IPARAM(7) = 1;  /* regular mode (standard problem) */
      }
      else {
	 //	    printf ( "matB != NULL\n" );
	 IPARAM(7) = 2;  /* regular invert mode (generalized problem) */
	 /* OP = inv[B]*A */
	 SolverCreate(&solver, SOLVER_PETSC, 2, 1, matB);
      }
   }
   printf ( "bmat = %s, n = %d, which = %s, nev = %d\n", bmat, n, which, nev );

   /* repeatedly call ARPACK's dsaupd subroutine */
   printf("PARPACK: starting Arnoldi update iterations.\n");
   flag = true;
   for (it = 1; flag; ++it) {
      F77name(pdsaupd)(&comm, &ido, bmat, &n, which, &nev,
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
	       mat_vec (&WORKD(IPNTR(2)), &solver, matB, &WORKD(IPNTR(1)));
//	       sleep(1);
	    }
	    else {
	       //		    printf ( "mode == REGULAR_INVERT\n" );
	       /* solver中的矩阵是B，存储在matB，右端项A*WORKD(IPNTR(1)) */
	       if (matB==NULL)
		  mat_vec (&WORKD(IPNTR(2)), NULL, matA, &WORKD(IPNTR(1)));
	       else {
		  mat_vec (&WORKD(IPNTR(2)), NULL, matA, &WORKD(IPNTR(1)));
		  mat_vec (&WORKD(IPNTR(1)), NULL, NULL, &WORKD(IPNTR(2)));
		  mat_vec (&WORKD(IPNTR(2)), &solver, NULL, &WORKD(IPNTR(1)));
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
	       mat_vec (&WORKD(IPNTR(2)), &solver, NULL, &WORKD(IPNTR(3)));
	    }
	    else {
	       //		    printf ( "mode == REGULAR_INVERT\n" );
	       /* solver中的矩阵是B，右端项A*WORKD(IPNTR(1)) */
	       if (matB==NULL)
		  mat_vec (&WORKD(IPNTR(2)), NULL, matA, &WORKD(IPNTR(1)));
	       else
	       {
		  mat_vec (&WORKD(IPNTR(2)), NULL, matA, &WORKD(IPNTR(1)));
		  mat_vec (&WORKD(IPNTR(1)), NULL, NULL, &WORKD(IPNTR(2)));
		  mat_vec (&WORKD(IPNTR(2)), &solver, NULL, &WORKD(IPNTR(1)));
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
	    mat_vec(&WORKD(IPNTR(2)), NULL, matB, &WORKD(IPNTR(1)));
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

   if (matB!=NULL||mode==SHIFT_INVERT)
      SolverDestroy(&solver);

   if (mode == SHIFT_INVERT) {
      /** 将matA还原 */
      MatrixAXPBY(matA, 1.0, matA, sigma, matB);
   }

   if (info < 0) printf( "error in DSAUPD, info = %d\n", info);
   /* 调用ARPACK，dseupd，返回特征值eval_r，特征向量v */
   F77name(pdseupd)(&comm, &rvec, howmny, select, eval_r, v, &ldv, &sigma,
	 bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
	 iparam, ipntr, workd, workl, &lworkl, &info);
   if (info != 0) printf( "error in DSEUPD, info = %d\n", info);

//   VectorPrint(eval_r, nev);
   nev = IPARAM(5);		    /* number of converged eigenvalues */
   printf ( "number of converged eigenvalues = %d\n", nev );
   /** Note   v = malloc(n*ncv*sizeof(double))  */
   /** v是分布式存储的，需要Allgather进行整合 */
   if (eval_i!=NULL) {
      for (i = 0; i < nprocs; ++i)
      {
	 rows[i] = begin_rows[i+1]-begin_rows[i];
      }

      for (k = 0; k < nev; ++k) {
	 MPI_Allgatherv( (void *)(v+k*n), rows[myid], MPI_DOUBLE, 
	       (void *)evec_r[k].values, rows, begin_rows, MPI_DOUBLE, MPI_COMM_WORLD);

	 eval_i[k] = 0;
	 for (i = 0; i < N; ++i) {
	    evec_i[k].values[i] = 0;
	 }
      }
   }
   else {
      for (i = 0; i < nprocs; ++i)
      {
	 rows[i] = begin_rows[i+1]-begin_rows[i];
      }
      for (k = 0; k < nev; ++k) {
	 MPI_Allgatherv( (void *)(v+k*n), rows[myid], MPI_DOUBLE, 
	       (void *)evec_r[k].values, rows, begin_rows, MPI_DOUBLE, MPI_COMM_WORLD);
      }
//      for (k = 0; k < nev; ++k) {
//	 for (i = 0; i < n; ++i) {
//	    evec_r[k].values[i] = *(v+k*n+i);
//	 }
//      }
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

//   sleep(1);
   VectorDestroy(&va);
//   VectorDestroy(&vb);

   free(resid); free(v); free(workd); free(workl); free(select);
   free(begin_rows); free(rows);
   /** 返回收敛特征值的个数 */
   printf ( "SolveEigenProblemPARPACK\n" );
   return nev;
}

#endif
