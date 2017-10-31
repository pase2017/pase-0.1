/*
 * =====================================================================================
 *
 *       Filename:  pase_mg.c
 *
 *    Description:  PASE_ParCSRMatrix下用MG方法求解特征值问题
 *
 *        Version:  1.0
 *        Created:  2017年09月11日
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  HONGQICHEN 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_mg.h"
#include "pase_hypre.h"
#include "pase_pcg.h"

/**
 * @brief 利用PASE_ParCSRMatrix的矩阵结构
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 * 就是对solve进行赋初值, PASE_Solver这个结构是空指针, 此函数指向pase_MGData
 * 主要基于PASE_ParCSRMatrixMatvec_HYPRE_ParVector进行矩阵乘向量的运算
 *
 */

/*
 * MGFunctions的创建 
 */
pase_MGFunctions *pase_MGFunctionsCreate( 
   char *       (*CAlloc)        ( size_t count, size_t elt_size ),
   PASE_Int     (*Free)          ( char *ptr ),
   PASE_Int     (*CommInfo)      ( void  *A, PASE_Int   *my_id,
                                   PASE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   PASE_Int     (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   PASE_Int     (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x, HYPRE_Complex beta, void *y ),
   PASE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   PASE_Int    (*CopyVector)    ( void *x, void *y ),
   PASE_Int    (*ClearVector)   ( void *x ),
   PASE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   PASE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   PASE_Int    (*DirSolver)    ( PASE_Solver solver ),
   PASE_Int    (*PreSmooth)     ( PASE_Solver solver ),
   PASE_Int    (*PostSmooth)    ( PASE_Solver solver ),
   PASE_Int    (*PreSmoothInAux)( PASE_Solver solver ),
   PASE_Int    (*PostSmoothInAux)( PASE_Solver solver )
    ) 
{
    pase_MGFunctions *mg_functions;
    mg_functions = (pase_MGFunctions*)CAlloc( 1, sizeof(pase_MGFunctions));

    mg_functions->CAlloc 	= CAlloc;
    mg_functions->Free   	= Free;
    mg_functions->CommInfo 	= CommInfo;
    mg_functions->CreateVector 	= CreateVector;
    mg_functions->DestroyVector = DestroyVector;
    mg_functions->MatvecCreate 	= MatvecCreate;
    mg_functions->Matvec 	= Matvec;
    mg_functions->MatvecDestroy = MatvecDestroy;
    mg_functions->InnerProd 	= InnerProd;
    mg_functions->CopyVector 	= CopyVector;
    mg_functions->ClearVector 	= ClearVector;
    mg_functions->ScaleVector 	= ScaleVector;
    mg_functions->Axpy 		= Axpy;

    mg_functions->DirSolver 	= DirSolver;
    mg_functions->PreSmooth 	= PreSmooth;
    mg_functions->PostSmooth 	= PostSmooth;
    mg_functions->PreSmoothInAux = PreSmoothInAux;
    mg_functions->PostSmoothInAux = PostSmoothInAux;

   return mg_functions;
}

/*
 * MGData的创建，与默认参数的赋值
 */
void *pase_MGCreate( pase_MGFunctions *mg_functions)
{
    pase_MGData *mg_data;

    mg_data = hypre_CTAllocF(pase_MGData, 1, mg_functions);

    mg_data->functions = mg_functions;

    /* set defaults */
    mg_data->block_size = 1;
    mg_data->pre_iter 	= 1;
    mg_data->post_iter 	= 1;
    mg_data->max_iter  	= 1;
    mg_data->max_level 	= -1;
    mg_data->cur_level 	= -1;
    mg_data->rtol	= 1e-8;
    mg_data->atol	= 1e-5;
    mg_data->r_norm	= 0;
    mg_data->A 		= NULL;
    mg_data->M 		= NULL;
    mg_data->P 		= NULL;
    mg_data->Ap 	= NULL;
    mg_data->Mp 	= NULL;
    mg_data->u 		= NULL;
    mg_data->eigenvalues= NULL;

    mg_data->exact_eigenvalues	= NULL;
    mg_data->num_converged	= 0;
    mg_data->num_iter		= 0;
    mg_data->print_level	= 1;

    return (void*)mg_data;
}

/*
 * MGData的创建，调用pase_MGFunctionsCreate函数，对MGFunction赋予默认的函数指针集 
 */
PASE_Int PASE_ParCSRMGCreate( PASE_Solver* solver)
{
   if( !solver)
   {
       hypre_error_in_arg( 2);
       return hypre_error_flag;
   }
   pase_MGFunctions *mg_functions;
   mg_functions = pase_MGFunctionsCreate(
	   hypre_CAlloc,
	   hypre_ParKrylovFree,
	   pase_ParKrylovCommInfo,
	   pase_ParKrylovCreateVector, 
	   pase_ParKrylovDestroyVector,
	   hypre_ParKrylovMatvecCreate,
	   pase_ParKrylovMatvec, 
	   hypre_ParKrylovMatvecDestroy,
	   pase_ParKrylovInnerProd, 
	   pase_ParKrylovCopyVector,
	   pase_ParKrylovClearVector,
	   pase_ParKrylovScaleVector, 
	   pase_ParKrylovAxpy,
	   pase_ParCSRMGDirSolver,
	   //pase_ParCSRMGSmootherCG,
	   //pase_ParCSRMGSmootherCG,
	   //pase_ParCSRMGInAuxSmootherCG,
	   //pase_ParCSRMGInAuxSmootherCG);
	   PASE_Cg,
	   PASE_Cg,
	   PASE_Cg,
	   PASE_Cg);
   *solver = ((HYPRE_Solver)pase_MGCreate( mg_functions));

   return hypre_error_flag;
}

/*
 * 加入新的网格，可以是一层层加，也可以多层一起加。
 * flag表示输入的网格是编号模式，1表示从细（0）到粗（n），0表示从粗（0）到细（n）
 */
PASE_Int PASE_MGAddLevel( PASE_Solver solver, 
	                 HYPRE_ParCSRMatrix* A, 
			 HYPRE_ParCSRMatrix* M, 
			 HYPRE_ParCSRMatrix* P,
			 PASE_Int n,
			 PASE_Int flag)
{
    PASE_Int i 		= 0;
    pase_MGData* data 	= (pase_MGData*) solver;
    PASE_Int max_level 	= data->max_level;
    //pase_MGFunctions* functions = data->functions;

    printf("Adding %d levels.........\n", n);
    if( A != NULL)
    {
	HYPRE_ParCSRMatrix* solver_A 	= (HYPRE_ParCSRMatrix*)data->A;
	HYPRE_ParCSRMatrix* new_A 	= hypre_CTAlloc( HYPRE_ParCSRMatrix, max_level+1+n);
	for( i=0; i<max_level+1; i++)
	{	
	    new_A[i] = solver_A[i];
	}	
	if( flag == 1)
	{
	    for( i=0; i<n; i++)
	    {
		new_A[i+max_level+1] = A[n-1-i];
	    }
	}
	else
	{
	    for( i=0; i<n; i++)
	    {
		new_A[i+max_level+1] = A[i];
	    }
	}
	data->A = (void**)new_A;
	if( solver_A != NULL)
	{
	    hypre_TFreeF( solver_A, data->functions);
	}
    }
    else
    {
	printf("Error: A is NULL when adding MG levels!\n");
    }

    if( M != NULL)
    {
	HYPRE_ParCSRMatrix* solver_M 	= (HYPRE_ParCSRMatrix*)data->M;
	HYPRE_ParCSRMatrix* new_M 	= hypre_CTAlloc( HYPRE_ParCSRMatrix, max_level+1+n);
	for( i=0; i<max_level+1; i++)
	{	
	    new_M[i] = solver_M[i];
	}	
	if( flag == 1)
	{
	    for( i=0; i<n; i++)
	    {
		new_M[i+max_level+1] = M[n-1-i];
	    }
	}
	else
	{
	    for( i=0; i<n; i++)
	    {
		new_M[i+max_level+1] = M[i];
	    }
	}
	data->M = (void**)new_M;
	if( solver_M != NULL)
	{
	    hypre_TFreeF( solver_M, data->functions);
	}
    }
     
    if( P != NULL)
    {
	HYPRE_ParCSRMatrix* solver_P 	= (HYPRE_ParCSRMatrix*)data->P;
	HYPRE_ParCSRMatrix* new_P 	= hypre_CTAlloc( HYPRE_ParCSRMatrix, max_level+n);
	if( solver_P != NULL)
	{
	    for( i=0; i<max_level; i++)
	    {	
		new_P[i] = solver_P[i];
	    }	
	    if( flag == 1)
	    {
		for( i=0; i<n-1; i++)
		{
		    new_P[i+max_level] = P[n-2-i];
		}
	    }
	    else
	    {
		for( i=0; i<n-1; i++)
		{
		    new_P[i+max_level] = P[i];
		}
	    }
	    hypre_TFreeF( solver_P, data->functions);
	}
	else
	{
	    //printf("The first time to add projector P\n");
	    if( flag == 1)
	    {
	        for( i=0; i<n-1; i++)
	        {
	            new_P[i] = P[n-2-i];
	        }
	    }
	    else
	    {
	        for( i=0; i<n-1; i++)
	        {
	            new_P[i] = P[i];
	        }
	    }
	}
	data->P = (void**)new_P;
    }
    
    data->max_level += n;
    data->cur_level = data->max_level;
    return 0;
}

/*
 * 生成各个层的初始向量，
 * 对于最细层，可以是给定的初始向量，也可以是随机向量。
 * 对于其他层的pase向量，初始值为除了对应于多出来的维度为1，其他维度均为0，比如(0,...,0, 1)为多出来维数为1的情况。
 */
PASE_Int PASE_ParCSRMGInit( MPI_Comm comm, PASE_Solver solver, HYPRE_ParVector* u_h, PASE_Int block_size, PASE_Int* seed)
{
    PASE_Int i, j;
    PASE_Int N_h, M_h;
    PASE_Int* partitioning;
    pase_MGData* data 		= (pase_MGData*) solver;
    pase_MGFunctions* functions = data->functions;
    PASE_Int max_level 		= data->max_level;
    HYPRE_ParCSRMatrix* A 	= (HYPRE_ParCSRMatrix*) data->A;
    HYPRE_ParCSRMatrix* M 	= (HYPRE_ParCSRMatrix*) data->M;
    HYPRE_Complex** eigenvalues;

#if 1
    eigenvalues = hypre_CTAlloc( HYPRE_Complex*, max_level+1);
    for( i=0; i<max_level+1; i++)
    {
	eigenvalues[i] = hypre_CTAlloc( HYPRE_Complex, block_size);
    }
    for( i=0; i<block_size; i++)
    {
	eigenvalues[max_level][i] = 1.0;
    }
    data->eigenvalues = eigenvalues;
#endif

#if 1
    pase_ParVector*** U 	= hypre_CTAllocF( pase_ParVector**, max_level+1, functions);   
    pase_ParVector** U_h 	= hypre_CTAllocF( pase_ParVector*, block_size, functions);
    if( u_h != NULL)
    {
	PASE_Real inner_A, inner_M;
	HYPRE_ParVector workspace;
	//printf("MGInit: Given init vector!!!\n");
	N_h 		= u_h[0]->global_size;
	partitioning 	= u_h[0]->partitioning;
        workspace 	= hypre_ParVectorCreate( comm, N_h, partitioning); 	
	HYPRE_ParVectorInitialize( workspace);
	hypre_ParVectorSetPartitioningOwner( workspace, 0);
	for( i=0; i<block_size; i++)
	{
	    PASE_ParVectorCreate( comm, N_h, block_size, u_h[i], NULL, U_h+i); 	
	    hypre_ParCSRMatrixMatvec( 1.0, A[max_level], u_h[i], 0.0, workspace);
	    inner_A = hypre_ParVectorInnerProd( workspace, u_h[i]);
	    if( M != NULL)
	    {
		hypre_ParCSRMatrixMatvec( 1.0, M[max_level], u_h[i], 0.0, workspace);
		inner_M = hypre_ParVectorInnerProd( workspace, u_h[i]);
	    }
	    else
	    {
		inner_M = hypre_ParVectorInnerProd( u_h[i], u_h[i]);
	    }
	    eigenvalues[max_level][i] = inner_A/inner_M;
	}
	HYPRE_ParVectorDestroy( workspace);
    }
    else
    {
	HYPRE_ParCSRMatrixGetRowPartitioning( A[max_level], &partitioning);
	HYPRE_ParCSRMatrixGetDims( A[max_level], &M_h, &N_h);
	for( i=0; i<block_size; i++)
	{
	    PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, U_h+i); 	
	    hypre_ParVectorSetRandomValues( U_h[i]->b_H, seed[i]);
	    hypre_SeqVectorSetConstantValues( U_h[i]->aux_h, 0.0);
	    hypre_ParVectorSetPartitioningOwner( U_h[i]->b_H,0);
	}
	hypre_ParVectorSetPartitioningOwner( U_h[0]->b_H,1);
    }
    U[max_level] = U_h;
    for( j=0; j<max_level; j++)
    {
	U_h = hypre_CTAllocF( pase_ParVector*, block_size, functions);
	HYPRE_ParCSRMatrixGetRowPartitioning( A[j], &partitioning);
	//printf("p[0] = %d, p[1] = %d, p[2] = %d\n", partitioning[0], partitioning[1], partitioning[2]);
	HYPRE_ParCSRMatrixGetDims( A[j], &M_h, &N_h);
	for( i=0; i<block_size; i++)
	{
	    PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, U_h+i); 	
	    hypre_ParVectorSetConstantValues( U_h[i]->b_H, 0.0);
	    hypre_SeqVectorSetConstantValues( U_h[i]->aux_h, 0.0);
	    HYPRE_Complex* aux_data 	= U_h[i]->aux_h->data;
	    aux_data[i] 		= 1.0;
	    hypre_ParVectorSetPartitioningOwner( U_h[i]->b_H, 0);
	}
	hypre_ParVectorSetPartitioningOwner( U_h[0]->b_H, 1);
	U[j] = U_h;
    }
    data->u = (void***) U;
#endif

#if 1
    PASE_ParCSRMatrix* Ap = hypre_CTAlloc( PASE_ParCSRMatrix, max_level+1);
    data->Ap = (void**)Ap;
    if( data->M != NULL)
    {
	PASE_ParCSRMatrix* Mp 	= hypre_CTAlloc( PASE_ParCSRMatrix, max_level+1);
	data->Mp 		= (void**)Mp;
    }	
#endif
    data->block_size 		= block_size;

    return 0;
}

/**
 * MG求解 
 */
PASE_Int PASE_ParCSRMGSolve( PASE_Solver solver)
{
   pase_MGData *mg_solver 	= (pase_MGData*)solver;

   do
   {
       mg_solver->num_iter ++;
       pase_ParCSRMGIteration( solver);
       pase_ParCSRMGErrorEstimate( solver);
       pase_ParCSRMGRestart( solver);
   }
   while( mg_solver->max_iter > mg_solver->num_iter && mg_solver->num_converged < mg_solver->block_size);

   return 0;
}

PASE_Int pase_ParCSRMGIteration( PASE_Solver solver)
{
   pase_MGData *data 		= (pase_MGData*)solver;
   pase_MGFunctions *functions 	= data->functions;
   PASE_Int	cur_level 	= data->cur_level;
   PASE_Int	max_level 	= data->max_level;

   if( cur_level>0 && cur_level<=max_level)
   {
       //printf("cur_level = %d, max_level = %d\n", data->cur_level, data->max_level);
       /*前光滑*/
       //printf("PreSmoothing..........\n");
       pase_ParCSRMGPreSmooth( solver);
       PASE_Orth(solver);
       //printf("Creating AuxMatrix..........\n");
       //pase_ParCSRMGAuxMatrixCreate( solver);
       pase_MGAuxMatrixCreate( solver);
       
       /*粗空间校正*/
       //printf("Correction on low-dim space\n");
       data->cur_level--;
       pase_ParCSRMGIteration( solver);
       data->cur_level++;

       /*后光滑*/
       //printf("PostCorrecting..........\n");
       pase_ParCSRMGPostCorrection( solver);
       //printf("PostSmoothing..........\n");
       pase_ParCSRMGPostSmooth( solver);
   }
   else if( cur_level==0 && max_level>0)
   {
       functions->DirSolver( solver);
   }
   else
   {
       printf("Error: cur_level = %d, max_level = %d\n", data->cur_level, data->max_level);
   }
   return 0;
}

/*
 * 前光滑
 */
PASE_Int pase_ParCSRMGPreSmooth( PASE_Solver solver)
{
    pase_MGData *data 		= (pase_MGData*)solver;
    pase_MGFunctions *functions = data->functions;
    PASE_Int cur_level 		= data->cur_level;
    PASE_Int max_level 		= data->max_level;
    if( cur_level == max_level)
    {
	functions->PreSmooth( solver);
    }
    else
    {
	functions->PreSmoothInAux( solver);
    }
    return 0;
}

/*
 * 后光滑
 */
PASE_Int pase_ParCSRMGPostSmooth( PASE_Solver solver)
{
    pase_MGData *data 		= (pase_MGData*)solver;
    pase_MGFunctions *functions = data->functions;
    PASE_Int cur_level 		= data->cur_level;
    PASE_Int max_level 		= data->max_level;
    if( cur_level == max_level)
    {
	functions->PostSmooth( solver);
    }
    else
    {
	functions->PostSmoothInAux( solver);
    }
    return 0;
}

/*
 * 把辅助粗空间的向量投影到(辅助)细空间
 */
PASE_Int pase_ParCSRMGPostCorrection( PASE_Solver solver)
{
    PASE_Int i, j;
    pase_MGData* data          	= (pase_MGData*) solver;
    pase_MGFunctions* functions = data->functions;
    PASE_Int cur_level          = data->cur_level;
    PASE_Int block_size         = data->block_size;
    pase_ParVector*** u         = (pase_ParVector***) data->u;
    pase_ParVector** u0 	= u[cur_level-1];
    pase_ParVector** u1 	= u[cur_level];
    hypre_ParCSRMatrix** P 	= (hypre_ParCSRMatrix**) data->P;
    hypre_ParCSRMatrix* P0 	= P[cur_level-1];

    /* 重新申请一组pase向量 */
    PASE_Int N_h 		= u1[0]->N_H;
    PASE_Int* partitioning 	= u1[0]->b_H->partitioning;
    pase_ParVector** u_new 	= hypre_CTAllocF( pase_ParVector*, block_size, functions);

    /* u_new->b_H += P*u0->b_H */
    /* u_new += u1*u_0->aux_h */
    PASE_Real* aux_data 	= NULL;

#if 0
    /* test */
    HYPRE_ParCSRMatrix* M = (HYPRE_ParCSRMatrix*)data->M;
    pase_ParCSRMatrix** Mp = (pase_ParCSRMatrix**)data->Mp;
    PASE_Real inner_AH, inner_Ah, inner_AHH, inner_Ahh, middle_term_H, middle_term_h, last_term_H, last_term_h, beta;
    PASE_Int N_H = u0[0]->N_H;
    PASE_Int *partitioning_H = u0[0]->b_H->partitioning;
    PASE_Int *partitioning_h = u1[0]->b_H->partitioning;
    HYPRE_ParVector workspace_H = hypre_ParVectorCreate( MPI_COMM_WORLD, N_H, partitioning_H);
    hypre_ParVectorInitialize(workspace_H);
    hypre_ParVectorOwnsPartitioning(workspace_H) = 0;
    HYPRE_ParVector workspace_h = hypre_ParVectorCreate( MPI_COMM_WORLD, N_h, partitioning_h);
    hypre_ParVectorInitialize(workspace_h);
    hypre_ParVectorOwnsPartitioning(workspace_h) = 0;
    HYPRE_ParCSRMatrixMatvec(1.0, M[cur_level-1], u0[0]->b_H, 0.0, workspace_H);
    HYPRE_ParVectorInnerProd( workspace_H, u0[0]->b_H, &inner_AH);
    HYPRE_ParVectorInnerProd( u0[0]->b_H, Mp[cur_level-1]->aux_Hh[0], &middle_term_H); 
    middle_term_H = 2*middle_term_H*u0[0]->aux_h->data[0];
    last_term_H = Mp[cur_level-1]->aux_hh->data[0]*u0[0]->aux_h->data[0]*u0[0]->aux_h->data[0];
    inner_AHH = inner_AH + middle_term_H + last_term_H;
    printf("inner_AH = %.30f, middle_term_H = %.30f, last_term_H = %.30f, inner_A = %.30f, beta = %.30f\n", inner_AH, middle_term_H, last_term_H, inner_AHH, Mp[cur_level-1]->aux_hh->data[0]);
#endif

    for( i=0; i<block_size; i++)
    {
	MPI_Comm comm 		= u1[i]->comm;
	PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, &(u_new[i])); 	
	if( u1[0]->b_H->owns_partitioning == 1)
	{
	    hypre_ParVectorSetPartitioningOwner( u_new[0]->b_H, 1);
	    hypre_ParVectorSetPartitioningOwner( u1[0]->b_H, 0);
	}
	//printf("P: %d rows and %d cols, u_H: %d rows\n", P0->global_num_rows, P0->global_num_cols, N_h);
	hypre_ParCSRMatrixMatvec( 1.0 , P0 , u0[i]->b_H , 0.0 , u_new[i]->b_H );

#if 0

	HYPRE_ParCSRMatrixMatvec(1.0, M[cur_level], u_new[0]->b_H, 0.0, workspace_h);
	HYPRE_ParVectorInnerProd( workspace_h, u_new[0]->b_H, &inner_Ah);
	HYPRE_ParCSRMatrixMatvec(1.0, M[cur_level], u1[0]->b_H, 0.0, workspace_h);
	HYPRE_ParVectorInnerProd( workspace_h, u_new[0]->b_H, &middle_term_h);
	middle_term_h = 2*middle_term_h*u0[0]->aux_h->data[0];
	HYPRE_ParCSRMatrixMatvec(1.0, M[cur_level], u1[0]->b_H, 0.0, workspace_h);
	HYPRE_ParVectorInnerProd( workspace_h, u1[0]->b_H, &beta);
	last_term_h = beta*u0[0]->aux_h->data[0]*u0[0]->aux_h->data[0];
	inner_Ahh = inner_Ah+middle_term_h+last_term_h;
	//PASE_Real t = middle_term_H-middle_term_h;
	//inner_Ahh-=100*t;
	printf("inner_Ah = %.30f, middle_term_h= %.30f, last_term_h = %.30f, inner_A = %.30f, beta = %.30f\n", inner_Ah, middle_term_h, last_term_h, inner_Ahh,beta);
	//printf("D_inner_Ah = %.30f, D_middle_term = %.30f, D_last_term = %.30f, D_inner_A = %.30f\n", inner_AH-inner_Ah, t, last_term_H-last_term_h, inner_AHH-inner_Ahh);
#endif

	aux_data = u0[i]->aux_h->data;
	for( j=0; j<block_size; j++)
	{
	    //printf("aux_h[0] = %.30f\n", aux_data[0]);
	    PASE_ParVectorAxpy( aux_data[j], u1[j], u_new[i]);
	}
    }
    u[cur_level] = u_new;

    /* 释放空间 */
    for( i=0; i<block_size; i++)
    {
	PASE_ParVectorDestroy( u1[i]);
    }
    hypre_TFreeF( u1, functions);

    PASE_Complex* eigenvalues0 = data->eigenvalues[cur_level-1];
    PASE_Complex* eigenvalues1 = data->eigenvalues[cur_level];
    for( i=0; i<block_size; i++)
    {
	eigenvalues1[i] = eigenvalues0[i];
    }
    return 0;
}

/*
 * 构造辅助粗空间
 */
PASE_Int pase_ParCSRMGAuxMatrixCreate( PASE_Solver solver)
{
    PASE_Int i, j, k, l; 
    pase_MGData*        data 	     = (pase_MGData*)solver;
    PASE_Int            cur_level    = data->cur_level;
    PASE_Int            max_level    = data->max_level;
    PASE_Int            block_size   = data->block_size;
    PASE_Int            num_nonzeros = block_size*block_size;
    HYPRE_ParCSRMatrix  *A 	     = (HYPRE_ParCSRMatrix*)data->A;
    HYPRE_ParCSRMatrix  *P 	     = (HYPRE_ParCSRMatrix*)data->P;
    pase_ParCSRMatrix   **Ap 	     = (pase_ParCSRMatrix**)data->Ap;
    pase_ParVector      ***u 	     = (pase_ParVector***)data->u;

    //printf("cur_level = %d, max_level = %d\n", cur_level, max_level);
    HYPRE_ParCSRMatrix  A0 	     = A[cur_level-1];
    HYPRE_ParCSRMatrix  A1 	     = A[cur_level];
    //printf("N_H of A0 = %d, N_H of A1 = %d\n", A0->global_num_rows, A1->global_num_rows);
    HYPRE_ParCSRMatrix  P0 	= P[cur_level-1];
    pase_ParCSRMatrix   *Ap0 	= NULL;
    pase_ParCSRMatrix   *Ap1 	= Ap[cur_level];
    pase_ParVector      **U 	= u[cur_level];
    HYPRE_ParVector     *u_h 	= hypre_CTAlloc(HYPRE_ParVector, block_size); 

    for( i=0; i<block_size; i++)
    {
	u_h[i] = U[i]->b_H;
    }
    pase_ParVector** U_H 	= u[cur_level-1]; 
    PASE_Int* partitioning_H 	= U_H[0]->b_H->partitioning;
    PASE_Int N_H 		= U_H[0]->b_H->global_size;
    HYPRE_ParVector workspace_H = hypre_ParVectorCreate( MPI_COMM_WORLD, N_H, partitioning_H); 
    hypre_ParVectorInitialize( workspace_H);
    hypre_ParVectorSetPartitioningOwner( workspace_H, 0);
    PASE_Int *partitioning_h	= NULL;
    //partitioning_h = u_h[0]->partitioning;
    //N_h = u_h[0]->global_size;
    HYPRE_ParCSRMatrixGetRowPartitioning( A1, &partitioning_h);
    PASE_Int N_h 		= A1->global_num_rows;
    HYPRE_ParVector workspace_h = hypre_ParVectorCreate( MPI_COMM_WORLD, N_h, partitioning_h); 
    hypre_ParVectorInitialize( workspace_h);
    hypre_ParVectorSetPartitioningOwner( workspace_h, 1);
    //printf("N_H = %d, N_h = %d\n", partitioning_H[1], partitioning_h[1]);
    //printf("P0: N_H = %d, N_h = %d\n", P0->global_num_rows, P0->global_num_cols);
	//printf("P: %d rows and %d cols, u_H: %d rows\n", P0->global_num_rows, P0->global_num_cols, N_h);

    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, A0, P0, A1, u_h, &Ap0, workspace_H, workspace_h);  
    Ap[cur_level-1] = (void*)Ap0;
    /*还需修正aux_hH,aux_Hh,aux_hh*/
    if( cur_level < max_level)
    {
	hypre_ParVector** b0 	= Ap0->aux_Hh;
	HYPRE_Real* va_a0 	= Ap0->aux_hh->data;
	hypre_ParVector** b1 	= Ap1->aux_Hh;
	HYPRE_Real* va_a1  	= Ap1->aux_hh->data;
       /* aux_hH[cur_level-1] += P^T*aux_hH[cur_level]*Z */
       for( i=0; i<block_size; i++)
       {
	   for( j=0; j<block_size; j++)
	   {
	       hypre_ParCSRMatrixMatvecT( U[i]->aux_h->data[j], P0, b1[j], 1.0, b0[i]);
	   }
       }
       /* aux_hh[cur_level-1] += 2*U^T*aux_hH[cur_level]*Z + Z^T*aux_hh[cur_level]*Z */
       PASE_Real *uTb = (PASE_Real*)malloc( num_nonzeros*sizeof(PASE_Real));
       for( i=0; i<block_size; i++)
       {
	   for(j=0; j<block_size; j++)
	   {
	       uTb[j+i*block_size] = hypre_ParVectorInnerProd( U[i]->b_H, b1[j]);
	   }
       }
       for( i=0; i<block_size; i++)
       {
	   for( j=0; j<block_size; j++)
	   {
	      for( k=0; k<block_size; k++)
	      {
		  va_a0[i*block_size+j] += 2 * U[i]->aux_h->data[k] * uTb[j*block_size+k];
		  for( l=0; l<block_size; l++)
		  {
		      va_a0[i*block_size+j] += U[i]->aux_h->data[l] * va_a1[l*block_size+k] * U[j]->aux_h->data[k];
		  }
	      } 
	   }
       }
       free(uTb);
    }

    HYPRE_ParCSRMatrix *M 	= (HYPRE_ParCSRMatrix*)data->M;
    pase_ParCSRMatrix **Mp 	= (pase_ParCSRMatrix**)data->Mp;
    HYPRE_ParCSRMatrix M0 	= NULL;
    HYPRE_ParCSRMatrix M1  	= NULL;
    pase_ParCSRMatrix *Mp0 	= NULL;
    pase_ParCSRMatrix *Mp1 	= NULL;
    {
	M0 	= M[cur_level-1];
	M1  	= M[cur_level];
	Mp1 	= Mp[cur_level];
	//printf("N_H of M0 = %d, N_H of M1 = %d\n", M0->global_num_rows, M1->global_num_rows);
	PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, M0, P0, M1, u_h, &Mp0, workspace_H, workspace_h);  
	Mp[cur_level-1] = (void*)Mp0;
	if( cur_level < max_level)
	{
	    hypre_ParVector** beta1 	= Mp1->aux_Hh;
	    hypre_ParVector** beta0 	= Mp0->aux_Hh;
	    HYPRE_Real* va_alpha1 	= Mp1->aux_hh->data;
	    HYPRE_Real* va_alpha0 	= Mp0->aux_hh->data;
	    /* aux_hH[cur_level-1] += P^T*aux_hH[cur_level]*Z */
	    for( i=0; i<block_size; i++)
	    {
		for( j=0; j<block_size; j++)
		{
		    hypre_ParCSRMatrixMatvecT( U[i]->aux_h->data[j], P0, beta1[j], 1.0, beta0[i]);
		}
	    }
	    /* aux_hh[cur_level-1] += 2*U^T*aux_hH[cur_level]*Z + Z^T*aux_hh[cur_level]*Z */
	    PASE_Real *uTbeta = (PASE_Real*)malloc( num_nonzeros*sizeof(PASE_Real));
	    for( i=0; i<block_size; i++)
	    {
		for( j=0; j<block_size; j++)
		{
		    uTbeta[j+i*block_size] = hypre_ParVectorInnerProd( U[i]->b_H, beta1[j]);
		}
	    }
	    for( i=0; i<block_size; i++)
	    {
		for( j=0; j<block_size; j++)
		{
		    for( k=0; k<block_size; k++)
		    {
			/* '2*'这个乘法操作可以优化 */
			va_alpha0[i*block_size+j] += 2 * U[i]->aux_h->data[k] * uTbeta[j*block_size+k];
			for( l=0; l<block_size; l++)
			{
			    va_alpha0[i*block_size+j] += U[i]->aux_h->data[l] * va_alpha1[l*block_size+k] * U[j]->aux_h->data[k];
			}
		    } 
		}
	    }
	    free(uTbeta);
	}
    }
	printf("beta[0] = %e, alpha[0] = %e\n", Ap0->aux_hh->data[0], Mp0->aux_hh->data[0]);
	printf("b[0] = %e, b[1] = %e\n", Ap0->aux_hH[0]->local_vector->data[0], Ap0->aux_hH[0]->local_vector->data[1]);
	printf("a[0] = %e, a[1] = %e\n", Mp0->aux_hH[0]->local_vector->data[0], Mp0->aux_hH[0]->local_vector->data[1]);
    hypre_TFree( u_h);
    HYPRE_ParVectorDestroy( workspace_H);
    HYPRE_ParVectorDestroy( workspace_h);
    PASE_Complex* eigenvalues0 = data->eigenvalues[cur_level-1];
    PASE_Complex* eigenvalues1 = data->eigenvalues[cur_level];
    for( i=0; i<block_size; i++)
    {
	eigenvalues0[i] = eigenvalues1[i];
    }
    return 0;
}

PASE_Int pase_MGAuxMatrixCreate( PASE_Solver solver)
{
    pase_MGData*        mg_solver    = (pase_MGData*)solver;
    PASE_Int            cur_level    = mg_solver->cur_level;
    PASE_Int            max_level    = mg_solver->max_level;
    PASE_Int            block_size   = mg_solver->block_size;
    HYPRE_ParCSRMatrix  *A 	     = (HYPRE_ParCSRMatrix*)mg_solver->A;
    HYPRE_ParCSRMatrix  *M 	= (HYPRE_ParCSRMatrix*)mg_solver->M;
    HYPRE_ParCSRMatrix  *P 	     = (HYPRE_ParCSRMatrix*)mg_solver->P;
    pase_ParCSRMatrix   **Ap 	     = (pase_ParCSRMatrix**)mg_solver->Ap;
    pase_ParCSRMatrix   **Mp 	= (pase_ParCSRMatrix**)mg_solver->Mp;
    pase_ParVector      ***u 	     = (pase_ParVector***)mg_solver->u;

    HYPRE_ParCSRMatrix  A0 	     = A[cur_level-1];
    HYPRE_ParCSRMatrix  A1 	     = A[cur_level];
    HYPRE_ParCSRMatrix  M0 	= M[cur_level-1];
    HYPRE_ParCSRMatrix  M1  	= M[cur_level];
    HYPRE_ParCSRMatrix  P0 	= P[cur_level-1];
    pase_ParCSRMatrix   *Ap0 	= NULL;
    pase_ParCSRMatrix   *Ap1 	= Ap[cur_level];
    pase_ParVector      **u1 	= u[cur_level];
    pase_ParVector      **u0 	= u[cur_level-1]; 
    pase_ParCSRMatrix   *Mp0 	= NULL;
    pase_ParCSRMatrix   *Mp1 	= Mp[cur_level];

    PASE_Int* partitioning_H 	= u0[0]->b_H->partitioning;
    PASE_Int N_H 		= u0[0]->b_H->global_size;
    HYPRE_ParVector workspace_H = hypre_ParVectorCreate( MPI_COMM_WORLD, N_H, partitioning_H); 
    hypre_ParVectorInitialize( workspace_H);
    hypre_ParVectorSetPartitioningOwner( workspace_H, 0);
    PASE_Int *partitioning_h = u1[0]->b_H->partitioning;
    PASE_Int  N_h            = u1[0]->b_H->global_size;

    PASE_Int i = 0;
    if(cur_level == max_level)
    {
        HYPRE_ParVector workspace_h = hypre_ParVectorCreate( MPI_COMM_WORLD, N_h, partitioning_h); 
        hypre_ParVectorInitialize( workspace_h);
        hypre_ParVectorSetPartitioningOwner( workspace_h, 0);

        HYPRE_ParVector     *u_h 	= hypre_CTAlloc(HYPRE_ParVector, block_size); 
        for( i=0; i<block_size; i++)
        {
            u_h[i] = u1[i]->b_H;
        }
        PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, A0, P0, A1, u_h, &Ap0, workspace_H, workspace_h);  
        PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, M0, P0, M1, u_h, &Mp0, workspace_H, workspace_h);  
        hypre_TFree( u_h);
        HYPRE_ParVectorDestroy( workspace_h);
    }
    else
    {
	PASE_ParVector workspace_hH;
      	PASE_ParVectorCreate( MPI_COMM_WORLD, N_h, block_size, NULL, partitioning_h, &workspace_hH);
	PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, A0, P0, Ap1, u1, &Ap0, workspace_H, workspace_hH);
	PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_COMM_WORLD, block_size, M0, P0, Mp1, u1, &Mp0, workspace_H, workspace_hH);
	PASE_ParVectorDestroy( workspace_hH);
	//printf("beta[0] = %e, alpha[0] = %e\n", Ap0->aux_hh->data[0], Mp0->aux_hh->data[0]);
	//printf("b[0] = %e, b[1] = %e\n", Ap0->aux_hH[0]->local_vector->data[0], Ap0->aux_hH[0]->local_vector->data[1]);
	//printf("a[0] = %e, a[1] = %e\n", Mp0->aux_hH[0]->local_vector->data[0], Mp0->aux_hH[0]->local_vector->data[1]);
    }
    Ap[cur_level-1] = (void*)Ap0;
    Mp[cur_level-1] = (void*)Mp0;

    HYPRE_ParVectorDestroy( workspace_H);

    PASE_Complex* eigenvalues0 = mg_solver->eigenvalues[cur_level-1];
    PASE_Complex* eigenvalues1 = mg_solver->eigenvalues[cur_level];
    for( i=0; i<block_size; i++)
    {
	eigenvalues0[i] = eigenvalues1[i];
    }
    return 0;
}

/*
 * 如果要重开始，释放辅助矩阵的内存空间
 */
PASE_Int pase_ParCSRMGRestart( PASE_Solver solver)
{
   PASE_Int 	i, j;
   pase_MGData* mg_solver 	= (pase_MGData*) solver;
   PASE_Int max_level 		= mg_solver->max_level;
   PASE_Int max_iter 		= mg_solver->max_iter;
   PASE_Int block_size		= mg_solver->block_size;
   PASE_Real num_converged	= mg_solver->num_converged;
   PASE_Int num_iter		= mg_solver->num_iter;

   //printf("max_iter = %d, num_iter = %d, R_norm = %f, atol = %f!\n", max_iter, num_iter, R_norm, atol);
   if(max_iter > num_iter && block_size > num_converged)
   {
	printf("Destroy memory for next iteration...\n");
        if( mg_solver->Ap != NULL)
        {
            PASE_ParCSRMatrix* Ap = (PASE_ParCSRMatrix*)mg_solver->Ap;
            for( i=0; i<max_level; i++)
            {
		PASE_ParCSRMatrixDestroy( Ap[i]);
            }
        }
        if( mg_solver->Mp != NULL)
        {
            PASE_ParCSRMatrix* Mp = (PASE_ParCSRMatrix*)mg_solver->Mp;
            for( i=0; i<max_level; i++)
            {
		PASE_ParCSRMatrixDestroy( Mp[i] );
            }
        }
        pase_ParVector*** u  = (pase_ParVector***) mg_solver->u;
        pase_ParVector** U_h = NULL;
        for( j=0; j<max_level; j++)
        {
	    U_h = u[j];
            for( i=0; i<block_size; i++)
            {
                hypre_ParVectorSetConstantValues( U_h[i]->b_H, 0.0);
                hypre_SeqVectorSetConstantValues( U_h[i]->aux_h, 0.0);
                HYPRE_Complex* aux_data = U_h[i]->aux_h->data;
                aux_data[i] 		= 1.0;
            }
        }
   }
   return 0;
}

/*
 * 释放内存
 */
PASE_Int PASE_ParCSRMGDestroy( PASE_Solver solver)
{
    PASE_Int i, j;
    pase_MGData* data 	= (pase_MGData*) solver;
    PASE_Int max_level 	= data->max_level;
    PASE_Int block_size	= data->block_size;

    if( data)
    {
	pase_MGFunctions* functions = data->functions;
	if( data->A != NULL)
	{
	    hypre_TFreeF( data->A, functions);
	    data->A = NULL;
	}
	if( data->M != NULL)
	{
	    hypre_TFreeF( data->M, functions);
	    data->M = NULL;
	}
	if( data->P != NULL)
	{
	    hypre_TFreeF( data->P, functions);
	    data->P = NULL;
	}
	if( data->Ap != NULL)
	{
	    PASE_ParCSRMatrix* Ap = (PASE_ParCSRMatrix*)data->Ap;
	    for( i=0; i<max_level; i++)
	    {
		PASE_ParCSRMatrixDestroy( Ap[i]);
	    }
	    hypre_TFreeF( data->Ap, functions);
	    data->Ap = NULL;
	}
	if( data->Mp != NULL)
	{
	    PASE_ParCSRMatrix* Mp = (PASE_ParCSRMatrix*)data->Mp;
	    for( i=0; i<max_level; i++)
	    {
		PASE_ParCSRMatrixDestroy( Mp[i]);
	    }
	    hypre_TFreeF( data->Mp, functions);
	    data->Mp = NULL;
	}
	if( data->u != NULL)
	{
	    PASE_ParVector** u = (PASE_ParVector**)data->u;
	    for( i=0; i<max_level+1; i++)
	    {
		PASE_ParVector* u_i = u[i];
		for( j=0; j<block_size; j++)
		{
		    PASE_ParVectorDestroy( u_i[j]);
		}
		hypre_TFreeF( u_i, functions);
		data->u[i] = NULL;
	    }
	    hypre_TFreeF( data->u, functions);
	    data->u = NULL;
	}
	if( data->eigenvalues != NULL)
	{
	    for( i=0; i<max_level+1; i++)
	    {
		hypre_TFreeF( data->eigenvalues[i], functions);
	    }
	    hypre_TFreeF( data->eigenvalues, functions);
	    data->eigenvalues = NULL;
	}
	hypre_TFreeF( data, functions);
	hypre_TFreeF( functions, functions);
    }
    return hypre_error_flag;
}

/*
 * 设置MG最大迭代次数
 */ 
PASE_Int PASE_MGSetMaxIter( PASE_Solver solver, PASE_Int max_iter)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->max_iter		= max_iter;
    return 0;
}

/*
 * 设置前光滑最大迭代次数
 */
PASE_Int PASE_MGSetPreIter( PASE_Solver solver, PASE_Int pre_iter)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->pre_iter		= pre_iter;
    return 0;
}

/*
 * 设置后光滑最大迭代次数
 */
PASE_Int PASE_MGSetPostIter( PASE_Solver solver, PASE_Int post_iter)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->post_iter	= post_iter;
    return 0;
}

/*
 * 设置求解特征值个数
 */
PASE_Int PASE_MGSetBlockSize( PASE_Solver solver, PASE_Int block_size)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->block_size	= block_size;
    return 0;
}

PASE_Int PASE_MGSetPrintLevel( PASE_Solver solver, PASE_Int print_level)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->print_level	= print_level;
    return 0;
}

PASE_Int PASE_MGSetATol( PASE_Solver solver, PASE_Real atol)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->atol		= atol;
    return 0;
}

PASE_Int PASE_MGSetRTol( PASE_Solver solver, PASE_Real rtol)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->rtol		= rtol;
    return 0;
}

PASE_Int PASE_MGSetExactEigenvalues( PASE_Solver solver, PASE_Complex* exact_eigenvalues)
{
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    mg_solver->exact_eigenvalues= exact_eigenvalues;
    return 0;
}

/*
 * 误差估计
 */
PASE_Int pase_ParCSRMGErrorEstimate( PASE_Solver solver)
{
    pase_MGData* mg_solver 	= (pase_MGData*) solver;
    PASE_Int max_level		= mg_solver->max_level;
    PASE_Int block_size		= mg_solver->block_size; 
    pase_ParVector*** u		= (pase_ParVector***) mg_solver->u;	
    pase_ParVector** u0		= u[max_level];
    PASE_Complex* eigenvalues 	= mg_solver->eigenvalues[max_level];
    hypre_ParCSRMatrix** A	= (hypre_ParCSRMatrix**) mg_solver->A;
    hypre_ParCSRMatrix** M	= (hypre_ParCSRMatrix**) mg_solver->M;
    hypre_ParCSRMatrix* A0	= A[max_level];
    hypre_ParCSRMatrix* M0	= NULL;    

    if( M != NULL)
    {
	M0 = M[max_level];
    }

    PASE_Int N_H  		= u0[0]->N_H;
    PASE_Int *partitioning 	= u0[0]->b_H->partitioning;
    HYPRE_ParVector r = hypre_ParVectorCreate(MPI_COMM_WORLD, N_H, partitioning);
    hypre_ParVectorInitialize( r);
    hypre_ParVectorSetPartitioningOwner( r, 0);

    /* 计算最细层的残差：r = Au - kMu */
    PASE_Real atol = mg_solver->atol;
    //PASE_Real rtol = mg_solver->rtol;
    PASE_Int flag 	= 0;
    PASE_Int i		= 0;
    PASE_Real r_norm 	= 0;
    while( mg_solver->num_converged < block_size && flag == 0)
    {
	//printf("A0 have %d rows amd %d cols, u0[0] has %d rows\n", A0->global_num_rows, A0->global_num_cols, u0[i]->b_H->global_size);
	i = mg_solver->num_converged;
	hypre_ParCSRMatrixMatvec( 1.0, A0, u0[i]->b_H, 0.0, r);
	if( M0 != NULL)
	{
	    hypre_ParCSRMatrixMatvec( -eigenvalues[i], M0, u0[i]->b_H, 1.0, r); 
	}
	else
	{
	    hypre_ParVectorAxpy( -eigenvalues[i], u0[i]->b_H, r);
	}
	//u_norm 	= hypre_ParVectorInnerProd( u0[i]->b_H, u0[i]->b_H);
	//u_norm 	= sqrt(u_norm);
	r_norm 	= hypre_ParVectorInnerProd( r, r);
	r_norm	= sqrt(r_norm);
	mg_solver->r_norm = r_norm;
	if( r_norm < atol)
	{
	    mg_solver->num_converged ++;
	}
	else
	{
	    flag = 1;
	}
    }
    hypre_ParVectorDestroy(r);

    if( mg_solver->print_level > 0)
    {
	//printf("eigenvalues[0] = %.16f, exact_eigenvalues[0] = %.16f\n", mg_solver->eigenvalues[max_level][0], mg_solver->exact_eigenvalues[0]);
	PASE_Real error = fabs(mg_solver->eigenvalues[max_level][0] - mg_solver->exact_eigenvalues[0]);	
	printf("iter = %d, error of eigen[0] = %1.6e, the num of converged = %d, the norm of residul = %1.6e\n", mg_solver->num_iter, error, mg_solver->num_converged, mg_solver->r_norm);
    }	
    return 0;
}

/*
 * 基于辅助矩阵结构的特征值问题直接求解
 */
PASE_Int pase_ParCSRMGDirSolver( PASE_Solver solver)
{
    HYPRE_Solver lobpcg_solver 	= NULL; 
    int maxIterations 		= 5000; 	/* maximum number of iterations */
    int pcgMode 		= 1;    	/* use rhs as initial guess for inner pcg iterations */
    int verbosity 		= 0;    	/* print iterations info */
    double tol 			= 1.e-10;	/* absolute tolerance (all eigenvalues) */
    int lobpcgSeed 		= 77;

    PASE_Int i;
    pase_MGData *mg_solver 	= (pase_MGData *)solver;
    PASE_Int block_size 	= mg_solver->block_size;
    PASE_Int cur_level 		= mg_solver->cur_level;
    pase_ParCSRMatrix** Ap 	= (pase_ParCSRMatrix**)mg_solver->Ap;
    pase_ParCSRMatrix** Mp 	= (pase_ParCSRMatrix**)mg_solver->Mp;
    pase_ParCSRMatrix* Ap0 	= Ap[cur_level];
    pase_ParCSRMatrix* Mp0 	= NULL;
    pase_ParVector*** u 	= (pase_ParVector***)mg_solver->u;
    pase_ParVector** u0 	= u[cur_level];

    HYPRE_Complex* eigenvalues 	= mg_solver->eigenvalues[cur_level];
    //eigenvalues = (HYPRE_Complex*)calloc(block_size, sizeof(HYPRE_Complex));
    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc( mv_InterfaceInterpreter, 1);
    PASE_ParCSRSetupInterpreter( interpreter_Hh);
    PASE_ParCSRSetupMatvec( &matvec_fn_Hh);
    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector( interpreter_Hh, block_size, u0[0]);
    mv_MultiVectorSetRandom( eigenvectors_Hh, lobpcgSeed);
    PASE_ParVector x 		= NULL;
    PASE_Int N_H 		= u0[0]->b_H->global_size;
    PASE_Int *partitioning 	= u0[0]->b_H->partitioning;
    PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL, partitioning, &x);
     
    HYPRE_LOBPCGCreate( interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
    HYPRE_LOBPCGSetMaxIter( lobpcg_solver, maxIterations);
    HYPRE_LOBPCGSetPrecondUsageMode( lobpcg_solver, pcgMode);
    HYPRE_LOBPCGSetTol( lobpcg_solver, tol);
    HYPRE_LOBPCGSetPrintLevel( lobpcg_solver, verbosity);

    PASE_LOBPCGSetup( lobpcg_solver, Ap0, u0[0], u0[0]);
    if( Mp != NULL)
    {
	Mp0 = Mp[cur_level];
	if( Mp0 != NULL)
	{
	    PASE_LOBPCGSetupB( lobpcg_solver, Mp0, u0[0]);
	}
	else
	{
	    printf("Error: cur_level = %d, matrix Mp is NULL!\n", cur_level);
	}
    }
    HYPRE_LOBPCGSolve( lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

#if 0
    PASE_Real inner_A, inner_M;
	PASE_ParCSRMatrixMatvec( 1.0, Ap0, u0[0], 0.0, x);
	PASE_ParVectorInnerProd( x, u0[0], &inner_A);
	if(Mp0!=NULL)
	{
	    PASE_ParCSRMatrixMatvec( 1.0, Mp0, u0[0], 0.0, x);
	    PASE_ParVectorInnerProd( x, u0[0], &inner_M);
	}
	else
	{
	    PASE_ParVectorInnerProd( u0[0], u0[0], &inner_M);
	}
	printf("eigenvalues[%d] = %.16f, inner_A = %.30f, inner_M = %.20f,  u0[0]->b_H->data[0] = %.20f\n", 0, inner_A/inner_M, inner_A, inner_M, u0[0]->aux_h->data[0]);
#endif

    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
    u[cur_level] = (PASE_ParVector*)(tmp -> vector);
    hypre_ParVectorSetPartitioningOwner( u0[0]->b_H, 0);
    hypre_ParVectorSetPartitioningOwner( u[cur_level][0]->b_H, 1);

#if 0
    PASE_Real inner_A, inner_M;
    PASE_ParCSRMatrixMatvec( 1.0, Ap0, u[cur_level][0], 0.0, x);
    PASE_ParVectorInnerProd( x, u[cur_level][0], &inner_A);
    if( Mp0 != NULL)
    {
	PASE_ParCSRMatrixMatvec( 1.0, Mp0, u[cur_level][0], 0.0, x);
        PASE_ParVectorInnerProd( x, u[cur_level][0], &inner_M);
    }
    else
    {
        PASE_ParVectorInnerProd( u[cur_level][0], u[cur_level][0], &inner_M);
    }
    printf("after directly solving, eigenvalues[%d] = %.16f, inner_A = %.16f, inner_M = %.16f\n", 0, inner_A/inner_M, inner_A, inner_M);
#endif
    
    if( mg_solver->print_level > 1)
    {
	printf("Cur_level %d", cur_level);
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }

    for( i=0; i<block_size; i++)
    {
	PASE_ParVectorDestroy( u0[i] );
    }
    hypre_TFreeF( u0, mg_solver->functions);
    free( (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
    hypre_TFree( eigenvectors_Hh);
    hypre_TFree( interpreter_Hh);

    PASE_ParVectorDestroy( x);
    HYPRE_LOBPCGDestroy( lobpcg_solver);
    return 0;
}

/*
 * CG光滑子
 */
PASE_Int pase_ParCSRMGSmootherCG( PASE_Solver solver)
{
    //printf("Using CG as the smoother of MG method!\n");
    pase_MGData *mg_solver = (pase_MGData*)solver;
    HYPRE_Solver cg_solver = NULL;
    HYPRE_ParCSRPCGCreate( MPI_COMM_WORLD, &cg_solver);
    /* Set some parameters */
    HYPRE_PCGSetMaxIter( cg_solver, mg_solver->pre_iter); /* max iterations */
    HYPRE_PCGSetTol( cg_solver, 1.0e-50); 
    HYPRE_PCGSetTwoNorm( cg_solver, 1); /* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel( cg_solver, 0); 
    HYPRE_PCGSetLogging( cg_solver, 1); /* needed to get run info lat    er */
           
    /* Setup and solve */
    PASE_Int cur_level 		= mg_solver->cur_level;
    PASE_Int block_size 		= mg_solver->block_size;
    PASE_Int num_converged      = mg_solver->num_converged; 
    PASE_Int i 			= 0;
    HYPRE_ParCSRMatrix* solver_A 	= (HYPRE_ParCSRMatrix*)mg_solver->A;
    HYPRE_ParCSRMatrix* solver_M 	= (HYPRE_ParCSRMatrix*)mg_solver->M;
    HYPRE_ParCSRMatrix A 		= solver_A[cur_level];
    HYPRE_ParCSRMatrix M 		= NULL;

    if( solver_M != NULL)
    {
	M = solver_M[cur_level];
    }
    pase_ParVector ***solver_u 	= (pase_ParVector***)mg_solver->u;
    pase_ParVector **u 		= solver_u[cur_level];
    hypre_ParVector *rhs 	= NULL; 
    PASE_Int N_h 		= u[0]->b_H->global_size;
    PASE_Int* partitioning 	= u[0]->b_H->partitioning;
    MPI_Comm comm 		= u[0]->b_H->comm;
    PASE_Complex inner_A, inner_M;
    PASE_Complex* eigenvalues 	= mg_solver->eigenvalues[cur_level];
    rhs 			= hypre_ParVectorCreate( comm, N_h, partitioning);
    hypre_ParVectorInitialize( rhs);
    hypre_ParVectorSetPartitioningOwner( rhs, 0);
    for( i=num_converged; i<block_size; i++)
    {
#if 0
	hypre_ParCSRMatrixMatvec( 1.0, A, u[i]->b_H, 0.0, rhs);
	inner_A = hypre_ParVectorInnerProd( rhs, u[i]->b_H);
	if( M != NULL)
	{
	    hypre_ParCSRMatrixMatvec( 1.0, M, u[i]->b_H, 0.0, rhs);
	    inner_M = hypre_ParVectorInnerProd( rhs, u[i]->b_H);
	}
	else
	{
	    inner_M = hypre_ParVectorInnerProd( u[i]->b_H, u[i]->b_H);
	}
	printf("before smoothing, eigenvalues[%d] = %.16f, inner_A = %.20f, inner_M = %.20f\n", i, inner_A/inner_M, inner_A, inner_M);
#endif

	if( M != NULL)
	{
	    hypre_ParCSRMatrixMatvec( 1.0, M, u[i]->b_H, 0.0, rhs);
	}
	else
	{
	    hypre_ParVectorCopy( u[i]->b_H, rhs);
	}
	hypre_ParVectorScale( eigenvalues[i], rhs);
	hypre_PCGSetup( cg_solver, A, rhs, u[i]->b_H);
	hypre_PCGSolve( cg_solver, A, rhs, u[i]->b_H);

	hypre_ParCSRMatrixMatvec( 1.0, A, u[i]->b_H, 0.0, rhs);
	inner_A = hypre_ParVectorInnerProd( rhs, u[i]->b_H);
	if( M != NULL)
	{
	    hypre_ParCSRMatrixMatvec( 1.0, M, u[i]->b_H, 0.0, rhs);
	    inner_M = hypre_ParVectorInnerProd( rhs, u[i]->b_H);
	}
	else
	{
	    inner_M = hypre_ParVectorInnerProd( u[i]->b_H, u[i]->b_H);
	}
	eigenvalues[i] = inner_A / inner_M;
    }
    if( mg_solver->print_level > 1)
    {
	printf("Cur_level %d", cur_level);
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }
    hypre_ParVectorDestroy( rhs);
    PASE_ParCSRPCGDestroy( cg_solver);
    return 0;
}

/*
 * 基于辅助矩阵结构的CG光滑子
 */
PASE_Int pase_ParCSRMGInAuxSmootherCG( PASE_Solver solver)
{
    pase_MGData *mg_solver 	= (pase_MGData*)solver;
    HYPRE_Solver cg_solver 	= NULL;
    PASE_ParCSRPCGCreate( MPI_COMM_WORLD, &cg_solver);
    HYPRE_PCGSetMaxIter( cg_solver, mg_solver->pre_iter); 	/* max iterations */
    HYPRE_PCGSetTol( cg_solver, 1e-50); 			/* conv. tolerance */
    HYPRE_PCGSetTwoNorm( cg_solver, 1); 			/* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel( cg_solver, 0); 			/* prints out the iteratio    n info */
    HYPRE_PCGSetLogging( cg_solver, 1); 			/* needed to get run info lat    er */
 
    /* Now setup and solve! */
    PASE_Int cur_level 	= mg_solver->cur_level;
    PASE_Int block_size 	= mg_solver->block_size;
    PASE_Int num_converged      = mg_solver->num_converged;
    PASE_Int i			= 0;
    pase_ParCSRMatrix **solver_Ap 	= (pase_ParCSRMatrix**)mg_solver->Ap;
    pase_ParCSRMatrix **solver_Mp 	= (pase_ParCSRMatrix**)mg_solver->Mp;
    pase_ParCSRMatrix *Ap 	= solver_Ap[cur_level];
    pase_ParCSRMatrix *Mp 	= NULL;
    PASE_Complex* eigenvalues 	= mg_solver->eigenvalues[cur_level];
    if( solver_Mp != NULL)
    {
	Mp = solver_Mp[cur_level];
    }
    pase_ParVector ***solver_u 	= (pase_ParVector***)mg_solver->u;
    pase_ParVector **u 		= solver_u[cur_level];

    PASE_Complex inner_A, inner_M;
    pase_ParVector *rhs 	= NULL;  
    PASE_Int N_H		= u[0]->b_H->global_size;
    PASE_Int *partitioning 	= u[0]->b_H->partitioning;
    PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL, partitioning, &rhs);

    for( i=num_converged; i<block_size; i++)
    {
#if 0
	PASE_ParCSRMatrixMatvec( 1.0, Ap, u[i], 0.0, rhs);
	PASE_ParVectorInnerProd( rhs, u[i], &inner_A);
	if( Mp != NULL)
	{
	    PASE_ParCSRMatrixMatvec( 1.0, Mp, u[i], 0.0, rhs);
	    PASE_ParVectorInnerProd( rhs, u[i], &inner_M);
	}
	else
	{
	    PASE_ParVectorInnerProd( u[i], u[i], &inner_M);
	}
	printf("before cg smoothing, eigenvalues[%d] = %.16f\n", i, inner_A/inner_M);
#endif
	if( Mp != NULL)
	{
	    PASE_ParCSRMatrixMatvec( 1.0, Mp, u[i], 0.0, rhs);
	}
	else
	{
	    PASE_ParVectorCopy( u[i], rhs);
	}
	//printf("eigenvalues = %.16f\n", eigenvalues[0]);
	PASE_ParVectorScale( eigenvalues[i], rhs);

	hypre_PCGSetup( cg_solver, Ap, rhs, u[i]);
	hypre_PCGSolve( cg_solver, Ap, rhs, u[i]);

	//PASE_ParCSRMatrixMatvec( -1.0, Ap, u[i], 1.0, rhs);
	//PASE_ParVectorInnerProd( rhs, rhs, &inner_A);
	//PASE_ParVectorInnerProd( u[i], u[i], &inner_M);
	//printf("the norm of residual after pcg is %.6f, the norm of u is %.6f\n", inner_A, inner_M);
	//PASE_Int num;
	//HYPRE_PCGGetNumIterations(cg_solver, &num);
	//printf("the num of iteration is %d\n", num);

	PASE_ParCSRMatrixMatvec( 1.0, Ap, u[i], 0.0, rhs);
	PASE_ParVectorInnerProd( rhs, u[i], &inner_A);
	if( Mp != NULL)
	{
	    PASE_ParCSRMatrixMatvec( 1.0, Mp, u[i], 0.0, rhs);
	    PASE_ParVectorInnerProd( rhs, u[i], &inner_M);
	}
	else
	{
	    PASE_ParVectorInnerProd( u[i], u[i], &inner_M);
	}
	eigenvalues[i] = inner_A / inner_M;
    }
    if( mg_solver->print_level > 1)
    {
	printf("Cur_level %d", cur_level);
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }

    PASE_ParVectorDestroy( rhs);
    PASE_ParCSRPCGDestroy( cg_solver);
    return 0;
}

void 
PASE_Orth(PASE_Solver solver)
{
    pase_MGData *mg_solver 	= (pase_MGData*)solver;
    PASE_Int cur_level 	= mg_solver->cur_level;
    PASE_Int max_level 	= mg_solver->max_level;
    PASE_Int block_size	= mg_solver->block_size;
    pase_ParVector ***solver_u 	= (pase_ParVector***)mg_solver->u;
    pase_ParVector **u 		= solver_u[cur_level];
    hypre_ParCSRMatrix **solver_M = (hypre_ParCSRMatrix**) mg_solver->M;
    hypre_ParCSRMatrix *M = solver_M[cur_level];
    pase_ParCSRMatrix **solver_Mp = (pase_ParCSRMatrix**) mg_solver->Ap;
    pase_ParCSRMatrix *Mp = solver_Mp[cur_level];

    PASE_Int i, j;
    PASE_Real inner_ij, norm_i;

    pase_ParVector *rhs 	= NULL;  
    PASE_Int N_H		= u[0]->b_H->global_size;
    PASE_Int *partitioning 	= u[0]->b_H->partitioning;
    PASE_ParVectorCreate( MPI_COMM_WORLD, N_H, block_size, NULL, partitioning, &rhs);

    if(cur_level == max_level) {
	for(i=0; i<block_size; i++) {
	   for(j=0; j<i; j++) {
	       HYPRE_ParCSRMatrixMatvec(1.0, M, u[i]->b_H, 0.0, rhs->b_H); 
	       inner_ij = hypre_ParVectorInnerProd(rhs->b_H, u[j]->b_H);
	       hypre_ParVectorAxpy(-inner_ij, u[j]->b_H, u[i]->b_H);
	   } 
	   HYPRE_ParCSRMatrixMatvec(1.0, M, u[i]->b_H, 0.0, rhs->b_H); 
	   norm_i = hypre_ParVectorInnerProd(rhs->b_H, u[i]->b_H);
	   norm_i = sqrt(norm_i);
	   //printf("norm[%d] = %.3f\n", i, norm_i);
	   hypre_ParVectorScale( 1.0/norm_i, u[i]->b_H);
	}
	       //HYPRE_ParCSRMatrixMatvec(1.0, M, u[0]->b_H, 0.0, rhs->b_H); 
	       //inner_ij = hypre_ParVectorInnerProd(rhs->b_H, u[1]->b_H);
	       //printf("inner_ij = %.4f\n", inner_ij);
    } else {
	for(i=0; i<block_size; i++) {
	   for(j=0; j<i; j++) {
	       PASE_ParCSRMatrixMatvec(1.0, Mp, u[i], 0.0, rhs);
	       PASE_ParVectorInnerProd(rhs, u[j], &inner_ij);
	       PASE_ParVectorAxpy(-inner_ij, u[j], u[i]);
	   } 
	   PASE_ParCSRMatrixMatvec(1.0, Mp, u[i], 0.0, rhs);
	   PASE_ParVectorInnerProd(rhs, u[i], &norm_i);
	   norm_i = sqrt(norm_i);
	   //printf("norm[%d] = %.3f\n", i, norm_i);
	   PASE_ParVectorScale( 1.0/norm_i, u[i]);
	}
	       //PASE_ParCSRMatrixMatvec(1.0, Mp, u[0], 0.0, rhs);
	       //PASE_ParVectorInnerProd(rhs, u[1], &inner_ij);
	       //printf("inner_ij = %.4f\n", inner_ij);
	   //PASE_ParCSRMatrixMatvec(1.0, Mp, u[0], 0.0, rhs);
	   //PASE_ParVectorInnerProd(rhs, u[0], &norm_i);
	   //norm_i = sqrt(norm_i);
	   //printf("norm[0] = %.3f\n", norm_i);
	   //PASE_ParCSRMatrixMatvec(1.0, Mp, u[1], 0.0, rhs);
	   //PASE_ParVectorInnerProd(rhs, u[1], &norm_i);
	   //norm_i = sqrt(norm_i);
	   //printf("norm[1] = %.3f\n", norm_i);
    }
    PASE_ParVectorDestroy(rhs);
}

void PASE_Get_initial_vector(PASE_Solver solver)
{
    PASE_Int i;
    pase_MGData *mg_solver 	= (pase_MGData*)solver;
    PASE_Int max_level          = mg_solver->max_level;        
    PASE_Int cur_level          = mg_solver->cur_level;        
    PASE_Int block_size         = mg_solver->block_size;       
    HYPRE_ParCSRMatrix *P       = (HYPRE_ParCSRMatrix*)mg_solver->P;
    HYPRE_ParCSRMatrix *A       = (HYPRE_ParCSRMatrix*)mg_solver->A;
    HYPRE_ParCSRMatrix *M       = (HYPRE_ParCSRMatrix*)mg_solver->M;
    HYPRE_ParCSRMatrix A_H          = A[0];
    HYPRE_ParCSRMatrix M_H          = M[0];
    pase_ParVector ***solver_u 	= (pase_ParVector***)mg_solver->u;
    pase_ParVector **u		= solver_u[0];

    HYPRE_Solver lobpcg_solver 	= NULL; 
    int maxIterations 		= 5000; 	/* maximum number of iterations */
    int pcgMode 		= 1;    	/* use rhs as initial guess for inner pcg iterations */
    int verbosity 		= 0;    	/* print iterations info */
    double tol 			= 1.e-30;	/* absolute tolerance (all eigenvalues) */
    int lobpcgSeed 		= 77;
    HYPRE_Complex* eigenvalues 	= mg_solver->eigenvalues[max_level];
    //eigenvalues = (HYPRE_Complex*)calloc(block_size, sizeof(HYPRE_Complex));
    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc( mv_InterfaceInterpreter, 1);
    HYPRE_ParCSRSetupInterpreter( interpreter_Hh);
    HYPRE_ParCSRSetupMatvec( &matvec_fn_Hh);
    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector( interpreter_Hh, block_size, u[0]->b_H);
    mv_MultiVectorSetRandom( eigenvectors_Hh, lobpcgSeed);
     
    HYPRE_LOBPCGCreate( interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
    HYPRE_LOBPCGSetMaxIter( lobpcg_solver, maxIterations);
    HYPRE_LOBPCGSetPrecondUsageMode( lobpcg_solver, pcgMode);
    HYPRE_LOBPCGSetTol( lobpcg_solver, tol);
    HYPRE_LOBPCGSetPrintLevel( lobpcg_solver, verbosity);

    HYPRE_LOBPCGSetup( lobpcg_solver, (HYPRE_Matrix)A_H, (HYPRE_Vector)(u[0]->b_H), (HYPRE_Vector)(u[0]->b_H));
    HYPRE_LOBPCGSetupB( lobpcg_solver, (HYPRE_Matrix)M_H, (HYPRE_Vector)(u[0]->b_H));
    HYPRE_LOBPCGSolve( lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

    if( mg_solver->print_level > 1)
    {
	printf("Cur_level %d", cur_level);
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }

    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
    HYPRE_ParVector *u_H = (HYPRE_ParVector*)(tmp -> vector);
    HYPRE_ParVector u_h = NULL; 
    PASE_Int j;
    for(j=1; j<=max_level; j++) {
	MPI_Comm comm = A[j]->comm;
	PASE_Int N_h  = A[j]->global_num_rows;
	PASE_Int *partitioning = NULL;
	HYPRE_ParCSRMatrixGetRowPartitioning(A[j], &partitioning);
        for(i=0; i<block_size; i++) {
            HYPRE_ParVectorCreate(comm, N_h, partitioning, &u_h);
	    hypre_ParVectorInitialize(u_h);
	    if(i == 0) {
		hypre_ParVectorSetPartitioningOwner(u_h, 1);
	    } else {
		hypre_ParVectorSetPartitioningOwner(u_h, 0);
	    }
	    HYPRE_ParCSRMatrixMatvec(1.0, P[j-1], u_H[i], 0.0, u_h); 
	    HYPRE_ParVectorDestroy(u_H[i]);
	    u_H[i] = u_h;
	    u_h    = NULL;
        }
    }

    for( i=0; i<block_size; i++)
    {
	HYPRE_ParVectorDestroy( solver_u[max_level][i]->b_H );
	solver_u[max_level][i]->b_H = u_H[i];
    }
    hypre_TFreeF( u_H, mg_solver->functions);
    free( (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
    hypre_TFree( eigenvectors_Hh);
    hypre_TFree( interpreter_Hh);

    HYPRE_LOBPCGDestroy( lobpcg_solver);
}

PASE_Int
PASE_Cg(PASE_Solver solver)
{
    pase_MGData* mg_solver  = (pase_MGData*) solver;
    PASE_Int     cur_level  = mg_solver->cur_level;
    PASE_Int     max_level  = mg_solver->max_level;
    PASE_Int     block_size = mg_solver->block_size;

    HYPRE_ParCSRMatrix *A_array  = (HYPRE_ParCSRMatrix*) mg_solver->A;
    HYPRE_ParCSRMatrix  A        = A_array[cur_level];
    HYPRE_ParCSRMatrix *M_array  = (HYPRE_ParCSRMatrix*) mg_solver->M;
    HYPRE_ParCSRMatrix  M        = M_array[cur_level];
    PASE_ParCSRMatrix  *Ap_array = (PASE_ParCSRMatrix*) mg_solver->Ap;
    PASE_ParCSRMatrix   Ap       = Ap_array[cur_level];
    PASE_ParCSRMatrix  *Mp_array = (PASE_ParCSRMatrix*) mg_solver->Mp;
    PASE_ParCSRMatrix   Mp       = Mp_array[cur_level];
    PASE_ParVector    **u_array  = (PASE_ParVector**)   mg_solver->u;
    PASE_ParVector     *u        = u_array[cur_level];
    PASE_Complex       *eigenvalues = mg_solver->eigenvalues[cur_level];

    MPI_Comm comm = u[0]->comm;
    PASE_Int N_H  = u[0]->N_H;
    PASE_Int *partitioning = u[0]->b_H->partitioning;
    PASE_ParVector residual = NULL;
    PASE_ParVectorCreate(comm, N_H, block_size, NULL, partitioning, &residual);
    PASE_ParVector p = NULL;
    PASE_ParVectorCreate(comm, N_H, block_size, NULL, partitioning, &p);
    PASE_ParVector q = NULL;
    PASE_ParVectorCreate(comm, N_H, block_size, NULL, partitioning, &q);

    PASE_Int i, j;
    PASE_Real bnorm, rnorm, rho, rho_1, alpha, beta;
    PASE_Real tmp;
    PASE_Real tol = 1e-10;
    PASE_Real inner_A, inner_M;
    if(cur_level == max_level) {
	for(j=mg_solver->num_converged; j<block_size; j++) {
	    HYPRE_ParCSRMatrixMatvec(eigenvalues[j], M, u[j]->b_H, 0.0, residual->b_H);
	    HYPRE_ParVectorInnerProd(residual->b_H, residual->b_H, &bnorm);
	    bnorm = sqrt(bnorm);
	    HYPRE_ParCSRMatrixMatvec(-1.0, A, u[j]->b_H, 1.0, residual->b_H);
	    HYPRE_ParVectorInnerProd(residual->b_H, residual->b_H, &rho);
	    rnorm = sqrt(rho);
	    if(rnorm/bnorm < tol) {
	        continue;
	    }
            for(i=0; i<mg_solver->pre_iter; i++) {
		if(i>0) {
		    beta = rho / rho_1; 
		    HYPRE_ParVectorScale(beta, p->b_H);
		    HYPRE_ParVectorAxpy(1.0, residual->b_H, p->b_H);
		} else {
		    HYPRE_ParVectorCopy(residual->b_H, p->b_H);
		}
		HYPRE_ParCSRMatrixMatvec(1.0, A, p->b_H, 0.0, q->b_H);
		HYPRE_ParVectorInnerProd(p->b_H, q->b_H, &tmp);
		alpha = rho / tmp;
		HYPRE_ParVectorAxpy(alpha, p->b_H, u[j]->b_H);
		HYPRE_ParVectorAxpy(-1.0*alpha, q->b_H, residual->b_H);

		rho_1 = rho;
		HYPRE_ParVectorInnerProd(residual->b_H, residual->b_H, &rho);
		rnorm = sqrt(rho);

	        if(rnorm/bnorm < tol) {
	            continue;
	        }
	    }
	    PASE_Vector_inner_production_general_hypre(A, u[j]->b_H, u[j]->b_H, &inner_A);
	    PASE_Vector_inner_production_general_hypre(M, u[j]->b_H, u[j]->b_H, &inner_M);
	    eigenvalues[j] = inner_A / inner_M;
	}
    } else {
	for(j=mg_solver->num_converged; j<block_size; j++) {
	    PASE_ParCSRMatrixMatvec(eigenvalues[j], Mp, u[j], 0.0, residual);
	    PASE_ParVectorInnerProd(residual, residual, &bnorm);
	    bnorm = sqrt(bnorm);
	    PASE_ParCSRMatrixMatvec(-1.0, Ap, u[j], 1.0, residual);
	    PASE_ParVectorInnerProd(residual, residual, &rho);
	    rnorm = sqrt(rho);
	    if(rnorm/bnorm < tol) {
	        continue;
	    }
            for(i=0; i<mg_solver->pre_iter; i++) {
		if(i>0) {
		    beta = rho / rho_1; 
		    PASE_ParVectorScale(beta, p);
		    PASE_ParVectorAxpy(1.0, residual, p);
		} else {
		    PASE_ParVectorCopy(residual, p);
		}
		PASE_ParCSRMatrixMatvec(1.0, Ap, p, 0.0, q);
		PASE_ParVectorInnerProd(p, q, &tmp);
		alpha = rho / tmp;
		PASE_ParVectorAxpy(alpha, p, u[j]);
		PASE_ParVectorAxpy(-1.0*alpha, q, residual);

		rho_1 = rho;
		PASE_ParVectorInnerProd(residual, residual, &rho);
		rnorm = sqrt(rho);

	        if(rnorm/bnorm < tol) {
	            continue;
	        }
	    }
	    PASE_Vector_inner_production_general(Ap, u[j], u[j], &inner_A);
	    PASE_Vector_inner_production_general(Mp, u[j], u[j], &inner_M);
	    eigenvalues[j] = inner_A / inner_M;
	}
    }
#if 0
    if( mg_solver->print_level > 1)
    {
        printf("Cur_level %d", cur_level);
        for( i=0; i<block_size; i++)
        {
            printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
        }
        printf(".\n");
    }
#endif

    PASE_ParVectorDestroy(residual);
    PASE_ParVectorDestroy(p);
    PASE_ParVectorDestroy(q);

    
    return 0;
}

PASE_Int
PASE_Vector_inner_production_general_hypre(HYPRE_ParCSRMatrix A, HYPRE_ParVector x, HYPRE_ParVector y, PASE_Real *prod) 
{
    MPI_Comm comm = x->comm;
    PASE_Int N_H  = x->global_size;
    PASE_Int *partitioning = x->partitioning;
    HYPRE_ParVector tmp = NULL;
    HYPRE_ParVectorCreate(comm, N_H, partitioning, &tmp); 
    HYPRE_ParVectorInitialize(tmp);
    hypre_ParVectorSetPartitioningOwner(tmp, 0);
    HYPRE_ParCSRMatrixMatvec(1.0, A, y, 0.0, tmp);
    HYPRE_ParVectorInnerProd(x, tmp, prod);
    HYPRE_ParVectorDestroy(tmp);
    return 0;
}

PASE_Int
PASE_Vector_inner_production_general(PASE_ParCSRMatrix A, PASE_ParVector x, PASE_ParVector y, PASE_Real *prod) 
{
    MPI_Comm comm = x->comm;
    PASE_Int N_H  = x->N_H;
    PASE_Int block_size = x->block_size;
    PASE_Int *partitioning = x->b_H->partitioning;
    PASE_ParVector tmp = NULL;
    PASE_ParVectorCreate(comm, N_H, block_size, NULL, partitioning, &tmp); 
    PASE_ParCSRMatrixMatvec(1.0, A, y, 0.0, tmp);
    PASE_ParVectorInnerProd(x, tmp, prod);
    PASE_ParVectorDestroy(tmp);
    return 0;
}
