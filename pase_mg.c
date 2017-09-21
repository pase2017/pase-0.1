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
pase_MGFunctions *pase_MGFunctionsCreate( 
   char *       (*CAlloc)        ( size_t count, size_t elt_size ),
   PASE_Int     (*Free)          ( char *ptr ),
   PASE_Int     (*CommInfo)      ( void  *A, PASE_Int   *my_id,
                                   PASE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   PASE_Int     (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   PASE_Int     (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*DireSolver)    ( PASE_Solver solver ),
   HYPRE_Int    (*PreSmooth)     ( PASE_Solver solver ),
   HYPRE_Int    (*PostSmooth)    ( PASE_Solver solver ),
   HYPRE_Int    (*PreSmoothInAux)( PASE_Solver solver ),
   HYPRE_Int    (*PostSmoothInAux)( PASE_Solver solver )
    ) 
{
    pase_MGFunctions *mg_functions;
    mg_functions = (pase_MGFunctions*)CAlloc( 1, sizeof(pase_MGFunctions));

    mg_functions->CAlloc = CAlloc;
    mg_functions->Free   = Free;
    mg_functions->CommInfo = CommInfo;
    mg_functions->CreateVector = CreateVector;
    mg_functions->DestroyVector = DestroyVector;
    mg_functions->MatvecCreate = MatvecCreate;
    mg_functions->Matvec = Matvec;
    mg_functions->MatvecDestroy = MatvecDestroy;
    mg_functions->InnerProd = InnerProd;
    mg_functions->CopyVector = CopyVector;
    mg_functions->ClearVector = ClearVector;
    mg_functions->ScaleVector = ScaleVector;
    mg_functions->Axpy = Axpy;

    mg_functions->DireSolver = DireSolver;
    mg_functions->PreSmooth = PreSmooth;
    mg_functions->PostSmooth = PostSmooth;
    mg_functions->PreSmoothInAux = PreSmoothInAux;
    mg_functions->PostSmoothInAux = PostSmoothInAux;

   return mg_functions;
}

void *pase_MGCreate( pase_MGFunctions *mg_functions)
{
    pase_MGData *mg_data;

    mg_data = hypre_CTAllocF(pase_MGData, 1, mg_functions);

    mg_data->functions = mg_functions;

    /* set defaults */
    mg_data->block_size = 0;
    mg_data->pre_iter = 1;
    mg_data->post_iter = 1;
    mg_data->max_level = -1;
    mg_data->cur_level = -1;
    //mg_data->owns_uH = 1;
    mg_data->A = NULL;
    mg_data->M = NULL;
    mg_data->P = NULL;
    mg_data->Ap = NULL;
    mg_data->Mp = NULL;
    mg_data->u = NULL;

    return (void*)mg_data;
}

PASE_Int PASE_ParCSRMGCreate( PASE_Solver* solver)
{

   //pase_MGData *data = (pase_MGData*)solver;
   if(!solver)
   {
       hypre_error_in_arg(2);
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
	   pase_ParCSRMGDireSolver,
	   pase_ParCSRMGSmootherCG,
	   pase_ParCSRMGSmootherCG,
	   pase_ParCSRMGInAuxSmootherCG,
	   pase_ParCSRMGInAuxSmootherCG);
   *solver = ((HYPRE_Solver)pase_MGCreate( mg_functions));

   return hypre_error_flag;
}

PASE_Int PASE_MGAddLevel( PASE_Solver solver, 
	                 HYPRE_ParCSRMatrix* A, 
			 HYPRE_ParCSRMatrix* M, 
			 HYPRE_ParCSRMatrix* P,
			 HYPRE_Int n)
{
    PASE_Int i;
    pase_MGData* data = (pase_MGData*)solver;
    PASE_Int max_level = data->max_level;
    //pase_MGFunctions* functions = data->functions;

    if(A!=NULL)
    {
	HYPRE_ParCSRMatrix* solver_A = (HYPRE_ParCSRMatrix*)data->A;
	HYPRE_ParCSRMatrix* new_A = hypre_CTAlloc(HYPRE_ParCSRMatrix, max_level+1+n);
	for(i=0; i<max_level+1; i++)
	{	
	    new_A[i] = solver_A[i];
	}	
	for(i=0; i<n; i++)
	{
	    new_A[i+max_level+1] = A[i];
	}
	data->A = (void**)new_A;
	free(solver_A);
    }

    if(M!=NULL)
    {
	HYPRE_ParCSRMatrix* solver_M = (HYPRE_ParCSRMatrix*)data->M;
	HYPRE_ParCSRMatrix* new_M = hypre_CTAlloc(HYPRE_ParCSRMatrix, max_level+1+n);
	for(i=0; i<max_level+1; i++)
	{	
	    new_M[i] = solver_M[i];
	}	
	for(i=0; i<n; i++)
	{
	    new_M[i+max_level+1] = M[i];
	}
	data->M = (void**)new_M;
	free(solver_M);
    }
     
    if(data->max_level>=1 && P!=NULL)
    {
	HYPRE_ParCSRMatrix* solver_P = (HYPRE_ParCSRMatrix*)data->P;
	HYPRE_ParCSRMatrix* new_P = hypre_CTAlloc(HYPRE_ParCSRMatrix, max_level+n);
	for(i=0; i<max_level; i++)
	{	
	    new_P[i] = solver_P[i];
	}	
	for(i=0; i<n; i++)
	{
	    new_P[i+max_level] = P[i];
	}
	data->P = (void**)new_P;
       	free(solver_P);
    }
    
    data->max_level += n;
    data->cur_level = data->max_level;
    return 0;
}
PASE_Int PASE_ParCSRMGInit( MPI_Comm comm, PASE_Solver solver, HYPRE_ParVector* u_h, PASE_Int block_size, PASE_Int seed)
{
    PASE_Int i, j;
    PASE_Int N_h, M_h;
    PASE_Int* partitioning;
    pase_MGData* data = (pase_MGData*) solver;
    pase_MGFunctions* functions = data->functions;
    PASE_Int max_level = data->max_level;
    HYPRE_ParCSRMatrix* A = (HYPRE_ParCSRMatrix*) data->A;

#if 1
    pase_ParVector*** U;
    U = hypre_CTAllocF(pase_ParVector**, max_level+1, functions);   
    pase_ParVector** U_h = hypre_CTAllocF(pase_ParVector*, block_size, functions);
    if(u_h!=NULL)
    {
	//data->owns_uH = 0;
	for(i=0; i<block_size; i++)
	{
	    N_h = u_h[i]->global_size;
	    PASE_ParVectorCreate( comm, N_h, block_size, u_h[i], NULL, U_h+i); 	
	}
    }
    else
    {
	//data->owns_uH = 1;
	HYPRE_ParCSRMatrixGetRowPartitioning( A[max_level], &partitioning);
	HYPRE_ParCSRMatrixGetDims( A[max_level], &M_h, &N_h);
	for(i=0; i<block_size; i++)
	{
	    PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, U_h+i); 	
	    hypre_ParVectorSetRandomValues( U_h[i]->b_H, seed);
	    //hypre_SeqVectorSetRandomValues( U_h[i]->aux_h, seed);
	}
	free(partitioning);
    }
    U[max_level] = U_h;
    for(j=0; j<max_level; j++)
    {
	U_h = hypre_CTAllocF(pase_ParVector*, block_size, functions);
	HYPRE_ParCSRMatrixGetRowPartitioning( A[j], &partitioning);
	HYPRE_ParCSRMatrixGetDims( A[j], &M_h, &N_h);
	for(i=0; i<block_size; i++)
	{
	    PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, U_h+i); 	
	    hypre_ParVectorSetConstantValues( U_h[i]->b_H, 0.0);
	    hypre_SeqVectorSetConstantValues( U_h[i]->aux_h, 0.0);
	    HYPRE_Complex* aux_data = U_h[i]->aux_h->data;
	    aux_data[i] = 1.0;
	}
	U[j] = U_h;
	free(partitioning);
    }
    data->u = (void***) U;
#endif

    HYPRE_Complex** eigenvalues;
#if 1
    eigenvalues = hypre_CTAlloc(HYPRE_Complex*, max_level+1);
    for(i=0; i<max_level+1; i++)
    {
	eigenvalues[i] = hypre_CTAlloc(HYPRE_Complex, block_size);
    }
    data->eigenvalues = eigenvalues;
#endif
    data->block_size = block_size;

    return 0;
}
/**
 * @brief 
 *
 * @param solver
 * @param A
 * @param b
 * @param x
 *
 * @return 
 *
 */
PASE_Int PASE_ParCSRMGSolve(PASE_Solver solver)
{
   pase_MGData *data = (pase_MGData*)solver;
   pase_MGFunctions *functions = data->functions;
   PASE_Int	cur_level = data->cur_level;
   PASE_Int	max_level = data->max_level;

   if(cur_level>0 && cur_level<=max_level)
   {
       printf("cur_level = %d, max_level = %d\n", data->cur_level, data->max_level);
       /*前光滑*/
       printf("PreSmoothing..........\n");
       pase_ParCSRMGPreSmooth(solver);
       printf("Creating AuxMatrix..........\n");
       pase_ParCSRMGAuxMatrixCreate(solver);
       
       /*粗空间校正*/
       data->cur_level--;
       PASE_ParCSRMGSolve(solver);
       data->cur_level++;

       /*后光滑*/
       printf("PostCorrecting..........\n");
       pase_ParCSRMGPostCorrection(solver);
       printf("PostSmoothing..........\n");
       pase_ParCSRMGPostSmooth(solver);
   }
   else if(cur_level==0 && max_level>0)
   {
       functions->DireSolver(solver);
   }
   else
   {
       printf("Error: cur_level = %d, max_level = %d\n",data->cur_level, data->max_level);
   }

   return 0;
}

PASE_Int pase_ParCSRMGPreSmooth(PASE_Solver solver)
{
    pase_MGData *data = (pase_MGData*)solver;
    pase_MGFunctions *functions = data->functions;
    PASE_Int cur_level = data->cur_level;
    PASE_Int max_level = data->max_level;
    if(cur_level == max_level)
    {
	functions->PreSmooth(solver);
    }
    else
    {
	functions->PreSmoothInAux(solver);
    }
    return 0;
}

PASE_Int pase_ParCSRMGPostSmooth(PASE_Solver solver)
{
    pase_MGData *data = (pase_MGData*)solver;
    pase_MGFunctions *functions = data->functions;
    PASE_Int cur_level = data->cur_level;
    PASE_Int max_level = data->max_level;
    if(cur_level == max_level)
    {
	functions->PostSmooth(solver);
    }
    else
    {
	functions->PostSmoothInAux(solver);
    }
    return 0;
}

PASE_Int pase_ParCSRMGPostCorrection(PASE_Solver solver)
{
    PASE_Int i, j;
    pase_MGData* data = (pase_MGData*) solver;
    pase_MGFunctions* functions = data->functions;
    PASE_Int cur_level = data->cur_level;
    PASE_Int block_size = data->block_size;
    pase_ParVector*** u = (pase_ParVector***) data->u;
    pase_ParVector** u0 = u[cur_level-1];
    pase_ParVector** u1 = u[cur_level];
    hypre_ParCSRMatrix** P = (hypre_ParCSRMatrix**) data->P;
    hypre_ParCSRMatrix* P0 = P[cur_level-1];

    /* 重新申请一组pase向量 */
    PASE_Int N_h = u1[0]->b_H->global_size;
    PASE_Int* partitioning = u1[0]->b_H->partitioning;
    pase_ParVector** u_new = hypre_CTAllocF(pase_ParVector*, block_size, functions);

    /* u_new->b_H += P*u0->b_H */
    /* u_new += u1*u_0->aux_h */
    PASE_Real* aux_data;
    for(i=0; i<block_size; i++)
    {
	MPI_Comm comm = u1[i]->comm;
	PASE_ParVectorCreate( comm, N_h, block_size, NULL, partitioning, u_new+i); 	
	hypre_ParCSRMatrixMatvec( 1.0 , P0 , u0[i]->b_H , 0.0 , u_new[i]->b_H );
	aux_data = u0[i]->aux_h->data;
	for(j=0; j<block_size; j++)
	{
	    PASE_ParVectorAxpy( aux_data[j], u1[j], u_new[i]);
	}
    }
    u[cur_level] = u_new;

    /* 释放空间 */
    for(i=0; i<block_size; i++)
    {
	PASE_ParVectorDestroy( u1[i]);
    }
    hypre_TFreeF( u1, functions);

    return 0;
}

PASE_Int pase_ParCSRMGAuxMatrixCreate(PASE_Solver solver)
{
    PASE_Int	i, j, k, l; 
    pase_MGData *data = (pase_MGData*)solver;
    PASE_Int cur_level = data->cur_level;
    PASE_Int max_level = data->max_level;
    PASE_Int     block_size = data->block_size;
    PASE_Int     num_nonzeros = block_size*block_size;
    HYPRE_ParCSRMatrix *A = (HYPRE_ParCSRMatrix*)data->A;
    HYPRE_ParCSRMatrix *M = (HYPRE_ParCSRMatrix*)data->M;
    HYPRE_ParCSRMatrix *P = (HYPRE_ParCSRMatrix*)data->P;
    HYPRE_ParCSRMatrix A0 = A[cur_level-1];
    HYPRE_ParCSRMatrix M0 = M[cur_level-1];
    HYPRE_ParCSRMatrix A1 = A[cur_level];
    HYPRE_ParCSRMatrix M1 = M[cur_level];
    HYPRE_ParCSRMatrix P0 = P[cur_level-1];
    pase_ParCSRMatrix **Ap = (pase_ParCSRMatrix**)data->Ap;
    pase_ParCSRMatrix **Mp = (pase_ParCSRMatrix**)data->Mp;
    pase_ParCSRMatrix *Ap0 = Ap[cur_level-1];
    pase_ParCSRMatrix *Mp0 = Mp[cur_level-1];
    pase_ParCSRMatrix *Ap1 = Ap[cur_level];
    pase_ParCSRMatrix *Mp1 = Mp[cur_level];
    pase_ParVector ***u = (pase_ParVector***)data->u;
    pase_ParVector **U = u[cur_level];

    HYPRE_ParVector *u_h = hypre_CTAlloc(HYPRE_ParVector, block_size); 
    for(i=0; i<block_size; i++)
    {
	u_h[i] = U[i]->b_H;
    }

    PASE_Int* partitioning_h = u_h[1]->partitioning;
    MPI_Comm comm = u_h[1]->comm;
    PASE_Int N_h = u_h[1]->global_size;
    HYPRE_ParVector workspace_h = hypre_ParVectorCreate(comm, N_h, partitioning_h); 

    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, A0, P0, A1, u_h, &Ap0, NULL, workspace_h);  
    PASE_ParCSRMatrixCreate( MPI_COMM_WORLD, block_size, M0, P0, M1, u_h, &Mp0, NULL, workspace_h);  

    /*还需修正aux_hH,aux_Hh,aux_hh*/
    if(cur_level < max_level)
    {
       hypre_ParVector** b1 = Ap1->aux_Hh;
       hypre_ParVector** b0 = Ap0->aux_Hh;
       hypre_ParVector** beta1 = Mp1->aux_Hh;
       hypre_ParVector** beta0 = Mp0->aux_Hh;
       HYPRE_Real* va_a1 = Ap1->aux_hh->data;
       HYPRE_Real* va_a0 = Ap0->aux_hh->data;
       HYPRE_Real* va_alpha1 = Mp1->aux_hh->data;
       HYPRE_Real* va_alpha0 = Mp0->aux_hh->data;
       /* aux_hH[cur_level-1] += P^T*aux_hH[cur_level]*Z */
       for(i=0; i<block_size; i++)
       {
	   for(j=0; j<block_size; j++)
	   {
	       hypre_ParCSRMatrixMatvecT(U[j]->aux_h->data[i],P0,b1[j],1.0,b0[i]);
	       hypre_ParCSRMatrixMatvecT(U[j]->aux_h->data[i],P0,beta1[j],1.0,beta0[i]);
	   }
       }
       /* aux_hh[cur_level-1] += 2*U^T*aux_hH[cur_level]*Z + Z^T*aux_hh[cur_level]*Z */
       PASE_Real *uTb, *uTbeta;
       uTb = (PASE_Real*)malloc(num_nonzeros*sizeof(PASE_Real));
       uTbeta = (PASE_Real*)malloc(num_nonzeros*sizeof(PASE_Real));
       for(i=0; i<=block_size; i++)
       {
	   for(j=0; j<block_size; j++)
	   {
	       uTb[j+i*block_size] = hypre_ParVectorInnerProd(U[i]->b_H, b1[j]);
	       uTbeta[j+i*block_size] = hypre_ParVectorInnerProd(U[i]->b_H, beta1[j]);
	   }
       }
       for(i=0; i<block_size; i++)
       {
	   for(j=0; j<block_size; j++)
	   {
	      for(k=0; k<block_size; k++)
	      {
		  /* '2*'这个乘法操作可以优化 */
		  va_a0[i*block_size+j] += 2*U[i]->aux_h->data[k]*uTb[j*block_size+k];
		  va_alpha0[i*block_size+j] += 2*U[i]->aux_h->data[k]*uTbeta[j*block_size+k];
		  for(l=0; l<block_size; l++)
		  {
		      va_a0[i*block_size+j] += U[i]->aux_h->data[l]*va_a1[l*block_size+k]*U[j]->aux_h->data[k];
		      va_alpha0[i*block_size+j] += U[i]->aux_h->data[l]*va_alpha1[l*block_size+k]*U[j]->aux_h->data[k];
		  }
	      } 

	   }
       }

    }
    return 0;
}

PASE_Int pase_ParCSRMGDireSolver(PASE_Solver solver)
{
    HYPRE_Solver lobpcg_solver;
    int maxIterations = 100; /* maximum number of iterations */
    int pcgMode = 1;         /* use rhs as initial guess for inner pcg iterations */
    int verbosity = 0;       /* print iterations info */
    double tol = 1.e-8;     /* absolute tolerance (all eigenvalues) */
    int lobpcgSeed = 775;

    PASE_Int i;
    pase_MGData *mg_solver = (pase_MGData *)solver;
    PASE_Int block_size = mg_solver->block_size;
    PASE_Int cur_level = mg_solver->cur_level;
    pase_ParCSRMatrix** Ap = (pase_ParCSRMatrix**)mg_solver->Ap;
    pase_ParCSRMatrix** Mp = (pase_ParCSRMatrix**)mg_solver->Mp;
    pase_ParCSRMatrix* Ap0 = Ap[cur_level];
    pase_ParVector*** u = (pase_ParVector***) mg_solver->u;
    pase_ParVector** u0 = u[cur_level];

    HYPRE_Complex* eigenvalues = mg_solver->eigenvalues[cur_level];
    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter,1);
    PASE_ParCSRSetupInterpreter(interpreter_Hh);
    PASE_ParCSRSetupMatvec(&matvec_fn_Hh);
    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, u0[0]);
    mv_MultiVectorSetRandom (eigenvectors_Hh, lobpcgSeed);
     
    HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);

    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
    HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
    HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

    PASE_LOBPCGSetup (lobpcg_solver, Ap0, NULL, NULL);
    if( Mp!=NULL)
    {
	pase_ParCSRMatrix* Mp0 = Mp[cur_level];
	PASE_LOBPCGSetupB(lobpcg_solver, Mp0, NULL);
    }
    HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues );

    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
    u[cur_level] = (PASE_ParVector*)(tmp -> vector);
    for(i=0; i<block_size; i++)
    {
	PASE_ParVectorDestroy( u0[i] );
    }
    hypre_TFreeF( u0, mg_solver->functions);

    HYPRE_LOBPCGDestroy(lobpcg_solver);
    return 0;
}

PASE_Int pase_ParCSRMGSmootherCG(PASE_Solver solver)
{
    printf("Using CG as the smoother of MG method!\n");
    pase_MGData *mg_solver = (pase_MGData *)solver;
    HYPRE_Solver cg_solver;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &cg_solver);
    /* Set some parameters */
    HYPRE_PCGSetMaxIter(cg_solver, mg_solver->pre_iter); /* max iterations */
    //HYPRE_PCGSetTol(cg_solver, 1.0e-07); 
    HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel(cg_solver, 2); 
    HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */
           
    /* Setup and solve */
    HYPRE_Int cur_level = mg_solver->cur_level;
    HYPRE_Int block_size = mg_solver->block_size;
    HYPRE_Int i;
    HYPRE_ParCSRMatrix* solver_A = (HYPRE_ParCSRMatrix*)mg_solver->A;
    HYPRE_ParCSRMatrix* solver_M = (HYPRE_ParCSRMatrix*)mg_solver->M;
    HYPRE_ParCSRMatrix A = solver_A[cur_level];
    HYPRE_ParCSRMatrix M = NULL;
    if(A == NULL)
	printf("Error: A is NULL!\n");

    if(solver_M!=NULL)
    {
	M = solver_M[cur_level];
    }
    pase_ParVector ***solver_u = (pase_ParVector***)mg_solver->u;
    pase_ParVector **u = solver_u[cur_level];
    hypre_ParVector *x, *y, *b_H; 
    PASE_Int N_h = u[0]->b_H->global_size;
    printf("N_h = %d\n", N_h);
    //PASE_Int M_h;
    //HYPRE_ParCSRMatrixGetDims( A, &M_h, &N_h);
    //printf("M_h = %d, N_h = %d\n", M_h, N_h);
    PASE_Int* partitioning = u[0]->b_H->partitioning;
    MPI_Comm comm = u[0]->b_H->comm;
    for(i=0; i<block_size; i++)
    {
	b_H = u[i]->b_H;
	x = hypre_ParVectorCreate(comm, N_h, partitioning);
	y = hypre_ParVectorCreate(comm, N_h, partitioning);
	//hypre_ParVectorInitialize(x);
	//hypre_ParVectorSetPartitioningOwner(x,0);
	////hypre_ParVectorCopy( b_H, x);
	if(M!=NULL)
	{
	    hypre_ParCSRMatrixMatvec( 1.0, M, b_H, 0.0, b_H);
	}
	if(u==NULL)
	{
	    printf("u is NULL!!\n");
	}
	hypre_PCGSetup(cg_solver, A, y, x);
	printf("Error: Here !!!!!!!!!!!!!!!!!\n");
	hypre_PCGSolve(cg_solver, A, b_H, x);
	u[i]->aux_h = NULL;
    }

    PASE_ParCSRPCGDestroy(cg_solver);

    return 0;
}

PASE_Int pase_ParCSRMGInAuxSmootherCG(PASE_Solver solver)
{
    pase_MGData *mg_solver = (pase_MGData *)solver;
    HYPRE_Solver cg_solver;
    PASE_ParCSRPCGCreate(MPI_COMM_WORLD, &cg_solver);
    /* Set some parameters (See Reference Manual for more paramet    ers) */
    HYPRE_PCGSetMaxIter(cg_solver, mg_solver->pre_iter); /* max iterations */
    HYPRE_PCGSetTol(cg_solver, 1e-7); /* conv. tolerance */
    HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel(cg_solver, 2); /* prints out the iteratio    n info */
    HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */
 
    /* Now setup and solve! */
    HYPRE_Int cur_level = mg_solver->cur_level;
    HYPRE_Int block_size = mg_solver->block_size;
    HYPRE_Int i;
    pase_ParCSRMatrix **Ap = (pase_ParCSRMatrix**)mg_solver->Ap;
    pase_ParCSRMatrix **Mp = (pase_ParCSRMatrix**)mg_solver->Mp;
    pase_ParCSRMatrix *A = Ap[cur_level];
    pase_ParCSRMatrix *M = NULL;
    if(Mp!=NULL)
    {
	M = Mp[cur_level];
    }
    pase_ParVector ***solver_u = (pase_ParVector ***)mg_solver->u;
    pase_ParVector **u = solver_u[cur_level];
    for(i=0; i<block_size; i++)
    {
	if(M!=NULL)
	{
	    PASE_ParCSRMatrixMatvec( 1.0, M, u[i], 0.0, u[i]);
	}
	hypre_PCGSetup(cg_solver, A, u[i], u[i]);
	hypre_PCGSolve(cg_solver, A, u[i], u[i]);
    }

    PASE_ParCSRPCGDestroy(cg_solver);

    return 0;
}

PASE_Int PASE_ParCSRMGDestroy(PASE_Solver solver)
{
    pase_MGData* data = (pase_MGData*)solver;
    HYPRE_Int i, j, max_level, block_size;
    max_level = data->max_level;
    block_size = data->block_size;

    if(data)
    {
	pase_MGFunctions* functions = data->functions;
	if(data->A!=NULL)
	{
	    hypre_TFreeF( data->A, functions);
	    data->A = NULL;
	}
	if(data->M!=NULL)
	{
	    hypre_TFreeF( data->M, functions);
	    data->M = NULL;
	}
	if(data->P!=NULL)
	{
	    hypre_TFreeF( data->P, functions);
	    data->P = NULL;
	}
	if(data->Ap!=NULL)
	{
	    PASE_ParCSRMatrix* Ap = (PASE_ParCSRMatrix*)data->Ap;
	    for(i=0; i<max_level+1; i++)
	    {
		PASE_ParCSRMatrixDestroy( Ap[i] );
	    }
	    hypre_TFreeF( data->Ap, functions);
	    data->Ap = NULL;
	}
	if(data->Mp!=NULL)
	{
	    PASE_ParCSRMatrix* Mp = (PASE_ParCSRMatrix*)data->Mp;
	    for(i=0; i<max_level+1; i++)
	    {
		PASE_ParCSRMatrixDestroy( Mp[i] );
	    }
	    hypre_TFreeF( data->Mp, functions);
	    data->Mp = NULL;
	}
	if(data->u!=NULL)
	{
	    PASE_ParVector** u = (PASE_ParVector**)data->u;
	    for(i=0; i<max_level+1; i++)
	    {
		PASE_ParVector* u_i = u[i];
		for(j=0; j<block_size; j++)
		{
		    PASE_ParVectorDestroy( u_i[j] );
		}
		hypre_TFreeF( u_i, functions);
		data->u[i] = NULL;
	    }
	    hypre_TFreeF( data->u, functions);
	    data->u = NULL;
	}
	if(data->eigenvalues!=NULL)
	{
	    for(i=0; i<max_level+1; i++)
	    {
		hypre_TFreeF(data->eigenvalues[i], functions);
	    }
	    hypre_TFreeF(data->eigenvalues, functions);
	    data->eigenvalues = NULL;
	}
	hypre_TFreeF(data, functions);
	hypre_TFreeF(functions, functions);
    }
    return hypre_error_flag;
}
