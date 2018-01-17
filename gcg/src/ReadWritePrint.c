/*************************************************************************
	> File Name: SlepcMatrixVec.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:34:11 2018
 ************************************************************************/

#include "ReadWritePrint.h"

PetscErrorCode ReadPetscMatrixBinary(Mat *A, const char *filename)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Getting matrix...\n"); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A); CHKERRQ(ierr);
  ierr = MatLoad(*A, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Getting matrix... Done\n"); CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode ReadGenaralMatrixBinary(Mat *A, Mat *B, PetscInt ifBI)
{
    PetscErrorCode ierr;
    char filenameA[PETSC_MAX_PATH_LEN] = "fileinput1";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat1", filenameA, sizeof(filenameA), NULL); CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "filenameA: %s\n", filenameA);
    ierr = ReadPetscMatrixBinary(A, filenameA); CHKERRQ(ierr);
    if(ifBI == 1)
    {
        PetscInt i,n,Istart,Iend;
        ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
		PetscPrintf(PETSC_COMM_WORLD, "matrix B = I, n = %d\n", n);
        ierr = MatCreate(PETSC_COMM_WORLD,B);CHKERRQ(ierr);
        ierr = MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
        ierr = MatSetFromOptions(*B);CHKERRQ(ierr);
        ierr = MatSetUp(*B);CHKERRQ(ierr);
        
        ierr = MatGetOwnershipRange(*B,&Istart,&Iend);CHKERRQ(ierr);
        for (i=Istart;i<Iend;i++) {
            ierr = MatSetValue(*B,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
        }

        ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    else
    {
        char filenameB[PETSC_MAX_PATH_LEN] = "fileinput2";
        ierr = PetscOptionsGetString(NULL, NULL, "-mat2", filenameB, sizeof(filenameB), NULL); CHKERRQ(ierr);
		PetscPrintf(PETSC_COMM_WORLD, "filenameB: %s\n", filenameB);
        ierr = ReadPetscMatrixBinary(B, filenameB); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode GenerateFDMatrix(Mat *A, Mat *B)
{
    PetscInt       n=10, i, Istart, Iend;
    PetscErrorCode ierr;
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "FD matrix, n = %d\n", n);

    ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
    ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
    ierr = MatSetUp(*A);CHKERRQ(ierr);

    //ierr = ReadMatrix("A_3.txt", A);
    ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
        if (i>0) { ierr = MatSetValue(*A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
        if (i<n-1) { ierr = MatSetValue(*A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
        if (i<n-2) { ierr = MatSetValue(*A,i,i+2,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
        if (i>1) { ierr = MatSetValue(*A,i,i-2,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
        ierr = MatSetValue(*A,i,i,4.0,INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,B);CHKERRQ(ierr);
    ierr = MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*B);CHKERRQ(ierr);
    ierr = MatSetUp(*B);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(*B,&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
        ierr = MatSetValue(*B,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode PrintVec(Vec *V, PetscInt n_vec, PetscInt start, PetscInt end)
{
    PetscErrorCode ierr;
    PetscInt       i, j;
    const PetscScalar    *y;
    for( i=0; i<n_vec; i++ )
    {
        ierr = VecGetArrayRead(V[i], &y);CHKERRQ(ierr);
        for( j=start; j<end; j++ )
            PetscPrintf(PETSC_COMM_WORLD, "V[%d][%d] = %18.15lf\n", i, j, y[j]);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ReadCSRMatrix(Mat *A, Mat *B)
{
    PetscErrorCode ierr;
    char filenameA[PETSC_MAX_PATH_LEN] = "fileinput1";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat1", filenameA, sizeof(filenameA), NULL); CHKERRQ(ierr);
	//PetscPrintf(PETSC_COMM_WORLD, "filenameA: %s\n", filenameA);
    ierr = ReadMatrix(filenameA, A); CHKERRQ(ierr);
    char filenameB[PETSC_MAX_PATH_LEN] = "fileinput2";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat2", filenameB, sizeof(filenameB), NULL); CHKERRQ(ierr);
	//PetscPrintf(PETSC_COMM_WORLD, "filenameB: %s\n", filenameB);
    ierr = ReadMatrix(filenameB, B); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ReadMatrix(const char *filename, Mat *A)
{
    PetscErrorCode ierr;
    PetscInt       Istart, Iend, i, j, error;
    //PetscReal      aaa;
    FILE *file = fopen(filename,"r");
    if(!file)
    {
        PetscPrintf(PETSC_COMM_WORLD, "\ncannot open %s!\n", filename);
        exit(0);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Read matrix: %s\n", filename); 

    PetscInt nrow, ncol, nnz;

    error = fscanf(file, "%d\n", &nrow);
    error = fscanf(file, "%d\n", &ncol);
    error = fscanf(file, "%d\n", &nnz);

    ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
    ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,nrow,ncol);CHKERRQ(ierr);
    ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
    ierr = MatSetUp(*A);CHKERRQ(ierr);

    PetscInt *ia = malloc( (nrow+1)*sizeof(PetscInt) );
    PetscInt *ja = (PetscInt *)malloc( nnz*sizeof(PetscInt) );
    PetscReal *aa = (PetscReal *)malloc( nnz*sizeof(PetscReal) );


    for(i=0;i<nrow+1;i++)
    {
        error = fscanf(file, "%d\n", ia+i);
    }
    for(i=0;i<nnz;i++)
    {
        error = fscanf(file, "%d\n", ja+i);
    }
    for(i=0;i<nnz;i++)
    {
        error = fscanf(file, "%lf\n", aa+i);
    }
    if(error == 0)
        PetscPrintf(PETSC_COMM_WORLD, "in ReadMatrix, Error!\n");
    fclose(file);
    /*
    for(i=0; i<10; i++)
        PetscPrintf(PETSC_COMM_WORLD, "ia[%d] = %d\n", i, ia[i]);
    for(i=0; i<10; i++)
        PetscPrintf(PETSC_COMM_WORLD, "ja[%d] = %d\n", i, ja[i]);
    for(i=0; i<50; i++)
        PetscPrintf(PETSC_COMM_WORLD, "aa[%d] = %lf\n", i, aa[i]);
        */

    ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD, "Istart: %d, Iend: %d\n", Istart, Iend);
    for (i=Istart;i<Iend;i++) {
        for( j=ia[i]; j<ia[i+1]; j++ )
        {
            ierr = MatSetValue(*A,i,ja[j],aa[j],INSERT_VALUES);CHKERRQ(ierr);
        }
        //MatSetValues(*A, 1, &i, ia[i+1]-ia[i], ja+ia[i], aa+ia[i], INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /*
    for (i=0;i<10;i++) {
        for( j=ia[i]; j<ia[i+1]; j++ )
        {
            ierr = MatGetValue(*A, i, ja[j], &aaa);
            PetscPrintf(PETSC_COMM_WORLD, "i: %d, j: %d, aaa = %lf\n", i, ja[j], aaa);
        }
    }
    */
    free(ia);  ia = NULL;
    free(ja);  ja = NULL;
    free(aa);  aa = NULL;

    PetscFunctionReturn(0);
}

PetscErrorCode ReadVec(Vec *V, PetscInt nev)
{
    PetscErrorCode ierr;
    char filename[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-init_vec", filename, sizeof(filename), NULL); CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "init vec: filename: %s\n", filename);
    ierr = ReadVecformfile(filename, V, nev);
    PetscFunctionReturn(0);
}
PetscErrorCode ReadVecformfile(char *filename, Vec *V, PetscInt nev)
{
    FILE *file = fopen(filename,"r");
    if(!file)
    {
        PetscPrintf(PETSC_COMM_WORLD, "\ncannot open %s!\n", filename);
        exit(0);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Reading vector...\n"); 

    PetscInt       i, j, error, nrows, start, end;
    PetscErrorCode ierr;
    PetscReal **aa = (PetscReal **)malloc( nev*sizeof(PetscReal*) );
    nrows = V[0]->map->N;
    for( i=0; i<nev; i++ )
        aa[i] = (PetscReal *)calloc(nrows, sizeof(PetscReal));
    for( i=0; i<nev; i++ )
    {
        for( j=0; j<nrows; j++ )
        {
            error = fscanf(file, "%lf\n", aa[i]+j);
            if(error == 0)
                PetscPrintf(PETSC_COMM_WORLD, "in ReadVec, aa[%d][%d] = %18.15lf\n", i, j, aa[i][j]);
        }
    }
    fclose(file);

    PetscMPIInt    rank, np, process;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &np);CHKERRQ(ierr);
    PetscInt *ix = calloc(nrows, sizeof(PetscInt));
    for( i=0; i<nrows; i++ )
        ix[i] = i;
    for( i=0; i<nev; i++ )
    {
        for( process=0; process<np; process++ )
        {
            if(rank == process)
            {
                start = V[0]->map->rstart;
                end   = V[0]->map->rend;
	            //PetscPrintf(PETSC_COMM_WORLD, "\nvec: %d, rank %d, np: %d, start: %d, end: %d\n", i, rank, np, start, end);
	            //printf("\nvec: %d, rank %d, np: %d, start: %d, end: %d\n", i, rank, np, start, end);
                for( j=start; j<end; j++ )
                    ierr = VecSetValues(V[i], end-start, ix+start, aa[i]+start, INSERT_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(V[i]);
        ierr = VecAssemblyEnd(V[i]);
    }

    for( i=0; i<nev; i++ )
    {
        free(aa[i]);  aa[i] = NULL;
    }
    free(aa);  aa = NULL;
    free(ix);  ix = NULL;

    PetscFunctionReturn(0);
}

PetscErrorCode WriteVec(Vec *V, PetscInt nev)
{
    PetscErrorCode ierr;
    char filename[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-output_vec", filename, sizeof(filename), NULL); CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "output vec: filename: %s\n", filename);
    ierr = WriteVectofile(filename, V, nev);
    PetscFunctionReturn(0);
}
PetscErrorCode WriteVectofile(char *filename, Vec *V, PetscInt n_vec)
{
	FILE           *file;
    PetscErrorCode ierr;
    PetscInt       i, j, nrows;
    PetscScalar    *y;
	file = fopen(filename,"w");
	if(!file)
	{
		PetscPrintf(PETSC_COMM_WORLD, "\ncannot open the %s!\n", filename);
		exit(0);
	}
    fclose(file);

	PetscPrintf(PETSC_COMM_WORLD, "\nOutput vec to %s!\n", filename);
    PetscMPIInt    rank, np, process;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &np);CHKERRQ(ierr);
    for( i=0; i<n_vec; i++ )
    {
        for( process=0; process<np; process++ )
        {
            ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
            if(rank == process)
            {
                nrows = V[i]->map->n;
	            printf("\nvec: %d, rank: %d, nrows: %d\n", i, rank, nrows);
                ierr  = VecGetArray(V[i], &y);CHKERRQ(ierr);

				file = fopen(filename,"a+");
				if(!file)
				{
					PetscPrintf(PETSC_COMM_WORLD, "\ncannot open the %s!\n", filename);
					exit(0);
				}

                //for( j=start; j<end; j++ )
                for( j=0; j<nrows; j++ )
                    fprintf(file, "%18.12lf\n", y[j]);
                fclose(file);
            }
            ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(V[i], &y);
    }
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD, "\nOutput vec to %s Done!\n", filename);
    PetscFunctionReturn(0);
}
