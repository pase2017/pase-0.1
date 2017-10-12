#ifndef __PASE_AUX_VECTOR_H__
#define __PASE_AUX_VECTOR_H__

/*
 * aux vector = [vec    ]
 *              [block1d]
 */
typedef struct PASE_AUX_VECTOR_PRIVATE_ {
  PASE_VECTOR *vec;
  PASE_SCALAR *block1d;
  PASE_VECTOR_OPERATOR ops;
} PASE_AUX_VECTOR_PRIVATE;
typedef PASE_AUX_VECTOR_PRIVATE * PASE_AUX_VECTOR;

typedef struct PASE_MULTI_AUX_VECTOR_PRIVATE_ {
  PASE_INT size;
  //PASE_AUX_VECTOR *aux_vector;
  PASE_AUX_VECTOR **aux_vector;
} PASE_MULTI_AUX_VECTOR_PRIVATE;
typedef PASE_MULTI_AUX_VECTOR_PRIVATE * PASE_MULTI_AUX_VECTOR;

//PASE_AUX_VECTOR PASE_Create_aux_vector(PASE_VECTOR x,PASE_PARAMETER param);
void PASE_Destroy_aux_vector(PASE_VECTOR aux_x);

#endif
