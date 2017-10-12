#ifndef __PASE_CONFIG_H__
#define __PASE_CONFIG_H__
/* 基本数据类型的封装 */
typedef int       PASE_INT;
typedef double    PASE_DOUBLE;
typedef double    PASE_REAL;

#ifdef COMPLEX_FEILD
typedef complex PASE_SCALAR;
typedef complex PASE_COMPLEX;
#else
typedef double PASE_SCALAR;
#endif

typedef enum { CLJP = 1, FALGOUT = 2, PMHIS = 3 } PASE_COARSEN_TYPE;
typedef enum { HYPRE = 1 } EXTERNAL_PACKAGE;


#endif
