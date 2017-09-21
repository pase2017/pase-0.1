/*
 * =====================================================================================
 *
 *       Filename:  pase_lobpcg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017年09月14日 13时35分26秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_lobpcg.h"

PASE_Int
PASE_LOBPCGSetup(PASE_Solver solver,  PASE_ParCSRMatrix A, PASE_ParVector b, PASE_ParVector x)
{
   return( hypre_LOBPCGSetup( solver,  A,  b,  x ) );
}
PASE_Int 
PASE_LOBPCGSetupB(PASE_Solver solver, PASE_ParCSRMatrix B, PASE_ParVector x)
{
   return( hypre_LOBPCGSetupB( solver,  B,  x ) );
}
