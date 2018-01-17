########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = mpicxx
#CC        = mpiicc
#F77       = mpiifort
#CXX       = mpiicpc
#F90       = mpif90
HYPRE_DIR = ../../hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)
FOPTS     = -g
FINCLUDES = $(CINCLUDES)
FFLAGS    = $(FOPTS) $(FINCLUDES)
CXXOPTS   = $(COPTS) -Wno-deprecated
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS   = $(CDEFS)
IFLAGS_BXX = 
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
IF90FLAGS = 
F90FLAGS = $(FFLAGS) $(IF90FLAGS)


LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++
LFLAGS_B =\
 -L${HYPRE_DIR}/lib\
 -lbHYPREClient-C\
 -lbHYPREClient-CX\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2
LFLAGS77 = $(LFLAGS)
LFLAGS90 =

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c .f .cxx .f90

.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

########################################################################
# List of all source
########################################################################
LIB = libpase.a
OBJS = pase_mv.o pase_int.o pase_pcg.o pase_lobpcg.o pase_mg.o pase_es.o pase_ls.o

############################################################
RM = /bin/rm
RMFLAGS = -f
ARCH = ar
ARCHFLAGS = -rcu
RANLIB = ranlib

########################################################################
lib:	$(LIB)


$(LIB): $(OBJS)
	   @echo ''
	   @echo 'Creating library $(LIB)'
	   @$(ARCH) $(ARCHFLAGS) $(LIB)  $(OBJS)
	   @$(RANLIB) $(LIB)


clean:
	$(RM) $(RMFLAGS) $(OBJS)
