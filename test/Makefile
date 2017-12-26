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
HYPRE_DIR = ../../../hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include -I..
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
LIBS      = ../libpase.a -L$(HYPRE_DIR)/lib -lHYPRE -lm 
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
# List of all programs to be compiled
########################################################################
ALLPROGS = serPASE_ver01 mv cg lobpcg parPASE_ver01 parPASE_ver02

all: $(ALLPROGS)

default: all

########################################################################
# Example 1 (Multi-Correction)
########################################################################
serPASE_ver01: serPASE_ver01.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 2 (Matrix Vector)
########################################################################
mv: mv.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 3 (CG)
########################################################################
cg: cg.o
	$(CC) -o $@ $^ $(LFLAGS)


########################################################################
# Example 4 (LOBPCG)
########################################################################
lobpcg: lobpcg.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5 (Multi-Correction)
########################################################################
parPASE_ver01: parPASE_ver01.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5 (Multi-Correction)
########################################################################
parPASE_ver02: parPASE_ver02.o
	$(CC) -o $@ $^ $(LFLAGS)


########################################################################
# Example 5 (Multi-Correction)
########################################################################
parPASE_ver06: parPASE_ver06.o
	$(CC) -o $@ $^ $(LFLAGS)






########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o) cg lobpcg mv parPASE_ver01 parPASE_ver02 parPASE_ver06 serPASE_ver01
	rm -f *\.[0-9] *\.aux
