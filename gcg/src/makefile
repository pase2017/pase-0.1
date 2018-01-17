include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

exgcg: exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg.exe exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${SLEPC_EPS_LIB}
	${RM} exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_petscbin: exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_petscbin.exe exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${SLEPC_EPS_LIB}
	${RM} exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

extest: extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/extest.exe extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${SLEPC_EPS_LIB}
	${RM} extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_fd: exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_fd.exe exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${SLEPC_EPS_LIB}
	${RM} exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

cleanall:
	rm -f *.exe ../bin/*.exe
