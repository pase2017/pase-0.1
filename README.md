[TOC]

# 1. PASE or PASES

把 `PASE` 改为 `PASES` 会不会更明确我们软件包的意图? 最后的 `"S"` 可以有两种含义: `Solver` 和 `Smoother`, 即从 ` Parallel Auxillary Space Eigen-solver` 改为 `Parallel Auxillary Space Eigen Solver/Smoother`.

# 2. PASE_PARAMETER

``` c
/* 基本数据类型的封装 */
typedef int       PASE_INT;
typedef double    PASE_DOUBLE;
typedef double    PASE_REAL;
typedef PASE_REAL PASE_SCALAR;

/* 枚举类型只是为了向开发人员和用户说明 PASE_COARSEN_TYPE 的取值范围.
 * 为了外部接口实现方便, 实际并不使用该类型, 
 * 而是使用 PASE_INT (即 int) 类型的变量指定粗化策略, 如 PASE_INT coarse_type.
 * 其余枚举类型的意图也是如此.
 */
typedef enum { CLJP = 1, FALGOUT = 2, PMHIS = 3 } PASE_COARSEN_TYPE;
typedef enum { HYPRE = 1 } EXTERNAL_PACKAGE;

typedef struct PASE_PARAMETER_ {
  /* amg_parameter */
  /**
   * 列出 amg setup phase 的相关参数, 
   * 如 强影响因子, 粗话策略, 插值类型 等等.
   */
  PASE_INT coarse_type; // CLJP, falgout, ...
  
  /* 外部软件包, 用于提供实际的矩阵和向量的数据结构与操作 */
  PASE_INT external_package; // 1: HYPRE
  
   
  /* linear smoother/solver */
  /**
   * 1. 线性解法器/光滑子类型
   *    CG, GS, SOR, GMRES, AMG 等等.
   *    非负数表明使用默认求解方法, 负数表明用户自行提供方法.
   * 2. 求解参数
   *    最大光滑次数, 收敛阈值 等等.
   */
  PASE_INT linear_smoother;      // 光滑.
  PASE_INT linear_solver;        // 求解. 可用于最粗层网格的求解.
  PASE_INT linear_smoother_pre;  // 前光滑. 如无特别指定, 则默认与 linear_smoother 相同
  PASE_INT linear_smoother_post; // 后光滑. 如无特别指定, 则默认与 linear_smoother 相同
  
  /* eigen smoother/solver */
  /**
   * 特征值解法器/光滑子类型
   * Arnoldi, Lanczos, KrylovSchur, LOBPCG 等等.
   *    非负数表明使用默认求解方法, 负数表明用户自行提供方法.
   */
  PASE_INT eigen_smoother;      // 光滑.
  PASE_INT eigen_solver;        // 求解.
  PASE_INT eigen_smoother_pre;  // 前光滑. 如无特别指定, 则默认与 eigen_smoother 相同
  PASE_INT eigen_smoother_post; // 后光滑. 如无特别指定, 则默认与 eigen_smoother 相同
} PASE_PARAMETER;

/* 参数设置与获取 */
void PASE_Parameter_SetLinearSmoother(JPINT linear_smoother);
PASE_INT PASE_Paramater_GetLinearSmoother(void);
```

# 3. PASE_MULTIGRID

``` c
typedef struct PASE_MATRIX_ {
  void *matrix_data;   
  PASE_INT nrow; // 行数
  PASE_INT ncol; // 列数  
} PASE_MATRIX;

/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX *CreatePaseMatrix(void *mat_data, PASE_PARAMETER param);
/* 拷贝 PASE_MATRIX 变量 */
PASE_MATRIX *CopyPaseMatrix(PASE_MATRIX *pase_mat);

/* 销毁 PASE_MATRIX 变量 */
void DestroyPaseMatrix(PASE_MATRIX *A);

typedef struct PASE_VECTOR_ {
  void *vector_data;
  PASE_INT nrow; // 行数. PASE_VECTOR 均为列向量
} PASE_VECTOR;

typedef struct PASE_MULTI_VECTOR_ {
  PASE_INT size; // 多重向量个数
  PASE_VECTOR *vector; // 指向 size 个 PASE_VECTOR 变量
  // PASE_VECTOR **vector; // 指向 size 个 PASE_VECTOR * 变量
} PASE_MULTI_VECTOR;

/*
 * aux matrix = [mat   vec    ]
 *              [vecT  block2d]
 */
typedef struct PASE_AUX_MATRIX_ {
  PASE_MATRIX *mat;
  PASE_VECTOR *vec;
  PASE_VECTOR *vecT;
  PASE_SCALAR *block2d;
} PASE_AUX_MATRIX;

/*
 * aux vector = [vec    ]
 *              [block1d]
 */
typedef struct PASE_AUX_VECTOR_ {
  PASE_VECTOR *vec;
  PASE_SCALAR *block1d;
} PASE_AUX_VECTOR;

typedef struct PASE_MULTI_AUX_VECTOR_ {
  PASE_INT size;
  PASE_AUX_VECTOR *aux_vector;
  // PASE_AUX_VECTOR **aux_vector;
} PASE_MULTI_AUX_VECTOR;

typedef struct PASE_MULTIGRID_ {
  PASE_INT max_level;
  PASE_INT actual_level;
  
  PASE_MATRIX **A; // A_0 (细) ---->> A_n (粗)
  PASE_MATRIX **B; // B_0 (细) ---->> B_n (粗)
  
  PASE_MATRIX **P; // 相邻网格层扩张算子 I_{k+1}^{k}, k = 0,\cdots,n-1
  PASE_MATRIX **R; // 相邻网格层限制算子 I_{k}^{k+1}, k = 0,\cdots,n-1
  
  PASE_MATRIX **LP; // long term prolongation, 某层到最细层的扩张算子
  PASE_MATRIX **LR; // long term restriction, 最细层到某层的限制算子
  
  PASE_AUX_MATRIX **aux_A; // 辅助空间矩阵
  PASE_AUX_MATRIX **aux_B; // 辅助空间矩阵
} PASE_MULTIGRID;

PASE_MULTIGRID *CreatePaseMultigrid(PASE_MATRIX *A, PASE_MATRIX *B, PASE_PARAMETER param);
void DestroyPaseMultigrid(PASE_MULTIGRID *pase_multigrid);

/* 不同层向量之间的转移均通过如下函数进行 */
void PASE_Multigrid_VectorTransfer(PASE_MULTIGRID mg, 
                                   PASE_INT src_level, PASE_VECTOR src_vec,
                                   PASE_INT des_level, PASE_VECTOR des_vec)
{
  /**
   * if(NULL != mu.LP) {
   *   // 由 LP/LQ 直接做矩阵向量乘法即可得到目标向量
   * } else {
   *   // 由 P/Q 逐层做矩阵向量乘法得到目标向量
   * }
   */
}
```
# 4. PASE_EigenSolver

``` c
/* 特征值求解器 */
void PASE_EigenSolver(PASE_MATRIX *A, PASE_MATRIX *B, 
                      PASE_VECTOR *eval, PASE_MULTI_VECTOR *evec, 
                      PASE_PARAMETER param);
```

