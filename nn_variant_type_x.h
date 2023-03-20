#ifndef NN_VARIANT_TYPE_X_H
#define NN_VARIANT_TYPE_X_H

typedef double num_t_x;

enum { REAL_X, VEC_X, MAT_X, TENSOR_3D_X };

typedef struct {
  int type;
  void *val;
} var_t_x;

typedef struct {
  num_t_x real;
} real_t_x;

typedef struct {
  num_t_x *vec;
  int n;
} vec_t_x;

typedef struct {
  num_t_x **mat;
  int n, m;
} mat_t_x;

typedef struct {
  num_t_x ***tensor;
  int n, m, c;
} tensor_3d_t_x;

#endif