#include "nn_util_x.h"
#include <stdlib.h>

var_t_x *alloc_real_x(num_t_x num) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->type = REAL_X;
  res->val = malloc(sizeof(real_t_x));
#ifdef NN_NOCHECK_X
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
#endif
  real_t_x *val = (real_t_x *)res->val;
  val->real = num;
  return res;
}

var_t_x *alloc_vec_x(int n) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->type = VEC_X;
  res->val = malloc(sizeof(vec_t_x));
#ifdef NN_NOCHECK_X
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
#endif
  vec_t_x *val = (vec_t_x *)res->val;
  val->n = n;
  val->vec = (num_t_x *)malloc(n*sizeof(num_t_x));
#ifdef NN_NOCHECK_X
  if (val->vec == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
#endif
  return res;
}

var_t_x *alloc_mat_x(int n, int m) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->type = MAT_X;
  res->val = malloc(sizeof(mat_t_x));
#ifdef NN_NOCHECK_X
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
#endif
  int i;
  mat_t_x *val = (mat_t_x *)res->val;
  val->n = n;
  val->m = m;
  val->mat = (num_t_x **)malloc(n*sizeof(num_t_x*));
#ifdef NN_NOCHECK_X
  if (val->mat == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
#endif
  for (i = 0; i < n; ++i) {
    val->mat[i] = (num_t_x *)malloc(m*sizeof(num_t_x));
#ifdef NN_NOCHECK_X
    if (val->mat[i] == NULL) {
      int tmp;
      for (tmp = 0; tmp < i; ++tmp)
        free(val->mat[tmp]);
      free(val->mat);
      free(res->val);
      free(res);
      return NULL;
    }
#endif
  }
  return res;
}

var_t_x *alloc_tensor_3d_x(int n, int m, int c) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->type = TENSOR_3D_X;
  res->val = malloc(sizeof(tensor_3d_t_x));
#ifdef NN_NOCHECK_X
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
#endif
  int i = -1, j = -1;
  tensor_3d_t_x *val = (tensor_3d_t_x *)res->val;
  val->n = n;
  val->m = m;
  val->c = c;
  val->tensor = (num_t_x ***)malloc(n*sizeof(num_t_x **));
#ifdef NN_NOCHECK_X
  if (val->tensor == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
  int ok = 1;
#endif
  for (i = 0; i < n; ++i) {
    j = -1;
    val->tensor[i] = (num_t_x **)malloc(m*sizeof(num_t_x *));
#ifdef NN_NOCHECK_X
    if (val->tensor[i] == NULL) {
      ok = 0;
      break;
    }
#endif
    for (j = 0; j < m; ++j) {
      val->tensor[i][j] = (num_t_x *)malloc(c*sizeof(num_t_x));
#ifdef NN_NOCHECK_X
      if (val->tensor[i][j] == NULL) {
        ok = 0;
        break;
      }
#endif
    }
#ifdef NN_NOCHECK_X
    if (!ok)
      break;
#endif
  }
#ifdef NN_NOCHECK_X
  if (!ok) {
    int i2, j2;
    for (i2 = 0; i2 <= i; ++i2) {
      for (j2 = 0; j2 <= ((i2==i)?j:(m-1)); ++j2)
        free(val->tensor[i][j]);
      free(val->tensor[i]);
    }
    free(val->tensor);
    free(res->val);
    free(res);
    return NULL;
  }
#endif
  return res;
}

void free_var_x(var_t_x *pt) {
  if (pt == NULL)
    return;
  int i, j;
  if (pt->type == REAL_X) {
    real_t_x *val = (real_t_x *)pt->val;
    free(val);
  }
  else if (pt->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)pt->val;
    free(val->vec);
    free(val);
  }
  else if (pt->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)pt->val;
    for (i = 0; i < val->n; ++i)
      free(val->mat[i]);
    free(val->mat);
    free(val);
  }
  else if (pt->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)pt->val;
    for (i = 0; i < val->n; ++i) {
      for (j = 0; j < val->m; ++j)
        free(val->tensor[i][j]);
      free(val->tensor[i]);
    }
    free(val->tensor);
    free(val);
  }
  free(pt);
}

int fprint_var_x(FILE *out, var_t_x *pt) {
  int i, j, k;
  fprintf(out, "%d ", pt->type);
  if (pt->type == REAL_X)
    fprintf(out, "%.6f\n", ((real_t_x *)pt->val)->real);
  else if (pt->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)pt->val;
    fprintf(out, "%d\n", val->n);
    for (i = 0; i < val->n; ++i)
      fprintf(out, "%.6f ", val->vec[i]);
    fprintf(out, "\n");
  }
  else if (pt->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)pt->val;
    fprintf(out, "%d %d\n", val->n, val->m);
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        fprintf(out, "%.6f ", val->mat[i][j]);
    fprintf(out, "\n");
  }
  else if (pt->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)pt->val;
    fprintf(out, "%d %d %d\n", val->n, val->m, val->c);
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          fprintf(out, "%.6f ", val->tensor[i][j][k]);
    fprintf(out, "\n");
  }
  fflush(out);
  return ferror(out);
}

void set_seed_x(unsigned int seed) {
  srand(seed);
}

num_t_x random_uniform_x(num_t_x a, num_t_x b) {
  num_t_x r = ((num_t_x)rand())/RAND_MAX;
  return a+(b-a)*r;
}

void init_random_uniform_x(var_t_x *dat, num_t_x a, num_t_x b) {
  int i, j, k;
  if (dat->type == REAL_X) {
    real_t_x *val = (real_t_x *)dat->val;
    val->real = random_uniform_x(a, b);
  }
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      val->vec[i] = random_uniform_x(a, b);
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        val->mat[i][j] = random_uniform_x(a, b);
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          val->tensor[i][j][k] = random_uniform_x(a, b);
  }
}

void init_fixed_uniform_x(var_t_x *dat, num_t_x a) {
  int i, j, k;
  if (dat->type == REAL_X) {
    real_t_x *val = (real_t_x *)dat->val;
    val->real = a;
  }
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      val->vec[i] = a;
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        val->mat[i][j] = a;
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          val->tensor[i][j][k] = a;
  }
}

var_t_x *alloc_var_type_x(var_t_x *dat) {
  var_t_x *res = NULL;
  if (dat->type == REAL_X)
    res = alloc_real_x(0);
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    res = alloc_vec_x(val->n);
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    res = alloc_mat_x(val->n, val->m);
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    res = alloc_tensor_3d_x(val->n, val->m, val->c);
  }
  return res;
}

num_t_x var_squared_sum_x(var_t_x *dat) {
  num_t_x res = 0;
  int i, j, k;
  if (dat->type == REAL_X) {
    real_t_x *val = (real_t_x *)dat->val;
    res += val->real*val->real;
  }
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      res += val->vec[i]*val->vec[i];
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        res += val->mat[i][j]*val->mat[i][j];
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          res += val->tensor[i][j][k]*val->tensor[i][j][k];
  }
  return res;
}

int var_num_elements_x(var_t_x *dat) {
  if (dat->type == REAL_X)
    return 1;
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    return val->n;
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    return val->n*val->m;
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    return val->n*val->m*val->c;
  }
  return 0;
}

var_t_x *copy_var_type_x(var_t_x *dat) {
  var_t_x *res = NULL;
  int i, j, k;
  if (dat->type == REAL_X)
    res = alloc_real_x(((real_t_x *)dat->val)->real);
  else if (dat->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)dat->val;
    res = alloc_vec_x(val->n);
    if (res == NULL)
      return NULL;
    vec_t_x *val2 = (vec_t_x *)res->val;
    for (i = 0; i < val2->n; ++i)
      val2->vec[i] = val->vec[i];
  }
  else if (dat->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)dat->val;
    res = alloc_mat_x(val->n, val->m);
    if (res == NULL)
      return NULL;
    mat_t_x *val2 = (mat_t_x *)res->val;
    for (i = 0; i < val2->n; ++i)
      for (j = 0; j < val2->m; ++j)
        val2->mat[i][j] = val->mat[i][j];
  }
  else if (dat->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
    res = alloc_tensor_3d_x(val->n, val->m, val->c);
    if (res == NULL)
      return NULL;
    tensor_3d_t_x *val2 = (tensor_3d_t_x *)res->val;
    for (i = 0; i < val2->n; ++i)
      for (j = 0; j < val2->m; ++j)
        for (k = 0; k < val2->c; ++k)
          val2->tensor[i][j][k] = val->tensor[i][j][k];
  }
  return res;
}