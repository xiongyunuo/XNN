#include "nn_util_x.h"
#include <stdlib.h>

var_t_x *alloc_real_x(num_t_x num) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
  if (res == NULL)
    return NULL;
  res->type = REAL_X;
  res->val = malloc(sizeof(real_t_x));
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
  real_t_x *val = (real_t_x *)res->val;
  val->real = num;
  return res;
}

var_t_x *alloc_vec_x(int n) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
  if (res == NULL)
    return NULL;
  res->type = VEC_X;
  res->val = malloc(sizeof(vec_t_x));
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
  vec_t_x *val = (vec_t_x *)res->val;
  val->n = n;
  val->vec = (num_t_x *)malloc(n*sizeof(num_t_x));
  if (val->vec == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
  return res;
}

var_t_x *alloc_mat_x(int n, int m) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
  if (res == NULL)
    return NULL;
  res->type = MAT_X;
  res->val = malloc(sizeof(mat_t_x));
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
  int i;
  mat_t_x *val = (mat_t_x *)res->val;
  val->n = n;
  val->m = m;
  val->mat = (num_t_x **)malloc(n*sizeof(num_t_x*));
  if (val->mat == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
  for (i = 0; i < n; ++i) {
    val->mat[i] = (num_t_x *)malloc(m*sizeof(num_t_x));
    if (val->mat[i] == NULL) {
      int tmp;
      for (tmp = 0; tmp < i; ++tmp)
        free(val->mat[tmp]);
      free(val->mat);
      free(res->val);
      free(res);
      return NULL;
    }
  }
  return res;
}

var_t_x *alloc_tensor_3d_x(int n, int m, int c) {
  var_t_x *res = (var_t_x *)malloc(sizeof(var_t_x));
  if (res == NULL)
    return NULL;
  res->type = TENSOR_3D_X;
  res->val = malloc(sizeof(tensor_3d_t_x));
  if (res->val == NULL) {
    free(res);
    return NULL;
  }
  int i = -1, j = -1;
  tensor_3d_t_x *val = (tensor_3d_t_x *)res->val;
  val->n = n;
  val->m = m;
  val->c = c;
  val->tensor = (num_t_x ***)malloc(n*sizeof(num_t_x **));
  if (val->tensor == NULL) {
    free(res->val);
    free(res);
    return NULL;
  }
  int ok = 1;
  for (i = 0; i < n; ++i) {
    j = -1;
    val->tensor[i] = (num_t_x **)malloc(m*sizeof(num_t_x *));
    if (val->tensor[i] == NULL) {
      ok = 0;
      break;
    }
    for (j = 0; j < m; ++j) {
      val->tensor[i][j] = (num_t_x *)malloc(c*sizeof(num_t_x));
      if (val->tensor[i][j] == NULL) {
        ok = 0;
        break;
      }
    }
    if (!ok)
      break;
  }
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