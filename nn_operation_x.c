#include "nn_operation_x.h"
#include "nn_util_x.h"
#include <stdlib.h>
#include <math.h>

const char *nn_op_err_msg(int status) {
  if (status == 0)
    return "OK";
  else if (status == WRONG_ARG_NUM_X)
    return "Wrong number of arguments";
  else if (status == MISMATCH_DIMENSION_X)
    return "Mismatch dimensions";
  else if (status == MEMORY_ALLOC_FAILURE_X)
    return "Memory allocation failed";
  else if (status == WRONG_VAR_TYPE_X)
    return "Wrong argument types";
  else if (status == NULL_ARG_X)
    return "Null argument";
  else
    return "Unknown";
}

#ifdef NN_NOCHECK_X
int mat_mult_x(int n, var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#else
int mat_mult_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != MAT_X || ops[1]->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *m1 = (mat_t_x *)ops[0]->val;
  mat_t_x *m2 = (mat_t_x *)ops[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->m != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  if (*res == NULL) {
    *res = alloc_mat_x(m1->n, m2->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *m3 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n || m3->m != m2->m)
    return MISMATCH_DIMENSION_X;
#endif
  int i, j, k;
  for (i = 0; i < m3->n; ++i)
    for (j = 0; j < m3->m; ++j) {
      m3->mat[i][j] = 0;
      for (k = 0; k < m1->m; ++k)
        m3->mat[i][j] += m1->mat[i][k]*m2->mat[k][j];
    }
  return 0;
}

#ifdef NN_NOCHECK_X
int mat_vec_mult_x(int n, var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#else
int mat_vec_mult_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != MAT_X || ops[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *m1 = (mat_t_x *)ops[0]->val;
  vec_t_x *m2 = (vec_t_x *)ops[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->m != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m1->n != m3->n)
    return MISMATCH_DIMENSION_X;
#endif
  int i, j;
  for (i = 0; i < m3->n; ++i) {
    m3->vec[i] = 0;
    for (j = 0; j < m1->m; ++j)
      m3->vec[i] += m1->mat[i][j] * m2->vec[j];
  }
  return 0;
}

#ifdef NN_NOCHECK_X
int vec_mat_mult_x(int n, var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#else
int vec_mat_mult_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != VEC_X || ops[1]->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ops[0]->val;
  mat_t_x *m2 = (mat_t_x *)ops[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  if (*res == NULL) {
    *res = alloc_vec_x(m2->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m2->m != m3->n)
    return MISMATCH_DIMENSION_X;
#endif
  int i, j;
  for (i = 0; i < m3->n; ++i) {
    m3->vec[i] = 0;
    for (j = 0; j < m2->n; ++j)
      m3->vec[i] += m1->vec[j] * m2->mat[j][i];
  }
  return 0;
}

#ifdef NN_NOCHECK_X
int vec_inner_dot_x(int n, var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#else
int vec_inner_dot_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != VEC_X || ops[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ops[0]->val;
  vec_t_x *m2 = (vec_t_x *)ops[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  if (*res == NULL) {
    *res = alloc_real_x(0);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != REAL_X)
    return WRONG_VAR_TYPE_X;
#endif
  real_t_x *m3 = (real_t_x *)((*res)->val);
  m3->real = 0;
  int i;
  for (i = 0; i < m1->n; ++i)
    m3->real += m1->vec[i]*m2->vec[i];
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_vec_inner_dot_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#else
int deri_vec_inner_dot_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != VEC_X || ins[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
  if (out->type != REAL_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ins[0]->val;
  vec_t_x *m2 = (vec_t_x *)ins[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  num_t_x mult = 1;
  if (out_deri != NULL) {
#ifdef NN_NOCHECK_X
    if (out_deri->type != REAL_X)
      return WRONG_VAR_TYPE_X;
#endif
    mult = ((real_t_x *)out_deri->val)->real;
  }
  var_t_x **res = deri[0];
  int i;
  vec_t_x *m3;
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    m3->vec[i] += mult*m2->vec[i];
  res = deri[1];
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    m3->vec[i] += mult*m1->vec[i];
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_mat_vec_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#else
int deri_mat_vec_mult_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != MAT_X || ins[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *m1 = (mat_t_x *)ins[0]->val;
  vec_t_x *m2 = (vec_t_x *)ins[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->m != m2->n)
    return MISMATCH_DIMENSION_X;
  if (out->type != VEC_X)
    return WRONG_VAR_TYPE_X;
  if (out_deri == NULL)
    return NULL_ARG_X;
  if (out_deri->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *d = (vec_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
  if (d->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  var_t_x **res = deri[0];
  int i, j;
  mat_t_x *m3;
  if (*res == NULL) {
    *res = alloc_mat_x(m1->n, m1->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (mat_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      for (j = 0; j < m3->m; ++j)
        m3->mat[i][j] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n || m3->m != m1->m)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    for (j = 0; j < m3->m; ++j)
      m3->mat[i][j] += d->vec[i]*m2->vec[j];
  res = deri[1];
  vec_t_x *m4;
  if (*res == NULL) {
    *res = alloc_vec_x(m2->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m4 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m4->n; ++i)
      m4->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m4 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m4->n != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m4->n; ++i)
    for (j = 0; j < d->n; ++j)
      m4->vec[i] += d->vec[j]*m1->mat[j][i];
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_vec_mat_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#else
int deri_vec_mat_mult_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != VEC_X || ins[1]->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ins[0]->val;
  mat_t_x *m2 = (mat_t_x *)ins[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
  if (out->type != VEC_X)
    return WRONG_VAR_TYPE_X;
  if (out_deri == NULL)
    return NULL_ARG_X;
  if (out_deri->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *d = (vec_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
  if (d->n != m2->m)
    return MISMATCH_DIMENSION_X;
#endif
  var_t_x **res = deri[0];
  int i, j;
  vec_t_x *m3;
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    for (j = 0; j < d->n; ++j)
      m3->vec[i] += d->vec[j]*m2->mat[i][j];
  res = deri[1];
  mat_t_x *m4;
  if (*res == NULL) {
    *res = alloc_mat_x(m2->n, m2->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m4 = (mat_t_x *)((*res)->val);
    for (i = 0; i < m4->n; ++i)
      for (j = 0; j < m4->m; ++j)
        m4->mat[i][j] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  m4 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m4->n != m2->n || m4->m != m2->m)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m4->n; ++i)
    for (j = 0; j < m4->m; ++j)
      m4->mat[i][j] += m1->vec[i]*d->vec[j];
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_mat_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#else
int deri_mat_mult_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != MAT_X || ins[1]->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *m1 = (mat_t_x *)ins[0]->val;
  mat_t_x *m2 = (mat_t_x *)ins[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->m != m2->n)
    return MISMATCH_DIMENSION_X;
  if (out->type != MAT_X)
    return WRONG_VAR_TYPE_X;
  if (out_deri == NULL)
    return NULL_ARG_X;
  if (out_deri->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  mat_t_x *d = (mat_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
  if (d->n != m1->n || d->m != m2->m)
    return MISMATCH_DIMENSION_X;
#endif
  var_t_x **res = deri[0];
  int i, j, k;
  mat_t_x *m3;
  if (*res == NULL) {
    *res = alloc_mat_x(m1->n, m1->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (mat_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      for (j = 0; j < m3->m; ++j)
        m3->mat[i][j] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n || m3->m != m1->m)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    for (j = 0; j < m3->m; ++j)
      for (k = 0; k < d->m; ++k)
        m3->mat[i][j] += d->mat[i][k]*m2->mat[j][k];
  res = deri[1];
  if (*res == NULL) {
    *res = alloc_mat_x(m2->n, m2->m);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (mat_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      for (j = 0; j < m3->m; ++j)
        m3->mat[i][j] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != MAT_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m2->n || m3->m != m2->m)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    for (j = 0; j < m3->m; ++j)
      for (k = 0; k < d->n; ++k)
        m3->mat[i][j] += d->mat[k][j]*m1->mat[k][i];
  return 0;
}

#ifdef NN_NOCHECK_X
int var_activate_x(int n, var_t_x **ops, var_t_x **res, void *attr, activate_func_t_x acti) {
#else
int var_activate_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *attr, activate_func_t_x acti) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 1)
    return WRONG_ARG_NUM_X;
#endif
  if (ops[0]->type == REAL_X) {
    real_t_x *m1 = (real_t_x *)ops[0]->val;
    if (*res == NULL) {
      *res = alloc_real_x(0);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != REAL_X)
      return WRONG_VAR_TYPE_X;
#endif
    real_t_x *m2 = (real_t_x *)((*res)->val);
    m2->real = acti(m1->real, attr);
  }
  else if (ops[0]->type == VEC_X) {
    vec_t_x *m1 = (vec_t_x *)ops[0]->val;
    if (*res == NULL) {
      *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != VEC_X)
      return WRONG_VAR_TYPE_X;
#endif
    vec_t_x *m2 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n)
      return MISMATCH_DIMENSION_X;
#endif
    int i;
    for (i = 0; i < m1->n; ++i)
      m2->vec[i] = acti(m1->vec[i], attr);
  }
  else if (ops[0]->type == MAT_X) {
    mat_t_x *m1 = (mat_t_x *)ops[0]->val;
    if (*res == NULL) {
      *res = alloc_mat_x(m1->n, m1->m);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != MAT_X)
      return WRONG_VAR_TYPE_X;
#endif
    mat_t_x *m2 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n || m1->m != m2->m)
      return MISMATCH_DIMENSION_X;
#endif
    int i, j;
    for (i = 0; i < m1->n; ++i)
      for (j = 0; j < m1->m; ++j)
        m2->mat[i][j] = acti(m1->mat[i][j], attr);
  }
  else if (ops[0]->type == TENSOR_3D_X) {
    tensor_3d_t_x *m1 = (tensor_3d_t_x *)ops[0]->val;
    if (*res == NULL) {
      *res = alloc_tensor_3d_x(m1->n, m1->m, m1->c);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != TENSOR_3D_X)
      return WRONG_VAR_TYPE_X;
#endif
    tensor_3d_t_x *m2 = (tensor_3d_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n || m1->m != m2->m || m1->c != m2->c)
      return MISMATCH_DIMENSION_X;
#endif
    int i, j, k;
    for (i = 0; i < m1->n; ++i)
      for (j = 0; j < m1->m; ++j)
        for (k = 0; k < m1->c; ++k)
          m2->tensor[i][j][k] = acti(m1->tensor[i][j][k], attr);
  }
  return 0;
}

num_t_x relu_activate_x(num_t_x num, void *UNUSED(attr)) {
  if (num < 0)
    return 0;
  else
    return num;
}

int var_relu_x(int n, var_t_x **ops, var_t_x **res, void *attr) {
  return var_activate_x(n, ops, res, attr, relu_activate_x);
}

#ifdef NN_NOCHECK_X
int deri_var_activate_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr, activate_func_t_x deri_acti) {
#else
int deri_var_activate_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *attr, activate_func_t_x deri_acti) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 1)
    return WRONG_ARG_NUM_X;
#endif
  if (ins[0]->type == REAL_X) {
    real_t_x *m1 = (real_t_x *)ins[0]->val;
#ifdef NN_NOCHECK_X
    if (out->type != REAL_X)
      return WRONG_VAR_TYPE_X;
#endif
    num_t_x mult = 1;
    if (out_deri != NULL)
      mult = ((real_t_x *)out_deri->val)->real;
    var_t_x **res = deri[0];
    if (*res == NULL) {
      *res = alloc_real_x(0);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != REAL_X)
      return WRONG_VAR_TYPE_X;
#endif
    real_t_x *m2 = (real_t_x *)((*res)->val);
    m2->real += mult*deri_acti(m1->real, attr);
  }
  else if (ins[0]->type == VEC_X) {
    vec_t_x *m1 = (vec_t_x *)ins[0]->val;
#ifdef NN_NOCHECK_X
    if (out->type != VEC_X)
      return WRONG_VAR_TYPE_X;
    if (out_deri == NULL)
      return NULL_ARG_X;
    if (out_deri->type != VEC_X)
      return WRONG_VAR_TYPE_X;
#endif
    vec_t_x *d = (vec_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
    if (d->n != m1->n)
      return MISMATCH_DIMENSION_X;
#endif
    var_t_x **res = deri[0];
    vec_t_x *m2;
    int i;
    if (*res == NULL) {
      *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
      m2 = (vec_t_x *)((*res)->val);
      for (i = 0; i < m2->n; ++i)
        m2->vec[i] = 0;
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != VEC_X)
      return WRONG_VAR_TYPE_X;
#endif
    m2 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < m1->n; ++i)
      m2->vec[i] += d->vec[i]*deri_acti(m1->vec[i], attr);
  }
  else if (ins[0]->type == MAT_X) {
    mat_t_x *m1 = (mat_t_x *)ins[0]->val;
#ifdef NN_NOCHECK_X
    if (out->type != MAT_X)
      return WRONG_VAR_TYPE_X;
    if (out_deri == NULL)
      return NULL_ARG_X;
    if (out_deri->type != MAT_X)
      return WRONG_VAR_TYPE_X;
#endif
    mat_t_x *d = (mat_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
    if (d->n != m1->n || d->m != m1->m)
      return MISMATCH_DIMENSION_X;
#endif
    var_t_x **res = deri[0];
    mat_t_x *m2;
    int i, j;
    if (*res == NULL) {
      *res = alloc_mat_x(m1->n, m1->m);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
      m2 = (mat_t_x *)((*res)->val);
      for (i = 0; i < m2->n; ++i)
        for (j = 0; j < m2->m; ++j)
          m2->mat[i][j] = 0;
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != MAT_X)
      return WRONG_VAR_TYPE_X;
#endif
    m2 = (mat_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n || m1->m != m2->m)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < m1->n; ++i)
      for (j = 0; j < m1->m; ++j)
        m2->mat[i][j] += d->mat[i][j]*deri_acti(m1->mat[i][j], attr);
  }
  else if (ins[0]->type == TENSOR_3D_X) {
    tensor_3d_t_x *m1 = (tensor_3d_t_x *)ins[0]->val;
#ifdef NN_NOCHECK_X
    if (out->type != TENSOR_3D_X)
      return WRONG_VAR_TYPE_X;
    if (out_deri == NULL)
      return NULL_ARG_X;
    if (out_deri->type != TENSOR_3D_X)
      return WRONG_VAR_TYPE_X;
#endif
    tensor_3d_t_x *d = (tensor_3d_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
    if (d->n != m1->n || d->m != m1->m || d->c != m1->c)
      return MISMATCH_DIMENSION_X;
#endif
    var_t_x **res = deri[0];
    tensor_3d_t_x *m2;
    int i, j, k;
    if (*res == NULL) {
      *res = alloc_tensor_3d_x(m1->n, m1->m, m1->c);
#ifdef NN_NOCHECK_X
      if (*res == NULL)
        return MEMORY_ALLOC_FAILURE_X;
#endif
      m2 = (tensor_3d_t_x *)((*res)->val);
      for (i = 0; i < m2->n; ++i)
        for (j = 0; j < m2->m; ++j)
          for (k = 0; k < m2->c; ++k)
            m2->tensor[i][j][k] = 0;
    }
#ifdef NN_NOCHECK_X
    if ((*res)->type != TENSOR_3D_X)
      return WRONG_VAR_TYPE_X;
#endif
    m2 = (tensor_3d_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
    if (m1->n != m2->n || m1->m != m2->m || m1->c != m2->c)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < m1->n; ++i)
      for (j = 0; j < m1->m; ++j)
        for (k = 0; k < m1->c; ++k)
          m2->tensor[i][j][k] += d->tensor[i][j][k]*deri_acti(m1->tensor[i][j][k], attr);
  }
  return 0;
}

num_t_x deri_relu_activate_x(num_t_x num, void *UNUSED(attr)) {
  if (num < 0)
    return 0;
  else
    return 1;
}

int deri_var_relu_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr) {
  return deri_var_activate_x(n, ins, out, out_deri, deri, attr, deri_relu_activate_x);
}

#ifdef NN_NOCHECK_X
int vec_add_x(int n, var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#else
int vec_add_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != VEC_X || ops[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ops[0]->val;
  vec_t_x *m2 = (vec_t_x *)ops[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
#endif
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  int i;
  for (i = 0; i < m3->n; ++i)
    m3->vec[i] = m1->vec[i] + m2->vec[i];
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_vec_add_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#else
int deri_vec_add_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *UNUSED(attr)) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 2)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != VEC_X || ins[1]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ins[0]->val;
  vec_t_x *m2 = (vec_t_x *)ins[1]->val;
#ifdef NN_NOCHECK_X
  if (m1->n != m2->n)
    return MISMATCH_DIMENSION_X;
  if (out->type != VEC_X)
    return WRONG_VAR_TYPE_X;
  if (out_deri == NULL)
    return NULL_ARG_X;
  if (out_deri->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *d = (vec_t_x *)out_deri->val;
#ifdef NN_NOCHECK_X
  if (d->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  var_t_x **res = deri[0];
  int i;
  vec_t_x *m3;
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    m3->vec[i] += d->vec[i];
  res = deri[1];
  if (*res == NULL) {
    *res = alloc_vec_x(m2->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  for (i = 0; i < m3->n; ++i)
    m3->vec[i] += d->vec[i];
  return 0;
}

#ifdef NN_NOCHECK_X
int neg_log_softmax_x(int n, var_t_x **ops, var_t_x **res, void *attr) {
#else
int neg_log_softmax_x(int UNUSED(n), var_t_x **ops, var_t_x **res, void *attr) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 1)
    return WRONG_ARG_NUM_X;
  if (ops[0]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ops[0]->val;
#ifdef NN_NOCHECK_X
  if (attr == NULL)
    return NULL_ARG_X;
#endif
  int *index = (int *)attr;
  if (*res == NULL) {
    *res = alloc_real_x(0);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != REAL_X)
    return WRONG_VAR_TYPE_X;
#endif
  real_t_x *m3 = (real_t_x *)((*res)->val);
  int i;
  num_t_x max = m1->vec[0];
  for (i = 1; i < m1->n; ++i)
    if (m1->vec[i] > max)
      max = m1->vec[i];
  num_t_x a = exp(m1->vec[*index]-max);
  num_t_x b = 0;
  for (i = 0; i < m1->n; ++i)
    b += exp(m1->vec[i]-max);
  m3->real = -log(a/b);
  return 0;
}

#ifdef NN_NOCHECK_X
int deri_neg_log_softmax_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr) {
#else
int deri_neg_log_softmax_x(int UNUSED(n), var_t_x **ins, var_t_x *UNUSED(out), var_t_x *out_deri, var_t_x ***deri, void *attr) {
#endif
#ifdef NN_NOCHECK_X
  if (n != 1)
    return WRONG_ARG_NUM_X;
  if (ins[0]->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  vec_t_x *m1 = (vec_t_x *)ins[0]->val;
#ifdef NN_NOCHECK_X
  if (attr == NULL)
    return NULL_ARG_X;
#endif
  int *index = (int *)attr;
#ifdef NN_NOCHECK_X
  if (out->type != REAL_X)
    return WRONG_VAR_TYPE_X;
#endif
  num_t_x mult = 1;
  if (out_deri != NULL) {
#ifdef NN_NOCHECK_X
    if (out_deri->type != REAL_X)
      return WRONG_VAR_TYPE_X;
#endif
    mult = ((real_t_x *)out_deri->val)->real;
  }
  var_t_x **res = deri[0];
  int i;
  vec_t_x *m3;
  if (*res == NULL) {
    *res = alloc_vec_x(m1->n);
#ifdef NN_NOCHECK_X
    if (*res == NULL)
      return MEMORY_ALLOC_FAILURE_X;
#endif
    m3 = (vec_t_x *)((*res)->val);
    for (i = 0; i < m3->n; ++i)
      m3->vec[i] = 0;
  }
#ifdef NN_NOCHECK_X
  if ((*res)->type != VEC_X)
    return WRONG_VAR_TYPE_X;
#endif
  m3 = (vec_t_x *)((*res)->val);
#ifdef NN_NOCHECK_X
  if (m3->n != m1->n)
    return MISMATCH_DIMENSION_X;
#endif
  num_t_x max = m1->vec[0];
  for (i = 1; i < m1->n; ++i)
    if (m1->vec[i] > max)
      max = m1->vec[i];
  num_t_x b = 0;
  for (i = 0; i < m1->n; ++i)
    b += exp(m1->vec[i]-max);
  for (i = 0; i < m1->n; ++i) {
    if (i != *index) {
      num_t_x a = exp(m1->vec[i]-max);
      m3->vec[i] += mult*a/b;
    }
    else {
      num_t_x a = exp(m1->vec[*index]-max);
      m3->vec[i] += mult*(a/b-1);
    }
  }
  return 0;
}

void *copy_softmax_attr_x(void *arg) {
  int *index = (int *)arg;
  int *res = (int *)malloc(sizeof(int));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  *res = *index;
  return res;
}