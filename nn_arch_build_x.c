#include "nn_arch_build_x.h"
#include "nn_util_x.h"
#include "nn_operation_x.h"
#include <stdlib.h>

int nn_add_affine_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, int d1, int d2, nn_node_t_x **p1, nn_node_t_x **p2, nn_node_t_x **out) {
  nn_node_t_x *n1 = alloc_nn_node_x();
  n1->dat = alloc_mat_x(d2, d1);
  nn_add_node_x(n1, attr, PARAM_NODE_X);
  nn_node_t_x *n2 = alloc_nn_node_x();
  nn_add_node_x(n2, attr, INTERM_NODE_X);
  nn_node_t_x *in[2] = { n1, node };
  int status = nn_connect_x(2, in, n2, mat_vec_mult_x, deri_mat_vec_mult_x, NULL);
  if (status != 0)
    return status;
  nn_node_t_x *n3 = alloc_nn_node_x();
  n3->dat = alloc_vec_x(d2);
  nn_add_node_x(n3, attr, PARAM_NODE_X);
  nn_node_t_x *n4 = alloc_nn_node_x();
  nn_add_node_x(n4, attr, INTERM_NODE_X);
  nn_node_t_x *in2[2] = { n3, n2 };
  status = nn_connect_x(2, in2, n4, vec_add_x, deri_vec_add_x, NULL);
  if (status != 0)
    return status;
  if (p1 != NULL)
    *p1 = n1;
  if (p2 != NULL)
    *p2 = n3;
  if (out != NULL)
    *out = n4;
  return 0;
}

int nn_add_relu_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, nn_node_t_x **out) {
  nn_node_t_x *n1 = alloc_nn_node_x();
  nn_add_node_x(n1, attr, INTERM_NODE_X);
  nn_node_t_x *in[1] = { node };
  int status = nn_connect_x(1, in, n1, var_relu_x, deri_var_relu_x, NULL);
  if (status != 0)
    return status;
  if (out != NULL)
    *out = n1;
  return 0;
}

int nn_add_softmax_classifier_x(nn_attr_t_x *attr, nn_node_t_x *node, int type, nn_node_t_x **out) {
  nn_node_t_x *n1 = alloc_nn_node_x();
  nn_add_node_x(n1, attr, type);
  nn_node_t_x *in[1] = { node };
  int *index = (int *)malloc(sizeof(int));
  *index = 0;
  int status = nn_connect_x(1, in, n1, neg_log_softmax_x, deri_neg_log_softmax_x, index);
  if (status != 0)
    return status;
  if (out != NULL)
    *out = n1;
  return 0;
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

int var_subtract_from_x(var_t_x *op1, var_t_x *op2, num_t_x mult) {
  if (op1->type != op2->type)
    return WRONG_VAR_TYPE_X;
  int i, j, k;
  if (op1->type == REAL_X) {
    real_t_x *val = (real_t_x *)op1->val;
    real_t_x *sub = (real_t_x *)op2->val;
    val->real -= mult*sub->real;
  }
  else if (op1->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)op1->val;
    vec_t_x *sub = (vec_t_x *)op2->val;
    if (val->n != sub->n)
      return MISMATCH_DIMENSION_X;
    for (i = 0; i < val->n; ++i)
      val->vec[i] -= mult*sub->vec[i];
  }
  else if (op1->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)op1->val;
    mat_t_x *sub = (mat_t_x *)op2->val;
    if (val->n != sub->n || val->m != sub->m)
      return MISMATCH_DIMENSION_X;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        val->mat[i][j] -= mult*sub->mat[i][j];
  }
  else if (op1->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)op1->val;
    tensor_3d_t_x *sub = (tensor_3d_t_x *)op2->val;
    if (val->n != sub->n || val->m != sub->m || val->c != sub->c)
      return MISMATCH_DIMENSION_X;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          val->tensor[i][j][k] -= mult*sub->tensor[i][j][k];
  }
  return 0;
}

int gradient_descent_x(nn_attr_t_x *attr, num_t_x rate) {
  int status;
  int i;
  for (i = 0; i < attr->paramn; ++i) {
    if (attr->inputs[0]->btn) {
      int j;
      for (j = 0; j < attr->inputs[0]->btn; ++j) {
        status = var_subtract_from_x(attr->params[i]->dat, attr->params[i]->bt_back_dat[j], rate/attr->inputs[0]->btn);
        if (status != 0)
          return status;
      }
    }
    else {
      status = var_subtract_from_x(attr->params[i]->dat, attr->params[i]->back_dat, rate);
      if (status != 0)
        return status;
    }
  }
  return 0;
}