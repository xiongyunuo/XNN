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
#ifdef NN_NOCHECK_X
  if (status != 0)
    return status;
#endif
  nn_node_t_x *n3 = alloc_nn_node_x();
  n3->dat = alloc_vec_x(d2);
  nn_add_node_x(n3, attr, PARAM_NODE_X);
  nn_node_t_x *n4 = alloc_nn_node_x();
  nn_add_node_x(n4, attr, INTERM_NODE_X);
  nn_node_t_x *in2[2] = { n3, n2 };
  status = nn_connect_x(2, in2, n4, vec_add_x, deri_vec_add_x, NULL);
#ifdef NN_NOCHECK_X
  if (status != 0)
    return status;
#endif
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
#ifdef NN_NOCHECK_X
  int status = nn_connect_x(1, in, n1, var_relu_x, deri_var_relu_x, NULL);
#else
  nn_connect_x(1, in, n1, var_relu_x, deri_var_relu_x, NULL);
#endif
#ifdef NN_NOCHECK_X
  if (status != 0)
    return status;
#endif
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
#ifdef NN_NOCHECK_X
  int status = nn_connect_x(1, in, n1, neg_log_softmax_x, deri_neg_log_softmax_x, index);
#else
  nn_connect_x(1, in, n1, neg_log_softmax_x, deri_neg_log_softmax_x, index);
#endif
  n1->attr_cp = copy_softmax_attr_x;
#ifdef NN_NOCHECK_X
  if (status != 0)
    return status;
#endif
  if (out != NULL)
    *out = n1;
  return 0;
}

int var_subtract_from_x(var_t_x *op1, var_t_x *op2, num_t_x mult, num_t_x mult2) {
#ifdef NN_NOCHECK_X
  if (op1->type != op2->type)
    return WRONG_VAR_TYPE_X;
#endif
  int i, j, k;
  if (op1->type == REAL_X) {
    real_t_x *val = (real_t_x *)op1->val;
    real_t_x *sub = (real_t_x *)op2->val;
    val->real *= mult2;
    val->real -= mult*sub->real;
  }
  else if (op1->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)op1->val;
    vec_t_x *sub = (vec_t_x *)op2->val;
#ifdef NN_NOCHECK_X
    if (val->n != sub->n)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < val->n; ++i) {
      val->vec[i] *= mult2;
      val->vec[i] -= mult*sub->vec[i];
    }
  }
  else if (op1->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)op1->val;
    mat_t_x *sub = (mat_t_x *)op2->val;
#ifdef NN_NOCHECK_X
    if (val->n != sub->n || val->m != sub->m)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j) {
        val->mat[i][j] *= mult2;
        val->mat[i][j] -= mult*sub->mat[i][j];
      }
  }
  else if (op1->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)op1->val;
    tensor_3d_t_x *sub = (tensor_3d_t_x *)op2->val;
#ifdef NN_NOCHECK_X
    if (val->n != sub->n || val->m != sub->m || val->c != sub->c)
      return MISMATCH_DIMENSION_X;
#endif
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k) {
          val->tensor[i][j][k] *= mult2;
          val->tensor[i][j][k] -= mult*sub->tensor[i][j][k];
        }
  }
  return 0;
}

int gradient_descent_x(nn_attr_t_x *attr, num_t_x rate, int thread_count) {
  grad_descent_info_t_x info;
  info.attr = attr;
  info.rate = rate;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, grad_descent_worker_x);
}

grad_vel_t_x *alloc_grad_vel_x(nn_attr_t_x *attr) {
  grad_vel_t_x *vel = (grad_vel_t_x *)malloc(sizeof(grad_vel_t_x));
  int i;
  vel->n = attr->paramn;
  vel->vels = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  for (i = 0; i < vel->n; ++i) {
    var_t_x *dat = attr->params[i]->dat;
    if (dat->type == REAL_X)
      vel->vels[i] = alloc_real_x(0);
    else if (dat->type == VEC_X) {
      vec_t_x *val = (vec_t_x *)dat->val;
      vel->vels[i] = alloc_vec_x(val->n);
    }
    else if (dat->type == MAT_X) {
      mat_t_x *val = (mat_t_x *)dat->val;
      vel->vels[i] = alloc_mat_x(val->n, val->m);
    }
    else if (dat->type == TENSOR_3D_X) {
      tensor_3d_t_x *val = (tensor_3d_t_x *)dat->val;
      vel->vels[i] = alloc_tensor_3d_x(val->n, val->m, val->c);
    }
    init_fixed_uniform_x(vel->vels[i], 0);
  }
  return vel;
}

int var_mult_x(var_t_x *op, num_t_x mult) {
  int i, j, k;
  if (op->type == REAL_X) {
    real_t_x *val = (real_t_x *)op->val;
    val->real *= mult;
  }
  else if (op->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)op->val;
    for (i = 0; i < val->n; ++i)
      val->vec[i] *= mult;
  }
  else if (op->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)op->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        val->mat[i][j] *= mult;
  }
  else if (op->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)op->val;
    for (i = 0; i < val->n; ++i)
      for (j = 0; j < val->m; ++j)
        for (k = 0; k < val->c; ++k)
          val->tensor[i][j][k] *= mult;
  }
  return 0;
}

int momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, int thread_count) {
  momentum_descent_info_t_x info;
  info.attr = attr;
  info.vel = vel;
  info.alpha = alpha;
  info.rate = rate;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, momentum_descent_worker_x);
}

void free_grad_vel_x(grad_vel_t_x *vel) {
  int i;
  for (i = 0; i < vel->n; ++i)
    free_var_x(vel->vels[i]);
  free(vel->vels);
  free(vel);
}

int momentum_descent_worker_x(void *info, int i) {
  momentum_descent_info_t_x *inf = (momentum_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  grad_vel_t_x *vel = inf->vel;
  num_t_x alpha = inf->alpha;
  num_t_x rate = inf->rate;
  int status;
  if (attr->inputs[0]->btn) {
    int j;
    for (j = 0; j < attr->inputs[0]->btn; ++j) {
      if (j == 0)
        status = var_subtract_from_x(vel->vels[i], attr->params[i]->bt_back_dat[j], rate/attr->inputs[0]->btn, alpha);
      else
        status = var_subtract_from_x(vel->vels[i], attr->params[i]->bt_back_dat[j], rate/attr->inputs[0]->btn, 1.0);
#ifdef NN_NOCHECK_X
      if (status != 0)
        return status;
#endif
    }
  }
  else {
    status = var_subtract_from_x(vel->vels[i], attr->params[i]->back_dat, rate, alpha);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  status = var_subtract_from_x(attr->params[i]->dat, vel->vels[i], -1, 1.0);
#ifdef NN_NOCHECK_X
  if (status != 0)
    return status;
#endif
  return 0;
}

int grad_descent_worker_x(void *info, int i) {
  grad_descent_info_t_x *inf = (grad_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  num_t_x rate = inf->rate;
  int status;
  if (attr->inputs[0]->btn) {
    int j;
    for (j = 0; j < attr->inputs[0]->btn; ++j) {
      status = var_subtract_from_x(attr->params[i]->dat, attr->params[i]->bt_back_dat[j], rate/attr->inputs[0]->btn, 1.0);
#ifdef NN_NOCHECK_X
      if (status != 0)
        return status;
#endif
    }
  }
  else {
    status = var_subtract_from_x(attr->params[i]->dat, attr->params[i]->back_dat, rate, 1.0);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  return 0;
}