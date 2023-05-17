#include "nn_arch_build_x.h"
#include "nn_util_x.h"
#include "nn_operation_x.h"
#include <stdlib.h>
#include <math.h>

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
  vel->t = 0;
  vel->vels = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  vel->rs = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  vel->ss = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  for (i = 0; i < vel->n; ++i) {
    var_t_x *dat = attr->params[i]->dat;
    vel->vels[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->vels[i], 0);
    vel->rs[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->rs[i], 0);
    vel->ss[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->ss[i], 0);
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
  for (i = 0; i < vel->n; ++i) {
    free_var_x(vel->vels[i]);
    free_var_x(vel->rs[i]);
    free_var_x(vel->ss[i]);
  }
  free(vel->vels);
  free(vel->rs);
  free(vel->ss);
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

mold_t_x *alloc_mold_x(nn_attr_t_x *attr, int M, num_t_x T, num_t_x Q) {
  if (M%2 != 0)
    return NULL;
  mold_t_x *vel = (mold_t_x *)malloc(sizeof(mold_t_x));
  int i, j;
  vel->n = attr->paramn;
  vel->t = 0;
  vel->T = T;
  vel->M = M;
  vel->Q = (num_t_x *)malloc(sizeof(num_t_x)*M);
  for (i = 0; i < M; ++i)
    vel->Q[i] = Q;
  vel->vels = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  vel->rs = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  vel->ss = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
  for (i = 0; i < vel->n; ++i) {
    var_t_x *dat = attr->params[i]->dat;
    vel->vels[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->vels[i], 0.0);
    vel->rs[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->rs[i], 0);
    vel->ss[i] = alloc_var_type_x(dat);
    init_fixed_uniform_x(vel->ss[i], 0);
  }
  vel->thetas = (var_t_x ***)malloc(M*sizeof(var_t_x **));
  vel->vthetas = (var_t_x ***)malloc(M*sizeof(var_t_x **));
  for (j = 0; j < M; ++j) {
    vel->thetas[j] = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
    vel->vthetas[j] = (var_t_x **)malloc(vel->n*sizeof(var_t_x *));
    for (i = 0; i < vel->n; ++i) {
      var_t_x *dat = attr->params[i]->dat;
      vel->thetas[j][i] = alloc_var_type_x(dat);
      vel->vthetas[j][i] = alloc_var_type_x(dat);
      init_fixed_uniform_x(vel->thetas[j][i], 0);
      init_fixed_uniform_x(vel->vthetas[j][i], 1);
    }
  }
  return vel;
}

void free_mold_x(mold_t_x *vel) {
  int i, j;
  for (i = 0; i < vel->n; ++i) {
    free_var_x(vel->vels[i]);
    free_var_x(vel->rs[i]);
    free_var_x(vel->ss[i]);
  }
  free(vel->vels);
  free(vel->rs);
  free(vel->ss);
  free(vel->Q);
  for (j = 0; j < vel->M; ++j) {
    for (i = 0; i < vel->n; ++i) {
      free_var_x(vel->thetas[j][i]);
      free_var_x(vel->vthetas[j][i]);
    }
    free(vel->thetas[j]);
    free(vel->vthetas[j]);
  }
  free(vel->thetas);
  free(vel->vthetas);
  free(vel);
}

void calc_NH_force_x(num_t_x m, int f, num_t_x *v, num_t_x *Q, num_t_x *vtheta, num_t_x *res, int M, num_t_x beta) {
  num_t_x sum = 0;
  int i;
  for (i = 0; i < f; ++i)
    sum += m*v[i]*v[i];
  res[0] = (sum-f*(1/beta))/Q[0];
  for (i = 1; i < M; ++i)
    res[i] = (Q[i-1]*vtheta[i-1]*vtheta[i-1]-(1/beta))/Q[i];
}

void NH_update_1_x(int M, num_t_x *Q, num_t_x *theta, num_t_x *vtheta, num_t_x *x, num_t_x *v, num_t_x *force, num_t_x beta, num_t_x h) {
  num_t_x *NHF = (num_t_x *)malloc(M*sizeof(num_t_x));
  calc_NH_force_x(1.0, 1, v, Q, vtheta, NHF, M, beta);
  v[0] = v[0]*exp(-0.5*h*vtheta[0])+0.5*h*force[0]*exp(-0.25*h*vtheta[0]);
  int M2 = M/2;
  int k;
  for (k = 1; k <= M2; ++k)
    theta[2*k-2] = theta[2*k-2]+h*vtheta[2*k-2]/2;
  for (k = 1; k <= M2; ++k)
    vtheta[2*k-1] = vtheta[2*k-1]*exp(-0.5*h*((k==M2)?0:vtheta[2*k]))+0.5*h*NHF[2*k-1]*exp(-0.25*h*((k==M2)?0:vtheta[2*k]));
  x[0] = x[0]+h*v[0];
  for (k = 1; k <= M2; ++k)
    theta[2*k-1] = theta[2*k-1]+h*vtheta[2*k-1];
  calc_NH_force_x(1.0, 1, v, Q, vtheta, NHF, M, beta);
  for (k = 1; k <= M2; ++k)
    vtheta[2*k-2] = vtheta[2*k-2]*exp(-h*vtheta[2*k-1])+h*NHF[2*k-2]*exp(-0.5*h*vtheta[2*k-1]);
  free(NHF);
}

void NH_update_2_x(int M, num_t_x *Q, num_t_x *theta, num_t_x *vtheta, num_t_x *UNUSED(x), num_t_x *v, num_t_x *force, num_t_x beta, num_t_x h) {
  num_t_x *NHF = (num_t_x *)malloc(M*sizeof(num_t_x));
  int M2 = M/2;
  int k;
  v[0] = v[0]*exp(-0.5*h*vtheta[0])+0.5*h*force[0]*exp(-0.25*h*vtheta[0]);
  for (k = 1; k <= M2; ++k)
    theta[2*k-2] = theta[2*k-2]+h*vtheta[2*k-2]/2;
  calc_NH_force_x(1.0, 1, v, Q, vtheta, NHF, M, beta);
  for (k = 1; k <= M2; ++k)
    vtheta[2*k-1] = vtheta[2*k-1]*exp(-0.5*h*((k==M2)?0:vtheta[2*k]))+0.5*h*NHF[2*k-1]*exp(-0.25*h*((k==M2)?0:vtheta[2*k]));
  free(NHF);
}

void mold_single_update_x(mold_t_x *mold, num_t_x *x, num_t_x *v, num_t_x **theta_cp, num_t_x **vtheta_cp, num_t_x rate, num_t_x *theta, num_t_x *vtheta, num_t_x *force, int first) {
  int m;
  if (first)
    NH_update_1_x(mold->M, mold->Q, theta, vtheta, x, v, force, 1.0/mold->T, rate);
  else {
    NH_update_2_x(mold->M, mold->Q, theta, vtheta, x, v, force, 1.0/mold->T, rate);
    NH_update_1_x(mold->M, mold->Q, theta, vtheta, x, v, force, 1.0/mold->T, rate);
  }
  for (m = 0; m < mold->M; ++m) {
    (*(theta_cp[m])) = theta[m];
    (*(vtheta_cp[m])) = vtheta[m];
  }
}

void mold_var_update_x(mold_t_x *mold, var_t_x *x, var_t_x *v, var_t_x **theta, var_t_x **vtheta, var_t_x *force, num_t_x rate, int first) {
  int i, j, k, m;
  num_t_x *theta2 = (num_t_x *)malloc(mold->M*sizeof(num_t_x));
  num_t_x *vtheta2 = (num_t_x *)malloc(mold->M*sizeof(num_t_x));
  num_t_x **theta_cp = (num_t_x **)malloc(mold->M*sizeof(num_t_x *));
  num_t_x **vtheta_cp = (num_t_x **)malloc(mold->M*sizeof(num_t_x *));
  if (x->type == REAL_X) {
    real_t_x *val = (real_t_x *)x->val;
    num_t_x *x2 = &(val->real);
    val = (real_t_x *)v->val;
    num_t_x *v2 = &(val->real);
    for (m = 0; m < mold->M; ++m) {
      val = (real_t_x *)theta[m]->val;
      theta2[m] = val->real;
      theta_cp[m] = &(val->real);
      val = (real_t_x *)vtheta[m]->val;
      vtheta2[m] = val->real;
      vtheta_cp[m] = &(val->real);
    }
    val = (real_t_x *)force->val;
    num_t_x *f = &(val->real);
    mold_single_update_x(mold, x2, v2, theta_cp, vtheta_cp, rate, theta2, vtheta2, f, first);
  }
  else if (x->type == VEC_X) {
    vec_t_x *val = (vec_t_x *)x->val;
    int n = val->n;
    for (i = 0; i < n; ++i) {
      val = (vec_t_x *)x->val;
      num_t_x *x2 = &(val->vec[i]);
      val = (vec_t_x *)v->val;
      num_t_x *v2 = &(val->vec[i]);
      for (m = 0; m < mold->M; ++m) {
        val = (vec_t_x *)theta[m]->val;
        theta2[m] = val->vec[i];
        theta_cp[m] = &(val->vec[i]);
        val = (vec_t_x *)vtheta[m]->val;
        vtheta2[m] = val->vec[i];
        vtheta_cp[m] = &(val->vec[i]);
      }
      val = (vec_t_x *)force->val;
      num_t_x *f = &(val->vec[i]);
      mold_single_update_x(mold, x2, v2, theta_cp, vtheta_cp, rate, theta2, vtheta2, f, first);
    }
  }
  else if (x->type == MAT_X) {
    mat_t_x *val = (mat_t_x *)x->val;
    int n = val->n;
    int m2 = val->m;
    for (i = 0; i < n; ++i)
      for (j = 0; j < m2; ++j) {
        val = (mat_t_x *)x->val;
        num_t_x *x2 = &(val->mat[i][j]);
        val = (mat_t_x *)v->val;
        num_t_x *v2 = &(val->mat[i][j]);
        for (m = 0; m < mold->M; ++m) {
          val = (mat_t_x *)theta[m]->val;
          theta2[m] = val->mat[i][j];
          theta_cp[m] = &(val->mat[i][j]);
          val = (mat_t_x *)vtheta[m]->val;
          vtheta2[m] = val->mat[i][j];
          vtheta_cp[m] = &(val->mat[i][j]);
        }
        val = (mat_t_x *)force->val;
        num_t_x *f = &(val->mat[i][j]);
        mold_single_update_x(mold, x2, v2, theta_cp, vtheta_cp, rate, theta2, vtheta2, f, first);
      }
  }
  else if (x->type == TENSOR_3D_X) {
    tensor_3d_t_x *val = (tensor_3d_t_x *)x->val;
    int n = val->n;
    int m2 = val->m;
    int c = val->c;
    for (i = 0; i < n; ++i)
      for (j = 0; j < m2; ++j)
        for (k = 0; k < c; ++k) {
          val = (tensor_3d_t_x *)x->val;
          num_t_x *x2 = &(val->tensor[i][j][k]);
          val = (tensor_3d_t_x *)v->val;
          num_t_x *v2 = &(val->tensor[i][j][k]);
          for (m = 0; m < mold->M; ++m) {
            val = (tensor_3d_t_x *)theta[m]->val;
            theta2[m] = val->tensor[i][j][k];
            theta_cp[m] = &(val->tensor[i][j][k]);
            val = (tensor_3d_t_x *)vtheta[m]->val;
            vtheta2[m] = val->tensor[i][j][k];
            vtheta_cp[m] = &(val->tensor[i][j][k]);
          }
          val = (tensor_3d_t_x *)force->val;
          num_t_x *f = &(val->tensor[i][j][k]);
          mold_single_update_x(mold, x2, v2, theta_cp, vtheta_cp, rate, theta2, vtheta2, f, first);
        }
  }
  free(theta2);
  free(vtheta2);
  free(theta_cp);
  free(vtheta_cp);
}

int mold_descent_worker_x(void *info, int i) {
  mold_descent_info_t_x *inf = (mold_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  mold_t_x *mold = inf->mold;
  num_t_x rate = inf->rate;
  num_t_x alpha = inf->alpha;
  int first = inf->first;
  var_t_x *x = attr->params[i]->dat;
  var_t_x *v = mold->vels[i];
  var_t_x **theta = (var_t_x **)malloc(mold->M*sizeof(var_t_x *));
  var_t_x **vtheta = (var_t_x **)malloc(mold->M*sizeof(var_t_x *));
  int m;
  for (m = 0; m < mold->M; ++m) {
    theta[m] = mold->thetas[m][i];
    vtheta[m] = mold->vthetas[m][i];
  }
  var_t_x *force = alloc_var_type_x(x);
  init_fixed_uniform_x(force, 0);
  if (attr->inputs[0]->btn) {
    for (m = 0; m < attr->inputs[0]->btn; ++m)
      var_subtract_from_x(force, attr->params[i]->bt_back_dat[m], 1.0/attr->inputs[0]->btn, 1.0);
  }
  else
    var_subtract_from_x(force, attr->params[i]->back_dat, 1.0, 1.0);
  var_mult_x(v, alpha);
  var_mult_x(force, rate);
  //var_subtract_from_x(v, force, -1.0, 1.0);
  //var_subtract_from_x(x, v, -1.0, 1.0);
  mold_var_update_x(mold, x, v, theta, vtheta, force, 1.0, first);
  //mold_var_update_x(mold, x, v, theta, vtheta, force, rate, first);
  free(theta);
  free(vtheta);
  free_var_x(force);
  return 0;
}

int mold_grad_descent_x(nn_attr_t_x *attr, mold_t_x *mold, num_t_x rate, num_t_x alpha, int first, int thread_count) {
  mold_descent_info_t_x info;
  info.attr = attr;
  info.mold = mold;
  info.rate = rate;
  info.alpha = alpha;
  info.first = first;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, mold_descent_worker_x);
}

num_t_x calc_temperature_x(int n, var_t_x **vels) {
  num_t_x res = 0;
  int count = 0;
  int i;
  for (i = 0; i < n; ++i) {
    res += var_squared_sum_x(vels[i]);
    count += var_num_elements_x(vels[i]);
  }
  return res/(2*count);
}

int rms_prop_update_x(var_t_x *r, var_t_x *g, num_t_x eps, num_t_x p, int t) {
  int i, j, k;
  if (r->type == REAL_X) {
    real_t_x *rval = (real_t_x *)r->val;
    real_t_x *gval = (real_t_x *)g->val;
    rval->real = p*rval->real+(1-p)*gval->real*gval->real;
    rval->real = rval->real/(1.0-pow(p, t));
    gval->real = (eps/(sqrt(rval->real)+1e-3))*gval->real;
  }
  else if (r->type == VEC_X) {
    vec_t_x *rval = (vec_t_x *)r->val;
    vec_t_x *gval = (vec_t_x *)g->val;
    for (i = 0; i < rval->n; ++i) {
      rval->vec[i] = p*rval->vec[i]+(1-p)*gval->vec[i]*gval->vec[i];
      rval->vec[i] = rval->vec[i]/(1.0-pow(p, t));
      gval->vec[i] = (eps/(sqrt(rval->vec[i])+1e-3))*gval->vec[i];
    }
  }
  else if (r->type == MAT_X) {
    mat_t_x *rval = (mat_t_x *)r->val;
    mat_t_x *gval = (mat_t_x *)g->val;
    for (i = 0; i < rval->n; ++i)
      for (j = 0; j < rval->m; ++j) {
        rval->mat[i][j] = p*rval->mat[i][j]+(1-p)*gval->mat[i][j]*gval->mat[i][j];
        rval->mat[i][j] = rval->mat[i][j]/(1.0-pow(p, t));
        gval->mat[i][j] = (eps/(sqrt(rval->mat[i][j])+1e-3))*gval->mat[i][j];
      }
  }
  else if (r->type == TENSOR_3D_X) {
    tensor_3d_t_x *rval = (tensor_3d_t_x *)r->val;
    tensor_3d_t_x *gval = (tensor_3d_t_x *)g->val;
    for (i = 0; i < rval->n; ++i)
      for (j = 0; j < rval->m; ++j)
        for (k = 0; k < rval->c; ++k) {
          rval->tensor[i][j][k] = p*rval->tensor[i][j][k]+(1-p)*gval->tensor[i][j][k]*gval->tensor[i][j][k];
          rval->tensor[i][j][k] = rval->tensor[i][j][k]/(1.0-pow(p, t));
          gval->tensor[i][j][k] = (eps/(sqrt(rval->tensor[i][j][k])+1e-3))*gval->tensor[i][j][k];
        }
  }
  return 0;
}

int rms_momentum_descent_worker_x(void *info, int i) {
  rms_momentum_descent_info_t_x *inf = (rms_momentum_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  grad_vel_t_x *vel = inf->vel;
  num_t_x alpha = inf->alpha;
  num_t_x rate = inf->rate;
  num_t_x p = inf->p;
  int status;
  var_t_x *force = alloc_var_type_x(vel->vels[i]);
  init_fixed_uniform_x(force, 0);
  if (attr->inputs[0]->btn) {
    int j;
    for (j = 0; j < attr->inputs[0]->btn; ++j) {
        status = var_subtract_from_x(force, attr->params[i]->bt_back_dat[j], 1.0/attr->inputs[0]->btn, 1.0);
#ifdef NN_NOCHECK_X
      if (status != 0)
        return status;
#endif
    }
  }
  else {
    status = var_subtract_from_x(force, attr->params[i]->back_dat, 1.0, 1.0);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  rms_prop_update_x(vel->rs[i], force, rate, p, vel->t);
  var_subtract_from_x(vel->vels[i], force, -1.0, alpha);
  var_subtract_from_x(attr->params[i]->dat, vel->vels[i], -1, 1.0);
  free_var_x(force);
  return 0;
}

int rms_momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, num_t_x p, int M, int thread_count) {
  rms_momentum_descent_info_t_x info;
  info.attr = attr;
  info.vel = vel;
  info.alpha = alpha;
  info.rate = rate;
  info.p = p;
  vel->t += M;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, rms_momentum_descent_worker_x);
}

int adam_update_x(var_t_x *s, var_t_x *r, var_t_x *g, num_t_x eps, num_t_x p, num_t_x p2, int t) {
  int i, j, k;
  if (r->type == REAL_X) {
    real_t_x *rval = (real_t_x *)r->val;
    real_t_x *sval = (real_t_x *)s->val;
    real_t_x *gval = (real_t_x *)g->val;
    rval->real = p*rval->real+(1-p)*gval->real*gval->real;
    rval->real = rval->real/(1.0-pow(p, t));
    sval->real = p2*sval->real+(1-p2)*gval->real;
    sval->real = sval->real/(1.0-pow(p2, t));
    gval->real = (eps/(sqrt(rval->real)+1e-8))*sval->real;
  }
  else if (r->type == VEC_X) {
    vec_t_x *rval = (vec_t_x *)r->val;
    vec_t_x *sval = (vec_t_x *)s->val;
    vec_t_x *gval = (vec_t_x *)g->val;
    for (i = 0; i < rval->n; ++i) {
      rval->vec[i] = p*rval->vec[i]+(1-p)*gval->vec[i]*gval->vec[i];
      rval->vec[i] = rval->vec[i]/(1.0-pow(p, t));
      sval->vec[i] = p2*sval->vec[i]+(1-p2)*gval->vec[i];
      sval->vec[i] = sval->vec[i]/(1.0-pow(p2, t));
      gval->vec[i] = (eps/(sqrt(rval->vec[i])+1e-8))*sval->vec[i];
    }
  }
  else if (r->type == MAT_X) {
    mat_t_x *rval = (mat_t_x *)r->val;
    mat_t_x *sval = (mat_t_x *)s->val;
    mat_t_x *gval = (mat_t_x *)g->val;
    for (i = 0; i < rval->n; ++i)
      for (j = 0; j < rval->m; ++j) {
        rval->mat[i][j] = p*rval->mat[i][j]+(1-p)*gval->mat[i][j]*gval->mat[i][j];
        rval->mat[i][j] = rval->mat[i][j]/(1.0-pow(p, t));
        sval->mat[i][j] = p2*sval->mat[i][j]+(1-p2)*gval->mat[i][j];
        sval->mat[i][j] = sval->mat[i][j]/(1.0-pow(p2, t));
        gval->mat[i][j] = (eps/(sqrt(rval->mat[i][j])+1e-8))*sval->mat[i][j];
      }
  }
  else if (r->type == TENSOR_3D_X) {
    tensor_3d_t_x *rval = (tensor_3d_t_x *)r->val;
    tensor_3d_t_x *sval = (tensor_3d_t_x *)s->val;
    tensor_3d_t_x *gval = (tensor_3d_t_x *)g->val;
    for (i = 0; i < rval->n; ++i)
      for (j = 0; j < rval->m; ++j)
        for (k = 0; k < rval->c; ++k) {
          rval->tensor[i][j][k] = p*rval->tensor[i][j][k]+(1-p)*gval->tensor[i][j][k]*gval->tensor[i][j][k];
          rval->tensor[i][j][k] = rval->tensor[i][j][k]/(1.0-pow(p, t));
          sval->tensor[i][j][k] = p2*sval->tensor[i][j][k]+(1-p2)*gval->tensor[i][j][k];
          sval->tensor[i][j][k] = sval->tensor[i][j][k]/(1.0-pow(p2, t));
          gval->tensor[i][j][k] = (eps/(sqrt(rval->tensor[i][j][k])+1e-8))*sval->tensor[i][j][k];
        }
  }
  return 0;
}

int ada_momentum_descent_worker_x(void *info, int i) {
  ada_momentum_descent_info_t_x *inf = (ada_momentum_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  grad_vel_t_x *vel = inf->vel;
  num_t_x alpha = inf->alpha;
  num_t_x rate = inf->rate;
  num_t_x p = inf->p;
  num_t_x p2 = inf->p2;
  int status;
  var_t_x *force = alloc_var_type_x(vel->vels[i]);
  init_fixed_uniform_x(force, 0);
  if (attr->inputs[0]->btn) {
    int j;
    for (j = 0; j < attr->inputs[0]->btn; ++j) {
        status = var_subtract_from_x(force, attr->params[i]->bt_back_dat[j], 1.0/attr->inputs[0]->btn, 1.0);
#ifdef NN_NOCHECK_X
      if (status != 0)
        return status;
#endif
    }
  }
  else {
    status = var_subtract_from_x(force, attr->params[i]->back_dat, 1.0, 1.0);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  adam_update_x(vel->ss[i], vel->rs[i], force, rate, p, p2, vel->t);
  var_subtract_from_x(vel->vels[i], force, -1.0, alpha);
  var_subtract_from_x(attr->params[i]->dat, vel->vels[i], -1, 1.0);
  //var_subtract_from_x(attr->params[i]->dat, force, -1, 1.0);
  free_var_x(force);
  return 0;
}

int ada_momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, num_t_x p, num_t_x p2, int M, int thread_count) {
  ada_momentum_descent_info_t_x info;
  info.attr = attr;
  info.vel = vel;
  info.alpha = alpha;
  info.rate = rate;
  info.p = p;
  info.p2 = p2;
  vel->t += M;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, ada_momentum_descent_worker_x);
}

int ada_mold_descent_worker_x(void *info, int i) {
  ada_mold_descent_info_t_x *inf = (ada_mold_descent_info_t_x *)info;
  nn_attr_t_x *attr = inf->attr;
  mold_t_x *mold = inf->mold;
  num_t_x alpha = inf->alpha;
  num_t_x rate = inf->rate;
  num_t_x p = inf->p;
  //num_t_x p2 = inf->p2;
  int first = inf->first;
  var_t_x *x = attr->params[i]->dat;
  var_t_x *v = mold->vels[i];
  var_t_x **theta = (var_t_x **)malloc(mold->M*sizeof(var_t_x *));
  var_t_x **vtheta = (var_t_x **)malloc(mold->M*sizeof(var_t_x *));
  int m;
  for (m = 0; m < mold->M; ++m) {
    theta[m] = mold->thetas[m][i];
    vtheta[m] = mold->vthetas[m][i];
  }
  int status;
  var_t_x *force = alloc_var_type_x(mold->vels[i]);
  init_fixed_uniform_x(force, 0);
  if (attr->inputs[0]->btn) {
    int j;
    for (j = 0; j < attr->inputs[0]->btn; ++j) {
        status = var_subtract_from_x(force, attr->params[i]->bt_back_dat[j], 1.0/attr->inputs[0]->btn, 1.0);
#ifdef NN_NOCHECK_X
      if (status != 0)
        return status;
#endif
    }
  }
  else {
    status = var_subtract_from_x(force, attr->params[i]->back_dat, 1.0, 1.0);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  //adam_update_x(mold->ss[i], mold->rs[i], force, rate, p, p2, mold->t);
  rms_prop_update_x(mold->rs[i], force, rate, p, mold->t);
  var_mult_x(v, alpha);
  mold_var_update_x(mold, x, v, theta, vtheta, force, 1.0, first);
  //var_mult_x(v, alpha);
  //var_subtract_from_x(mold->vels[i], force, -1.0, alpha);
  //var_subtract_from_x(attr->params[i]->dat, mold->vels[i], -1, 1.0);
  //var_subtract_from_x(attr->params[i]->dat, force, -1, 1.0);
  free(theta);
  free(vtheta);
  free_var_x(force);
  return 0;
}

int ada_mold_grad_descent_x(nn_attr_t_x *attr, mold_t_x *mold, num_t_x rate, num_t_x alpha, num_t_x p, num_t_x p2, int M, int first, int thread_count) {
  ada_mold_descent_info_t_x info;
  info.attr = attr;
  info.mold = mold;
  info.alpha = alpha;
  info.rate = rate;
  info.p = p;
  info.p2 = p2;
  info.first = first;
  mold->t += M;
  return mult_thread_prop_x(&info, attr->paramn, thread_count, ada_mold_descent_worker_x);
}