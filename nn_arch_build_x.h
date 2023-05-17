#ifndef NN_ARCH_BUILD_X_H
#define NN_ARCH_BUILD_X_H

#include "nn_arch_x.h"
#include "nn_variant_type_x.h"

int nn_add_affine_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, int d1, int d2, nn_node_t_x **p1, nn_node_t_x **p2, nn_node_t_x **out);
int nn_add_relu_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, nn_node_t_x **out);
int nn_add_softmax_classifier_x(nn_attr_t_x *attr, nn_node_t_x *node, int type, nn_node_t_x **out);

int var_subtract_from_x(var_t_x *op1, var_t_x *op2, num_t_x mult, num_t_x mult2);
int gradient_descent_x(nn_attr_t_x *attr, num_t_x rate, int thread_count);

typedef struct {
  int n;
  int t;
  var_t_x **vels;
  var_t_x **rs;
  var_t_x **ss;
} grad_vel_t_x;

typedef struct {
  nn_attr_t_x *attr;
  grad_vel_t_x *vel;
  num_t_x rate;
  num_t_x alpha;
} momentum_descent_info_t_x;

typedef struct {
  nn_attr_t_x *attr;
  grad_vel_t_x *vel;
  num_t_x rate;
  num_t_x alpha;
  num_t_x p;
} rms_momentum_descent_info_t_x;

typedef struct {
  nn_attr_t_x *attr;
  grad_vel_t_x *vel;
  num_t_x rate;
  num_t_x alpha;
  num_t_x p;
  num_t_x p2;
} ada_momentum_descent_info_t_x;

typedef struct {
  nn_attr_t_x *attr;
  num_t_x rate;
} grad_descent_info_t_x;

typedef struct {
  int n;
  int t;
  var_t_x **vels;
  var_t_x **rs;
  var_t_x **ss;
  int M;
  num_t_x T;
  num_t_x *Q;
  var_t_x ***vthetas;
  var_t_x ***thetas;
} mold_t_x;

typedef struct {
  nn_attr_t_x *attr;
  num_t_x rate;
  num_t_x alpha;
  mold_t_x *mold;
  int first;
} mold_descent_info_t_x;

typedef struct {
  nn_attr_t_x *attr;
  mold_t_x *mold;
  num_t_x rate;
  num_t_x alpha;
  num_t_x p;
  num_t_x p2;
  int first;
} ada_mold_descent_info_t_x;

void mold_single_update_x(mold_t_x *mold, num_t_x *x, num_t_x *v, num_t_x **theta_cp, num_t_x **vtheta_cp, num_t_x rate, num_t_x *theta, num_t_x *vtheta, num_t_x *force, int first);
void mold_var_update_x(mold_t_x *mold, var_t_x *x, var_t_x *v, var_t_x **theta, var_t_x **vtheta, var_t_x *force, num_t_x rate, int first);

int grad_descent_worker_x(void *info, int i);
int momentum_descent_worker_x(void *info, int i);
int mold_descent_worker_x(void *info, int i);
int rms_momentum_descent_worker_x(void *info, int i);
int ada_momentum_descent_worker_x(void *info, int i);
int ada_mold_descent_worker_x(void *info, int i);

grad_vel_t_x *alloc_grad_vel_x(nn_attr_t_x *attr);
int momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, int thread_count);
void free_grad_vel_x(grad_vel_t_x *vel);
int var_mult_x(var_t_x *op, num_t_x mult);
mold_t_x *alloc_mold_x(nn_attr_t_x *attr, int M, num_t_x T, num_t_x Q);
void free_mold_x(mold_t_x *vel);
void calc_NH_force_x(num_t_x m, int f, num_t_x *v, num_t_x *Q, num_t_x *vtheta, num_t_x *res, int M, num_t_x beta);
void NH_update_1_x(int M, num_t_x *Q, num_t_x *theta, num_t_x *vtheta, num_t_x *x, num_t_x *v, num_t_x *force, num_t_x beta, num_t_x h);
void NH_update_2_x(int M, num_t_x *Q, num_t_x *theta, num_t_x *vtheta, num_t_x *x, num_t_x *v, num_t_x *force, num_t_x beta, num_t_x h);
int mold_grad_descent_x(nn_attr_t_x *attr, mold_t_x *mold, num_t_x rate, num_t_x alpha, int first, int thread_count);
num_t_x calc_temperature_x(int n, var_t_x **vels);

int rms_prop_update_x(var_t_x *r, var_t_x *g, num_t_x eps, num_t_x p, int t);
int rms_momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, num_t_x p, int M, int thread_count);
int adam_update_x(var_t_x *s, var_t_x *r, var_t_x *g, num_t_x eps, num_t_x p, num_t_x p2, int t);
int ada_momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, num_t_x p, num_t_x p2, int M, int thread_count);
int ada_mold_grad_descent_x(nn_attr_t_x *attr, mold_t_x *mold, num_t_x rate, num_t_x alpha, num_t_x p, num_t_x p2, int M, int first, int thread_count);

typedef struct {
  void *info;
  prop_func_t_x prop_func;
  nn_attr_t_x *unn_attr;
  nn_attr_t_x *attr;
  num_t_x udecay, umin;
} uncertain_descent_info_t_x;

num_t_x interval_proportion_x(num_t_x a, num_t_x b, num_t_x x);
void uncertain_var_update_x(var_t_x *var, var_t_x *cvar, var_t_x *uvar, num_t_x udecay, num_t_x umin);
int uncertain_descent_worker_x(void *info, int i);
int uncertain_grad_descent_x(nn_attr_t_x *attr, nn_attr_t_x *unn_attr, void *info2, prop_func_t_x func, num_t_x udecay, num_t_x umin, int thread_count);

#endif