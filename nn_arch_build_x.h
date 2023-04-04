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
  var_t_x **vels;
} grad_vel_t_x;

typedef struct {
  nn_attr_t_x *attr;
  grad_vel_t_x *vel;
  num_t_x rate;
  num_t_x alpha;
} momentum_descent_info_t_x;

typedef struct {
  nn_attr_t_x *attr;
  num_t_x rate;
} grad_descent_info_t_x;

int grad_descent_worker_x(void *info, int i);
int momentum_descent_worker_x(void *info, int i);

grad_vel_t_x *alloc_grad_vel_x(nn_attr_t_x *attr);
int momentum_grad_descent_x(nn_attr_t_x *attr, grad_vel_t_x *vel, num_t_x rate, num_t_x alpha, int thread_count);
void free_grad_vel_x(grad_vel_t_x *vel);
int var_mult_x(var_t_x *op, num_t_x mult);

#endif