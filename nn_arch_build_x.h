#ifndef NN_ARCH_BUILD_X_H
#define NN_ARCH_BUILD_X_H

#include "nn_arch_x.h"
#include "nn_variant_type_x.h"

int nn_add_affine_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, int d1, int d2, nn_node_t_x **p1, nn_node_t_x **p2, nn_node_t_x **out);
int nn_add_relu_layer_x(nn_attr_t_x *attr, nn_node_t_x *node, nn_node_t_x **out);
int nn_add_softmax_classifier_x(nn_attr_t_x *attr, nn_node_t_x *node, int type, nn_node_t_x **out);

void set_seed_x(unsigned int seed);
num_t_x random_uniform_x(num_t_x a, num_t_x b);
void init_random_uniform_x(var_t_x *dat, num_t_x a, num_t_x b);
void init_fixed_uniform_x(var_t_x *dat, num_t_x a);

int var_subtract_from_x(var_t_x *op1, var_t_x *op2, num_t_x mult);
int gradient_descent_x(nn_attr_t_x *attr, num_t_x rate);

#endif