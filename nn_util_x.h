#ifndef NN_UTIL_X_H
#define NN_UTIL_X_H

#include "nn_variant_type_x.h"
#include <stdio.h>

var_t_x *alloc_real_x(num_t_x num);
var_t_x *alloc_vec_x(int n);
var_t_x *alloc_mat_x(int n, int m);
var_t_x *alloc_tensor_3d_x(int n, int m, int c);
void free_var_x(var_t_x *pt);
int fprint_var_x(FILE *out, var_t_x *pt);

void set_seed_x(unsigned int seed);
num_t_x random_uniform_x(num_t_x a, num_t_x b);
void init_random_uniform_x(var_t_x *dat, num_t_x a, num_t_x b);
void init_fixed_uniform_x(var_t_x *dat, num_t_x a);
var_t_x *alloc_var_type_x(var_t_x *dat);
num_t_x var_squared_sum_x(var_t_x *dat);
int var_num_elements_x(var_t_x *dat);
var_t_x *copy_var_type_x(var_t_x *dat);

#endif