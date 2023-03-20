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

#endif