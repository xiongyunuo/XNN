#ifndef NN_OPERATION_X_H
#define NN_OPERATION_X_H

#include "nn_variant_type_x.h"

#define WRONG_ARG_NUM_X 1
#define MISMATCH_DIMENSION_X 2
#define MEMORY_ALLOC_FAILURE_X 3
#define WRONG_VAR_TYPE_X 4
#define NULL_ARG_X 5

#ifdef __GNUC__
  #define UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
  #define UNUSED(x) UNUSED_ ## x
#endif

typedef num_t_x (*activate_func_t_x)(num_t_x, void *);

int var_activate_x(int n, var_t_x **ops, var_t_x **res, void *attr, activate_func_t_x acti);
int deri_var_activate_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr, activate_func_t_x deri_acti);
num_t_x relu_activate_x(num_t_x num, void *attr);
num_t_x deri_relu_activate_x(num_t_x num, void *attr);

const char *nn_op_err_msg(int status);
int mat_mult_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int mat_vec_mult_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int vec_mat_mult_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int vec_inner_dot_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int var_relu_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int vec_add_x(int n, var_t_x **ops, var_t_x **res, void *attr);
int neg_log_softmax_x(int n, var_t_x **ops, var_t_x **res, void *attr);

int deri_vec_inner_dot_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_mat_vec_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_vec_mat_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_mat_mult_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_var_relu_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_vec_add_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);
int deri_neg_log_softmax_x(int n, var_t_x **ins, var_t_x *out, var_t_x *out_deri, var_t_x ***deri, void *attr);

void *copy_softmax_attr_x(void *arg);

#endif