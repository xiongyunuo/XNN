#ifndef NN_ARCH_X_H
#define NN_ARCH_X_H

#include "nn_variant_type_x.h"

typedef int (*op_func_t_x)(int, var_t_x **, var_t_x **, void *);
typedef int (*deri_op_func_t_x)(int, var_t_x **, var_t_x *, var_t_x *, var_t_x ***, void *);

typedef struct nn_node_x {
  int inn;
  struct nn_node_x **ins;
  int outn;
  struct nn_node_x **outs;
  var_t_x *dat;
  var_t_x *back_dat;
  op_func_t_x op;
  deri_op_func_t_x back_op;
  void *attr;
  int ready;
  int btn;
  var_t_x **bt_dat;
  var_t_x **bt_back_dat;
  int *bt_ready;
  void **bt_attr;
} nn_node_t_x;

typedef struct {
  int inputn;
  nn_node_t_x **inputs;
  int outputn;
  nn_node_t_x **outputs;
  int paramn;
  nn_node_t_x **params;
  int intermn;
  nn_node_t_x **interms;
} nn_attr_t_x;

enum { INPUT_NODE_X, OUTPUT_NODE_X, PARAM_NODE_X, INTERM_NODE_X };

nn_node_t_x *alloc_nn_node_x();
nn_attr_t_x *alloc_nn_attr_x();
int nn_connect_x(int n, nn_node_t_x **in, nn_node_t_x *out, op_func_t_x op, deri_op_func_t_x op2, void *attr);
int nn_add_node_x(nn_node_t_x *node, nn_attr_t_x *attr, int type);
int nn_forward_prop_node_x(nn_node_t_x *node, int bt);
int nn_forward_prop_x(nn_attr_t_x *attr, int thread_count);
void nn_forward_reset_x(nn_attr_t_x *attr);
int nn_backward_prop_node_x(nn_node_t_x *node, int bt);
void nn_backward_reset_x(nn_attr_t_x *attr);
int nn_get_complete(nn_attr_t_x *attr);
int nn_backward_prop_x(nn_node_t_x *node, int thread_count);
int nn_set_batch(nn_attr_t_x *attr, int bt);
void nn_free_x(nn_attr_t_x *attr);
void nn_free_node_x(nn_node_t_x *node, int bt);

typedef int (*prop_func_t_x)(nn_node_t_x *, int);

typedef struct {
  int a, b;
  int status;
  nn_node_t_x *node;
  prop_func_t_x func;
} prop_worker_t_x;

int mult_thread_prop_x(nn_node_t_x *node, int btn, int thread_count, prop_func_t_x func);
void *prop_worker_x(void *dat);

#endif