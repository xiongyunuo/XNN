#ifndef NN_DATASET_X_H
#define NN_DATASET_X_H

#include "nn_variant_type_x.h"

typedef var_t_x *(*get_dat_func_t_x)(int index);

typedef struct {
  int num;
  var_t_x **dat_set;
  get_dat_func_t_x get;
} dat_set_t_x;

dat_set_t_x *alloc_dat_set_x(int num, get_dat_func_t_x get);
var_t_x *get_dat_set_entry_x(dat_set_t_x *set, int index);
void clear_dat_set_entry_x(dat_set_t_x *set, int index);
void clear_dat_set_x(dat_set_t_x *set);
void free_dat_set_x(dat_set_t_x *set);

#endif