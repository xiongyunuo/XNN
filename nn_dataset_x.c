#include "nn_dataset_x.h"
#include "nn_util_x.h"
#include <stdlib.h>

dat_set_t_x *alloc_dat_set_x(int num, get_dat_func_t_x get) {
  dat_set_t_x *set = (dat_set_t_x *)malloc(sizeof(dat_set_t_x));
  if (set == NULL)
    return NULL;
  set->num = num;
  set->get = get;
  set->dat_set = (var_t_x **)malloc(num*sizeof(var_t_x *));
  if (set->dat_set == NULL) {
    free(set);
    return NULL;
  }
  int i;
  for (i = 0; i < num; ++i)
    set->dat_set[i] = NULL;
  return set;
}

var_t_x *get_dat_set_entry_x(dat_set_t_x *set, int index) {
  if (set->dat_set[index] != NULL)
    return set->dat_set[index];
  set->dat_set[index] = set->get(index);
  return set->dat_set[index];
}

void clear_dat_set_entry_x(dat_set_t_x *set, int index) {
  if (set->dat_set[index] != NULL) {
    free_var_x(set->dat_set[index]);
    set->dat_set[index] = NULL;
  }
}

void clear_dat_set_x(dat_set_t_x *set) {
  int i;
  for (i = 0; i < set->num; ++i)
    clear_dat_set_entry_x(set, i);
}

void free_dat_set_x(dat_set_t_x *set) {
  clear_dat_set_x(set);
  free(set->dat_set);
  free(set);
}