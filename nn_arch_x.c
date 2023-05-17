#include "nn_arch_x.h"
#include "nn_util_x.h"
#include <stdlib.h>
#include <pthread.h>

int nn_forward_prop_node_x(nn_node_t_x *node, int bt) {
  var_t_x **input = (var_t_x **)malloc(node->inn*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
  if (input == NULL)
    return -1;
#endif
  int i;
  int ready;
  for (i = 0; i < node->inn; ++i) {
    if (node->ins[i]->btn)
      ready = node->ins[i]->bt_ready[bt];
    else
      ready = node->ins[i]->ready;
    if (!ready) {
      free(input);
      return 0;
    }
    if (node->ins[i]->btn)
      input[i] = node->ins[i]->bt_dat[bt];
    else
      input[i] = node->ins[i]->dat;
  }
  int status;
  if (node->btn)
    ready = node->bt_ready[bt];
  else
    ready = node->ready;
  if (!ready && node->op != NULL) {
    if (node->btn)
      status = node->op(node->inn, input, &node->bt_dat[bt], node->bt_attr[bt]);
    else
      status = node->op(node->inn, input, &node->dat, node->attr);
#ifdef NN_NOCHECK_X
    if (status != 0) {
      free(input);
      return status;
    }
#endif
    if (node->btn)
      node->bt_ready[bt] = 1;
    else
      node->ready = 1;
  }
  for (i = 0; i < node->outn; ++i) {
    status = nn_forward_prop_node_x(node->outs[i], bt);
#ifdef NN_NOCHECK_X
    if (status != 0) {
      free(input);
      return status;
    }
#endif
  }
  free(input);
  return 0;
}

nn_node_t_x *alloc_nn_node_x() {
  nn_node_t_x *res = (nn_node_t_x *)malloc(sizeof(nn_node_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->attr = NULL;
  res->dat = NULL;
  res->back_dat = NULL;
  res->inn = 0;
  res->ins = NULL;
  res->outn = 0;
  res->outs = NULL;
  res->op = NULL;
  res->back_op = NULL;
  res->ready = 0;
  res->btn = 0;
  res->bt_dat = NULL;
  res->bt_back_dat = NULL;
  res->bt_ready = NULL;
  res->bt_attr = NULL;
  res->attr_cp = NULL;
  return res;
}

nn_attr_t_x *alloc_nn_attr_x() {
  nn_attr_t_x *res = (nn_attr_t_x *)malloc(sizeof(nn_attr_t_x));
#ifdef NN_NOCHECK_X
  if (res == NULL)
    return NULL;
#endif
  res->inputn = 0;
  res->inputs = NULL;
  res->outputn = 0;
  res->outputs = NULL;
  res->paramn = 0;
  res->params = NULL;
  res->intermn = 0;
  res->interms = NULL;
  return res;
}

int nn_connect_x(int n, nn_node_t_x **in, nn_node_t_x *out, op_func_t_x op, deri_op_func_t_x op2, void *attr) {
  int i;
  for (i = 0; i < n; ++i) {
    in[i]->outn++;
    in[i]->outs = (nn_node_t_x **)realloc((void *)in[i]->outs, in[i]->outn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (in[i]->outs == NULL)
      return -1;
#endif
    in[i]->outs[in[i]->outn-1] = out;
    out->inn++;
    out->ins = (nn_node_t_x **)realloc((void *)out->ins, out->inn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (out->ins == NULL)
      return -1;
#endif
    out->ins[out->inn-1] = in[i];
  }
  out->op = op;
  out->back_op = op2;
  out->attr = attr;
  return 0;
}

int nn_add_node_x(nn_node_t_x *node, nn_attr_t_x *attr, int type) {
  if (type == INPUT_NODE_X) {
    attr->inputn++;
    attr->inputs = (nn_node_t_x **)realloc((void *)attr->inputs, attr->inputn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->inputs == NULL)
      return -1;
#endif
    attr->inputs[attr->inputn-1] = node;
  }
  else if (type == OUTPUT_NODE_X) {
    attr->outputn++;
    attr->outputs = (nn_node_t_x **)realloc((void *)attr->outputs, attr->outputn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->outputs == NULL)
      return -1;
#endif
    attr->outputs[attr->outputn-1] = node;
  }
  else if (type == PARAM_NODE_X) {
    attr->paramn++;
    attr->params = (nn_node_t_x **)realloc((void *)attr->params, attr->paramn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->params == NULL)
      return -1;
#endif
    attr->params[attr->paramn-1] = node;
  }
  else if (type == INTERM_NODE_X) {
    attr->intermn++;
    attr->interms = (nn_node_t_x **)realloc((void *)attr->interms, attr->intermn*sizeof(nn_node_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->interms == NULL)
      return -1;
#endif
    attr->interms[attr->intermn-1] = node;
  }
  return 0;
}

int nn_get_complete(nn_attr_t_x *attr) {
  int ready = 1;
  int j, k;
  for (j = 0; j < attr->outputn; ++j) {
    if (attr->outputs[j]->btn) {
      for (k = 0; k < attr->outputs[j]->btn; ++k) {
        ready = attr->outputs[j]->bt_ready[k];
        if (!ready)
          return 0;
      }
    }
    else
      ready = attr->outputs[j]->ready;
    if (!ready)
      return 0;
  }
  return 1;
}

int nn_forward_prop_x(nn_attr_t_x *attr, int thread_count) {
  int complete, status;
  int i;
  for (i = 0; i < attr->inputn; ++i) {
    complete = nn_get_complete(attr);
    if (complete)
      return 0;
    else {
      if (attr->inputs[0]->btn) {
        status = mult_thread_prop_x(attr->inputs[i], attr->inputs[0]->btn, thread_count, nn_forward_prop_node_wrap_x);
#ifdef NN_NOCHECK_X
        if (status != 0)
          return status;
#endif
      }
      else {
        status = nn_forward_prop_node_x(attr->inputs[i], 0);
#ifdef NN_NOCHECK_X
        if (status != 0)
          return status;
#endif
      }
    }
  }
  for (i = 0; i < attr->paramn; ++i) {
    complete = nn_get_complete(attr);
    if (complete)
      return 0;
    else {
      if (attr->inputs[0]->btn) {
        status = mult_thread_prop_x(attr->params[i], attr->inputs[0]->btn, thread_count, nn_forward_prop_node_wrap_x);
#ifdef NN_NOCHECK_X
        if (status != 0)
          return status;
#endif
      }
      else {
        status = nn_forward_prop_node_x(attr->params[i], 0);
#ifdef NN_NOCHECK_X
        if (status != 0)
          return status;
#endif
      }
    }
  }
  complete = nn_get_complete(attr);
  if (complete)
    return 0;
  else
    return -1;
}

void nn_forward_reset_x(nn_attr_t_x *attr) {
  int i, j;
  for (i = 0; i < attr->inputn; ++i) {
    attr->inputs[i]->ready = 1;
    if (attr->inputs[i]->btn) {
      for (j = 0; j < attr->inputs[i]->btn; ++j)
        attr->inputs[i]->bt_ready[j] = 1;
    }
  }
  for (i = 0; i < attr->paramn; ++i)
    attr->params[i]->ready = 1;
  for (i = 0; i < attr->outputn; ++i) {
    attr->outputs[i]->ready = 0;
#ifdef NN_NOCLEAR_X
    if (attr->outputs[i]->dat != NULL) {
      free_var_x(attr->outputs[i]->dat);
      attr->outputs[i]->dat = NULL;
    }
#endif
    if (attr->outputs[i]->btn) {
      for (j = 0; j < attr->outputs[i]->btn; ++j) {
        attr->outputs[i]->bt_ready[j] = 0;
#ifdef NN_NOCLEAR_X
        if (attr->outputs[i]->bt_dat[j] != NULL) {
          free_var_x(attr->outputs[i]->bt_dat[j]);
          attr->outputs[i]->bt_dat[j] = NULL;
        }
#endif
      }
    }
  }
  for (i = 0; i < attr->intermn; ++i) {
    attr->interms[i]->ready = 0;
#ifdef NN_NOCLEAR_X
    if (attr->interms[i]->dat != NULL) {
      free_var_x(attr->interms[i]->dat);
      attr->interms[i]->dat = NULL;
    }
#endif
    if (attr->interms[i]->btn) {
      for (j = 0; j < attr->interms[i]->btn; ++j) {
        attr->interms[i]->bt_ready[j] = 0;
#ifdef NN_NOCLEAR_X
        if (attr->interms[i]->bt_dat[j] != NULL) {
          free_var_x(attr->interms[i]->bt_dat[j]);
          attr->interms[i]->bt_dat[j] = NULL;
        }
#endif
      }
    }
  }
}

int nn_backward_prop_node_x(nn_node_t_x *node, int bt) {
  var_t_x **input = (var_t_x **)malloc(node->inn*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
  if (input == NULL)
    return -1;
#endif
  int i;
  int ready = 1;
  for (i = 0; i < node->outn; ++i) {
    if (node->outs[i]->btn)
      ready = node->outs[i]->bt_ready[bt];
    else
      ready = node->outs[i]->ready;
    if (!ready) {
      free(input);
      return 0;
    }
  }
  for (i = 0; i < node->inn; ++i) {
    if (node->ins[i]->btn)
      input[i] = node->ins[i]->bt_dat[bt];
    else
      input[i] = node->ins[i]->dat;
  }
  var_t_x ***deri = (var_t_x ***)malloc(node->inn*sizeof(var_t_x **));
#ifdef NN_NOCHECK_X
  if (deri == NULL) {
    free(input);
    return -1;
  }
#endif
  for (i = 0; i < node->inn; ++i) {
    if (node->btn)
      deri[i] = &node->ins[i]->bt_back_dat[bt];
    else
      deri[i] = &node->ins[i]->back_dat;
  }
  int status;
  if (node->btn)
    ready = node->bt_ready[bt];
  else
    ready = node->ready;
  if (!ready && node->back_op != NULL) {
    if (node->btn)
      status = node->back_op(node->inn, input, node->bt_dat[bt], node->bt_back_dat[bt], deri, node->bt_attr[bt]);
    else
      status = node->back_op(node->inn, input, node->dat, node->back_dat, deri, node->attr);
#ifdef NN_NOCHECK_X
    if (status != 0) {
      free(input);
      free(deri);
      return status;
    }
#endif
    if (node->btn)
      node->bt_ready[bt] = 1;
    else
      node->ready = 1;
  }
  for (i = 0; i < node->inn; ++i) {
    status = nn_backward_prop_node_x(node->ins[i], bt);
#ifdef NN_NOCHECK_X
    if (status != 0) {
      free(input);
      free(deri);
      return status;
    }
#endif
  }
  free(input);
  free(deri);
  return 0;
}

void nn_backward_reset_x(nn_attr_t_x *attr) {
  int i, j;
  for (i = 0; i < attr->inputn; ++i) {
    attr->inputs[i]->ready = 0;
    if (attr->inputs[i]->back_dat != NULL) {
#ifdef NN_NOCLEAR_X
      free_var_x(attr->inputs[i]->back_dat);
      attr->inputs[i]->back_dat = NULL;
#else
      init_fixed_uniform_x(attr->inputs[i]->back_dat, 0);
#endif
    }
    if (attr->inputs[i]->btn) {
      for (j = 0; j < attr->inputs[i]->btn; ++j) {
        attr->inputs[i]->bt_ready[j] = 0;
        if (attr->inputs[i]->bt_back_dat[j] != NULL) {
#ifdef NN_NOCLEAR_X
          free_var_x(attr->inputs[i]->bt_back_dat[j]);
          attr->inputs[i]->bt_back_dat[j] = NULL;
#else
          init_fixed_uniform_x(attr->inputs[i]->bt_back_dat[j], 0);
#endif
        }
      }
    }
  }
  for (i = 0; i < attr->paramn; ++i) {
    attr->params[i]->ready = 0;
    if (attr->params[i]->back_dat != NULL) {
#ifdef NN_NOCLEAR_X
      free_var_x(attr->params[i]->back_dat);
      attr->params[i]->back_dat = NULL;
#else
      init_fixed_uniform_x(attr->params[i]->back_dat, 0);
#endif
    }
    if (attr->inputs[0]->btn) {
      for (j = 0; j < attr->inputs[0]->btn; ++j) {
        if (attr->params[i]->bt_back_dat[j] != NULL) {
#ifdef NN_NOCLEAR_X
          free_var_x(attr->params[i]->bt_back_dat[j]);
          attr->params[i]->bt_back_dat[j] = NULL;
#else
          init_fixed_uniform_x(attr->params[i]->bt_back_dat[j], 0);
#endif
        }
      }
    }
  }
  for (i = 0; i < attr->outputn; ++i) {
    attr->outputs[i]->ready = 0;
    if (attr->outputs[i]->back_dat != NULL) {
      free_var_x(attr->outputs[i]->back_dat);
      attr->outputs[i]->back_dat = NULL;
    }
    if (attr->outputs[i]->btn) {
      for (j = 0; j < attr->outputs[i]->btn; ++j) {
        attr->outputs[i]->bt_ready[j] = 0;
        if (attr->outputs[i]->bt_back_dat[j] != NULL) {
          free_var_x(attr->outputs[i]->bt_back_dat[j]);
          attr->outputs[i]->bt_back_dat[j] = NULL;
        }
      }
    }
  }
  for (i = 0; i < attr->intermn; ++i) {
    attr->interms[i]->ready = 0;
    if (attr->interms[i]->back_dat != NULL) {
#ifdef NN_NOCLEAR_X
      free_var_x(attr->interms[i]->back_dat);
      attr->interms[i]->back_dat = NULL;
#else
      init_fixed_uniform_x(attr->interms[i]->back_dat, 0);
#endif
    }
    if (attr->interms[i]->btn) {
      for (j = 0; j < attr->interms[i]->btn; ++j) {
        attr->interms[i]->bt_ready[j] = 0;
        if (attr->interms[i]->bt_back_dat[j] != NULL) {
#ifdef NN_NOCLEAR_X
          free_var_x(attr->interms[i]->bt_back_dat[j]);
          attr->interms[i]->bt_back_dat[j] = NULL;
#else
          init_fixed_uniform_x(attr->interms[i]->bt_back_dat[j], 0);
#endif
        }
      }
    }
  }
}

int nn_backward_prop_x(nn_node_t_x *node, int thread_count) {
  int status;
  if (node->btn) {
    status = mult_thread_prop_x(node, node->btn, thread_count, nn_backward_prop_node_wrap_x);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  else {
    status = nn_backward_prop_node_x(node, 0);
#ifdef NN_NOCHECK_X
    if (status != 0)
      return status;
#endif
  }
  return 0;
}

int nn_set_batch(nn_attr_t_x *attr, int bt) {
  int i, j;
  for (i = 0; i < attr->inputn; ++i) {
    attr->inputs[i]->btn = bt;
    attr->inputs[i]->bt_ready = (int *)realloc(attr->inputs[i]->bt_ready, bt*sizeof(int));
#ifdef NN_NOCHECK_X
    if (attr->inputs[i]->bt_ready == NULL)
      return -1;
#endif
    attr->inputs[i]->bt_dat = (var_t_x **)realloc(attr->inputs[i]->bt_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->inputs[i]->bt_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->inputs[i]->bt_dat[j] = NULL;
    attr->inputs[i]->bt_back_dat = (var_t_x **)realloc(attr->inputs[i]->bt_back_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->inputs[i]->bt_back_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->inputs[i]->bt_back_dat[j] = NULL;
  }
  for (i = 0; i < attr->paramn; ++i) {
    attr->params[i]->bt_back_dat = (var_t_x **)realloc(attr->params[i]->bt_back_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->params[i]->bt_back_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->params[i]->bt_back_dat[j] = NULL;
  }
  for (i = 0; i < attr->intermn; ++i) {
    for (j = 0; j < attr->interms[i]->btn; ++j)
      free(attr->interms[i]->bt_attr[j]);
    attr->interms[i]->btn = bt;
    attr->interms[i]->bt_ready = (int *)realloc(attr->interms[i]->bt_ready, bt*sizeof(int));
#ifdef NN_NOCHECK_X
    if (attr->interms[i]->bt_ready == NULL)
      return -1;
#endif
    attr->interms[i]->bt_dat = (var_t_x **)realloc(attr->interms[i]->bt_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->interms[i]->bt_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->interms[i]->bt_dat[j] = NULL;
    attr->interms[i]->bt_back_dat = (var_t_x **)realloc(attr->interms[i]->bt_back_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->interms[i]->bt_back_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->interms[i]->bt_back_dat[j] = NULL;
    attr->interms[i]->bt_attr = (void **)realloc(attr->interms[i]->bt_attr, bt*sizeof(void *));
#ifdef NN_NOCHECK_X
    if (attr->interms[i]->bt_attr == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j) {
      if (attr->interms[i]->attr_cp != NULL)
        attr->interms[i]->bt_attr[j] = attr->interms[i]->attr_cp(attr->interms[i]->attr);
      else
        attr->interms[i]->bt_attr[j] = NULL;
    }
  }
  for (i = 0; i < attr->outputn; ++i) {
    for (j = 0; j < attr->outputs[i]->btn; ++j)
      free(attr->outputs[i]->bt_attr[j]);
    attr->outputs[i]->btn = bt;
    attr->outputs[i]->bt_ready = (int *)realloc(attr->outputs[i]->bt_ready, bt*sizeof(int));
#ifdef NN_NOCHECK_X
    if (attr->outputs[i]->bt_ready == NULL)
      return -1;
#endif
    attr->outputs[i]->bt_dat = (var_t_x **)realloc(attr->outputs[i]->bt_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->outputs[i]->bt_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->outputs[i]->bt_dat[j] = NULL;
    attr->outputs[i]->bt_back_dat = (var_t_x **)realloc(attr->outputs[i]->bt_back_dat, bt*sizeof(var_t_x *));
#ifdef NN_NOCHECK_X
    if (attr->outputs[i]->bt_back_dat == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j)
      attr->outputs[i]->bt_back_dat[j] = NULL;
    attr->outputs[i]->bt_attr = (void **)realloc(attr->outputs[i]->bt_attr, bt*sizeof(void *));
#ifdef NN_NOCHECK_X
    if (attr->outputs[i]->bt_attr == NULL)
      return -1;
#endif
    for (j = 0; j < bt; ++j) {
      if (attr->outputs[i]->attr_cp != NULL)
        attr->outputs[i]->bt_attr[j] = attr->outputs[i]->attr_cp(attr->outputs[i]->attr);
      else
        attr->outputs[i]->bt_attr[j] = NULL;
    }
  }
  return 0;
}

void nn_free_node_x(nn_node_t_x *node, int bt) {
  if (node->ins != NULL)
    free(node->ins);
  if (node->outs != NULL)
    free(node->outs);
  if (node->dat != NULL)
    free_var_x(node->dat);
  if (node->back_dat != NULL)
    free_var_x(node->back_dat);
  if (node->attr != NULL)
    free(node->attr);
  if (node->bt_ready != NULL)
    free(node->bt_ready);
  int i;
  if (node->bt_dat != NULL) {
    for (i = 0; i < bt; ++i)
      if (node->bt_dat[i] != NULL)
        free_var_x(node->bt_dat[i]);
    free(node->bt_dat);
  }
  if (node->bt_back_dat != NULL) {
    for (i = 0; i < bt; ++i)
      if (node->bt_back_dat[i] != NULL)
        free_var_x(node->bt_back_dat[i]);
    free(node->bt_back_dat);
  }
  if (node->bt_attr != NULL) {
    for (i = 0; i < bt; ++i)
      if (node->bt_attr[i] != NULL)
        free(node->bt_attr[i]);
    free(node->bt_attr);
  }
  free(node);
}

void nn_free_x(nn_attr_t_x *attr) {
  int i;
  for (i = 0; i < attr->inputn; ++i)
    nn_free_node_x(attr->inputs[i], attr->inputs[i]->btn);
  for (i = 0; i < attr->paramn; ++i)
    nn_free_node_x(attr->params[i], attr->inputs[0]->btn);
  for (i = 0; i < attr->intermn; ++i)
    nn_free_node_x(attr->interms[i], attr->interms[i]->btn);
  for (i = 0; i < attr->outputn; ++i)
    nn_free_node_x(attr->outputs[i], attr->outputs[i]->btn);
  free(attr->inputs);
  free(attr->params);
  free(attr->interms);
  free(attr->outputs);
  free(attr);
}

void *prop_worker_x(void *dat) {
  prop_worker_t_x *info = (prop_worker_t_x *)dat;
  int i;
  int status;
  while (1) {
    pthread_mutex_lock(&info->mutex);
    while (info->ready == 0) {
      status = pthread_cond_wait(&info->cond, &info->mutex);
#ifdef NN_NOCHECK_X
      if (status != 0) {
        info->status = status;
        return NULL;
      }
#endif
    }
    for (i = info->a; i < info->b; ++i) {
      status = info->func(info->node, i);
#ifdef NN_NOCHECK_X
      if (status != 0) {
        info->status = status;
        return NULL;
      }
#endif
    }
    info->ready = 0;
    pthread_mutex_unlock(&info->mutex);
    pthread_mutex_lock(info->mutex2);
    ++(*(info->count));
    pthread_cond_signal(info->cond2);
    pthread_mutex_unlock(info->mutex2);
  }
  return NULL;
}

static pthread_t *threads = NULL;
static prop_worker_t_x **thread_info = NULL;
static pthread_cond_t main_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t main_mut = PTHREAD_MUTEX_INITIALIZER;
static int thread_num = 0;
static int complete_count = 0;

int alloc_workers_x(int thread_count) {
  if (thread_num >= thread_count)
    return 0;
  threads = (pthread_t *)realloc(threads, thread_count*sizeof(pthread_t));
  thread_info = (prop_worker_t_x **)realloc(thread_info, thread_count*sizeof(prop_worker_t_x *));
  int i;
  int status;
  for (i = thread_num; i < thread_count; ++i) {
    thread_info[i] = (prop_worker_t_x *)malloc(sizeof(prop_worker_t_x));
    prop_worker_t_x *info = thread_info[i];
    pthread_mutex_init(&info->mutex, NULL);
    pthread_cond_init(&info->cond, NULL);
    info->cond2 = &main_cond;
    info->mutex2 = &main_mut;
    info->count = &complete_count;
    info->status = 0;
    info->ready = 0;
    status = pthread_create(&threads[i], NULL, prop_worker_x, info);
    if (status != 0)
      return status;
    pthread_detach(threads[i]);
  }
  thread_num = thread_count;
  return 0;
}

int mult_thread_prop_x(void *node, int btn, int thread_count, prop_func_t_x func) {
  if (thread_count > btn)
    return -1;
  int incre = btn/thread_count;
  num_t_x remain = ((num_t_x)btn/(num_t_x)thread_count)-incre;
  if (remain >= 0.5)
    ++incre;
  int i;
  int count = 0;
  int status;
  pthread_mutex_lock(&main_mut);
  complete_count = 0;
  for (i = 0; i < btn; i += incre)
    ++count;
  alloc_workers_x(count);
  count = 0;
  for (i = 0; i < btn; i += incre) {
    int n = incre;
    if (i+incre-1 >= btn)
      n = btn-i;
    prop_worker_t_x *info = thread_info[count];
    pthread_mutex_lock(&info->mutex);
    info->a = i;
    info->b = i+n;
    info->func = func;
    info->node = node;
    info->ready = 1;
    pthread_cond_signal(&info->cond);
    pthread_mutex_unlock(&info->mutex);
    ++count;
  }
  while (complete_count != count) {
    status = pthread_cond_wait(&main_cond, &main_mut);
    if (status != 0)
      return status;
  }
  pthread_mutex_unlock(&main_mut);
  return 0;
}

int nn_forward_prop_node_wrap_x(void *node, int bt) {
  return nn_forward_prop_node_x((nn_node_t_x *)node, bt);
}

int nn_backward_prop_node_wrap_x(void *node, int bt) {
  return nn_backward_prop_node_x((nn_node_t_x *)node, bt);
}