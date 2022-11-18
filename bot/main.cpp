#pragma GCC optimize(3,"Ofast","inline")
#include <iostream>
#include <algorithm>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "jsoncpp/json.h"

#define SECRET_NUM -1234

typedef struct
{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct
{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum
{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum
{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum
{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum
{
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum
{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct
{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer
{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

};

void free_layer(layer);

typedef enum
{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network
{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

} network;

typedef struct
{
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct
{
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct
{
    float x, y, w, h;
} box;

typedef struct detection
{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix
{
    int rows, cols;
    float **vals;
} matrix;


typedef struct
{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum
{
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args
{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct
{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node
{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list
{
    int size;
    node *front;
    node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

void compare_networks(network *n1, network *n2, data d);
char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);

float network_accuracy_multi(network *net, data d, int n);
int get_predicted_class_network(network *net);
void print_network(network *net);
int resize_network(network *net, int w, int h);
void calc_network_cost(network *net);

layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer l, network net);
void backward_activation_layer(layer l, network net);

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x)
{
    return x;
}
static inline float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}
static inline float loggy_activate(float x)
{
    return 2./(1. + exp(-x)) - 1;
}
static inline float relu_activate(float x)
{
    return x*(x>0);
}
static inline float elu_activate(float x)
{
    return (x >= 0)*x + (x < 0)*(exp(x)-1);
}
static inline float selu_activate(float x)
{
    return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);
}
static inline float relie_activate(float x)
{
    return (x>0) ? x : .01*x;
}
static inline float ramp_activate(float x)
{
    return x*(x>0)+.1*x;
}
static inline float leaky_activate(float x)
{
    return (x>0) ? x : .1*x;
}
static inline float tanh_activate(float x)
{
    return (exp(2*x)-1)/(exp(2*x)+1);
}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x)
{
    return 1;
}
static inline float logistic_gradient(float x)
{
    return (1-x)*x;
}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x)
{
    return (x>0);
}
static inline float elu_gradient(float x)
{
    return (x >= 0) + (x < 0)*(x + 1);
}
static inline float selu_gradient(float x)
{
    return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);
}
static inline float relie_gradient(float x)
{
    return (x>0) ? 1 : .01;
}
static inline float ramp_gradient(float x)
{
    return (x>0)+.1;
}
static inline float leaky_gradient(float x)
{
    return (x>0) ? 1 : .1;
}
static inline float tanh_gradient(float x)
{
    return 1-x*x;
}
static inline float plse_gradient(float x)
{
    return (x < 0 || x > 1) ? .01 : .125;
}

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);

layer make_batchnorm_layer(int batch, int w, int h, int c);
void forward_batchnorm_layer(layer l, network net);
void backward_batchnorm_layer(layer l, network net);

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_gpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

typedef struct
{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

void col2im_cpu(float* data_col,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_im);

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);

typedef layer convolutional_layer;

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void update_convolutional_layer(convolutional_layer layer, update_args a);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(convolutional_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

typedef layer cost_layer;

COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer(const cost_layer l, network net);
void backward_cost_layer(const cost_layer l, network net);
void resize_cost_layer(cost_layer *l, int inputs);

layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);

void forward_crnn_layer(layer l, network net);
void backward_crnn_layer(layer l, network net);
void update_crnn_layer(layer l, update_args a);

typedef layer crop_layer;

image get_crop_image(crop_layer l);
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(const crop_layer l, network net);
void resize_crop_layer(layer *l, int w, int h);

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}
void load_data_blocking(load_args args);


void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_regression(char **paths, int n, int m, int classes, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
data load_go(char *filename);


data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(const layer l, network net);
void update_deconvolutional_layer(layer l, update_args a);
void backward_deconvolutional_layer(layer l, network net);

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer l, network net);
void backward_detection_layer(const detection_layer l, network net);

typedef layer dropout_layer;

dropout_layer make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer l, network net);
void backward_dropout_layer(dropout_layer l, network net);
void resize_dropout_layer(dropout_layer *l, int inputs);

void gemm_bin(int M, int N, int K, float ALPHA,
              char  *A, int lda,
              float *B, int ldb,
              float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);

void forward_gru_layer(layer l, network state);
void backward_gru_layer(layer l, network state);
void update_gru_layer(layer l, update_args a);

void im2col_cpu(float* data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_col);


float get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
image image_distance(image a, image b);
void scale_image(image m, float s);
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h);
void letterbox_image_into(image im, int w, int h, image boxed);
image resize_max(image im, int max);
void translate_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
void distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);


image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void print_image(image m);

image make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

image get_image_layer(image m, int l);

layer make_iseg_layer(int batch, int w, int h, int classes, int ids);
void forward_iseg_layer(const layer l, network net);
void backward_iseg_layer(const layer l, network net);
void resize_iseg_layer(layer *l, int w, int h);
int iseg_num_detections(layer l, float thresh);

layer make_l2norm_layer(int batch, int inputs);
void forward_l2norm_layer(const layer l, network net);
void backward_l2norm_layer(const layer l, network net);

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);


void free_list_contents(list *l);

typedef layer local_layer;

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);

void forward_local_layer(const local_layer layer, network net);
void backward_local_layer(local_layer layer, network net);
void update_local_layer(local_layer layer, update_args a);

void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network net);
void backward_logistic_layer(const layer l, network net);

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);

void forward_lstm_layer(layer l, network net);
void update_lstm_layer(layer l, update_args a);

matrix copy_matrix(matrix m);
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

typedef layer maxpool_layer;

image get_maxpool_image(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer *layer, int h, int w);
void forward_normalization_layer(const layer layer, network net);
void backward_normalization_layer(const layer layer, network net);

typedef struct
{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void forward_region_layer(const layer l, network net);
void backward_region_layer(const layer l, network net);
void resize_region_layer(layer *l, int w, int h);

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network net);
void backward_reorg_layer(const layer l, network net);

layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam);

void forward_rnn_layer(layer l, network net);
void backward_rnn_layer(layer l, network net);
void update_rnn_layer(layer l, update_args a);

typedef layer route_layer;

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size);
void forward_route_layer(const route_layer l, network net);
void backward_route_layer(const route_layer l, network net);
void resize_route_layer(route_layer *l, network *net);

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer(const layer l, network net);
void backward_shortcut_layer(const layer l, network net);
void resize_shortcut_layer(layer *l, int w, int h);

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

layer make_upsample_layer(int batch, int w, int h, int c, int stride);
void forward_upsample_layer(const layer l, network net);
void backward_upsample_layer(const layer l, network net);
void resize_upsample_layer(layer *l, int w, int h);

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void translate_array(float *a, int n, float s);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float rand_scale(float s);
int rand_int(int min, int max);
void mean_arrays(float **a, int n, int els, float *avg);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
void print_statistics(float *a, int n);
int int_index(int *a, int val, int n);

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_yolo_layer(const layer l, network net);
void backward_yolo_layer(const layer l, network net);
void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)calloc(batch*inputs, sizeof(float*));
    l.delta = (float*)calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;

    l.activation = activation;
    // fprintf(stderr, (char*)"Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

char *get_activation_string(ACTIVATION a)
{
    switch(a)
    {
    case LOGISTIC:
        return (char*)"logistic";
    case LOGGY:
        return (char*)"loggy";
    case RELU:
        return (char*)"relu";
    case ELU:
        return (char*)"elu";
    case SELU:
        return (char*)"selu";
    case RELIE:
        return (char*)"relie";
    case RAMP:
        return (char*)"ramp";
    case LINEAR:
        return (char*)"linear";
    case TANH:
        return (char*)"tanh";
    case PLSE:
        return (char*)"plse";
    case LEAKY:
        return (char*)"leaky";
    case STAIR:
        return (char*)"stair";
    case HARDTAN:
        return (char*)"hardtan";
    case LHTAN:
        return (char*)"lhtan";
    default:
        break;
    }
    return (char*)"relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, (char*)"logistic")==0) return LOGISTIC;
    if (strcmp(s, (char*)"loggy")==0) return LOGGY;
    if (strcmp(s, (char*)"relu")==0) return RELU;
    if (strcmp(s, (char*)"elu")==0) return ELU;
    if (strcmp(s, (char*)"selu")==0) return SELU;
    if (strcmp(s, (char*)"relie")==0) return RELIE;
    if (strcmp(s, (char*)"plse")==0) return PLSE;
    if (strcmp(s, (char*)"hardtan")==0) return HARDTAN;
    if (strcmp(s, (char*)"lhtan")==0) return LHTAN;
    if (strcmp(s, (char*)"linear")==0) return LINEAR;
    if (strcmp(s, (char*)"ramp")==0) return RAMP;
    if (strcmp(s, (char*)"leaky")==0) return LEAKY;
    if (strcmp(s, (char*)"tanh")==0) return TANH;
    if (strcmp(s, (char*)"stair")==0) return STAIR;
    // fprintf(stderr, (char*)"Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a)
    {
    case LINEAR:
        return linear_activate(x);
    case LOGISTIC:
        return logistic_activate(x);
    case LOGGY:
        return loggy_activate(x);
    case RELU:
        return relu_activate(x);
    case ELU:
        return elu_activate(x);
    case SELU:
        return selu_activate(x);
    case RELIE:
        return relie_activate(x);
    case RAMP:
        return ramp_activate(x);
    case LEAKY:
        return leaky_activate(x);
    case TANH:
        return tanh_activate(x);
    case PLSE:
        return plse_activate(x);
    case STAIR:
        return stair_activate(x);
    case HARDTAN:
        return hardtan_activate(x);
    case LHTAN:
        return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a)
    {
    case LINEAR:
        return linear_gradient(x);
    case LOGISTIC:
        return logistic_gradient(x);
    case LOGGY:
        return loggy_gradient(x);
    case RELU:
        return relu_gradient(x);
    case ELU:
        return elu_gradient(x);
    case SELU:
        return selu_gradient(x);
    case RELIE:
        return relie_gradient(x);
    case RAMP:
        return ramp_gradient(x);
    case LEAKY:
        return leaky_gradient(x);
    case TANH:
        return tanh_gradient(x);
    case PLSE:
        return plse_gradient(x);
    case STAIR:
        return stair_gradient(x);
    case HARDTAN:
        return hardtan_gradient(x);
    case LHTAN:
        return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        delta[i] *= gradient(x[i], a);
    }
}

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    // fprintf(stderr, (char*)"avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l;
    memset(&l,0,sizeof(avgpool_layer));
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b)
    {
        for(k = 0; k < l.c; ++k)
        {
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i)
            {
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b)
    {
        for(k = 0; k < l.c; ++k)
        {
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i)
            {
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    // fprintf(stderr, (char*)"Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = (float*)calloc(h * w * c * batch, sizeof(float));
    l.delta  = (float*)calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = (float*)calloc(c, sizeof(float));
    l.scale_updates = (float*)calloc(c, sizeof(float));
    l.biases = (float*)calloc(c, sizeof(float));
    l.bias_updates = (float*)calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i)
    {
        l.scales[i] = 1;
    }

    l.mean = (float*)calloc(c, sizeof(float));
    l.variance = (float*)calloc(c, sizeof(float));

    l.rolling_mean = (float*)calloc(c, sizeof(float));
    l.rolling_variance = (float*)calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;

    return l;
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f)
    {
        float sum = 0;
        for(b = 0; b < batch; ++b)
        {
            for(i = 0; i < size; ++i)
            {
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j)
        {
            for (k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j)
    {
        for(f = 0; f < filters; ++f)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    // fprintf(stderr, (char*)"Not implemented\n");
}

void forward_batchnorm_layer(layer l, network net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train)
    {
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    }
    else
    {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
    if(!net.train)
    {
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b)
    {
        for(k = 0; k < c; ++k)
        {
            for(j = 0; j < h; ++j)
            {
                for(i = 0; i < w; ++i)
                {
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float*)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b)
    {
        for(c = 0; c < layers; ++c)
        {
            for(i = 0; i < size; ++i)
            {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b)
    {
        for(k = 0; k < minc; ++k)
        {
            for(j = 0; j < minh; ++j)
            {
                for(i = 0; i < minw; ++i)
                {
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        mean[i] = 0;
        for(j = 0; j < batch; ++j)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i)
    {
        variance[i] = 0;
        for(j = 0; j < batch; ++j)
        {
            for(k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    for(b = 0; b < batch; ++b)
    {
        for(i = 0; i < spatial; ++i)
        {
            float sum = 0;
            for(f = 0; f < filters; ++f)
            {
                int index = b*filters*spatial + f*spatial + i;
                sum += powf(x[index], 2);
            }
            sum = sqrtf(sum);
            for(f = 0; f < filters; ++f)
            {
                int index = b*filters*spatial + f*spatial + i;
                x[index] /= sum;
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}


void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b)
    {
        for(f = 0; f < filters; ++f)
        {
            for(i = 0; i < spatial; ++i)
            {
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j)
    {
        for(i = 0; i < NX; ++i)
        {
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i)
        {
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j)
    {
        for(i = 0; i < NX; ++i)
        {
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i)
        {
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1)
        {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else
        {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i)
    {
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i)
    {
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i)
    {
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b)
    {
        for(g = 0; g < groups; ++g)
        {
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b)
    {
        for(k = 0; k < c; ++k)
        {
            for(j = 0; j < h*stride; ++j)
            {
                for(i = 0; i < w*stride; ++i)
                {
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0)
    {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else
    {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i)
    {
        if(dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(i = 0; i < total; ++i)
    {
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator);
    for(i = 0; i < total; ++i)
    {
        if(dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for(j = i+1; j < total; ++j)
        {
            if(dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh)
            {
                dets[j].objectness = 0;
                for(k = 0; k < classes; ++k)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i)
    {
        if(dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k)
    {
        for(i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i)
        {
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

box float_to_box(float *f, int stride)
{
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}

dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2)
    {
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2)
    {
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2)
    {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2)
    {
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2)
    {
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2)
    {
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2)
    {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2)
    {
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) +
                pow(a.y-b.y, 2) +
                pow(a.w-b.w, 2) +
                pow(a.h-b.h, 2));
}

dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1)
    {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}


void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i)
    {
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any)
        {
            continue;
        }
        for(j = i+1; j < total; ++j)
        {
            if (box_iou(boxes[i], boxes[j]) > thresh)
            {
                for(k = 0; k < classes; ++k)
                {
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}

void col2im_add_pixel(float *im, int height, int width, int channels,
                      int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
            row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_im)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                                 im_row, im_col, c_im, pad, val);
            }
        }
    }
}

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l;
    memset(&l,0,sizeof(layer));
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = (float*)calloc(batch*outputs, sizeof(float));
    l.delta = (float*)calloc(batch*outputs, sizeof(float));

    l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    l.bias_updates = (float*)calloc(outputs, sizeof(float));

    l.weights = (float*)calloc(outputs*inputs, sizeof(float));
    l.biases = (float*)calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i)
    {
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i)
    {
        l.biases[i] = 0;
    }

    if(adam)
    {
        l.m = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        l.v = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = (float*)calloc(l.outputs, sizeof(float));
        l.scale_m = (float*)calloc(l.outputs, sizeof(float));
        l.bias_v = (float*)calloc(l.outputs, sizeof(float));
        l.scale_v = (float*)calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize)
    {
        l.scales = (float*)calloc(outputs, sizeof(float));
        l.scale_updates = (float*)calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(outputs, sizeof(float));
        l.mean_delta = (float*)calloc(outputs, sizeof(float));
        l.variance = (float*)calloc(outputs, sizeof(float));
        l.variance_delta = (float*)calloc(outputs, sizeof(float));

        l.rolling_mean = (float*)calloc(outputs, sizeof(float));
        l.rolling_variance = (float*)calloc(outputs, sizeof(float));

        l.x = (float*)calloc(batch*outputs, sizeof(float));
        l.x_norm = (float*)calloc(batch*outputs, sizeof(float));
    }

    l.activation = activation;
    // fprintf(stderr, (char*)"connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize)
    {
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    }
    else
    {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i)
    {
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j)
        {
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize)
    {
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f)
    {
        float mean = 0;
        for(i = 0; i < size; ++i)
        {
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i)
        {
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s)
    {
        float mean = 0;
        for(i = 0; i < n; ++i)
        {
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i)
        {
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l)
{
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l;
    memset(&l,0,sizeof(convolutional_layer));
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = (float*)calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = (float*)calloc(c/groups*n*size*size, sizeof(float));

    l.biases = (float*)calloc(n, sizeof(float));
    l.bias_updates = (float*)calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary)
    {
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.cweights = (char*)calloc(l.nweights, sizeof(char));
        l.scales = (float*)calloc(n, sizeof(float));
    }
    if(xnor)
    {
        l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
        l.binary_input = (float*)calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize)
    {
        l.scales = (float*)calloc(n, sizeof(float));
        l.scale_updates = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(n, sizeof(float));
        l.variance = (float*)calloc(n, sizeof(float));

        l.mean_delta = (float*)calloc(n, sizeof(float));
        l.variance_delta = (float*)calloc(n, sizeof(float));

        l.rolling_mean = (float*)calloc(n, sizeof(float));
        l.rolling_variance = (float*)calloc(n, sizeof(float));
        l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam)
    {
        l.m = (float*)calloc(l.nweights, sizeof(float));
        l.v = (float*)calloc(l.nweights, sizeof(float));
        l.bias_m = (float*)calloc(n, sizeof(float));
        l.scale_m = (float*)calloc(n, sizeof(float));
        l.bias_v = (float*)calloc(n, sizeof(float));
        l.scale_v = (float*)calloc(n, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    // fprintf(stderr, (char*)"conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i)
    {
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j)
        {
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = (float*)realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize)
    {
        l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b)
    {
        for(i = 0; i < n; ++i)
        {
            for(j = 0; j < size; ++j)
            {
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b)
    {
        for(i = 0; i < n; ++i)
        {
            for(j = 0; j < size; ++j)
            {
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b)
    {
        for(i = 0; i < n; ++i)
        {
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor)
    {
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i)
    {
        for(j = 0; j < l.groups; ++j)
        {
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1)
            {
                b = im;
            }
            else
            {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    if(l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    }
    else
    {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i)
    {
        for(j = 0; j < l.groups; ++j)
        {
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1)
            {
                b = im;
            }
            else
            {
                im2col_cpu(im, l.c/l.groups, l.h, l.w,
                           l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta)
            {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1)
                {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1)
                {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales)
    {
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i)
    {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3)
        {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i)
    {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3)
        {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image*)calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i)
    {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, (char*)"filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, (char*)"seg")==0) return SEG;
    if (strcmp(s, (char*)"sse")==0) return SSE;
    if (strcmp(s, (char*)"masked")==0) return MASKED;
    if (strcmp(s, (char*)"smooth")==0) return SMOOTH;
    if (strcmp(s, (char*)"L1")==0) return L1;
    if (strcmp(s, (char*)"wgan")==0) return WGAN;
    // fprintf(stderr, (char*)"Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a)
    {
    case SEG:
        return (char*)"seg";
    case SSE:
        return (char*)"sse";
    case MASKED:
        return (char*)"masked";
    case SMOOTH:
        return (char*)"smooth";
    case L1:
        return (char*)"L1";
    case WGAN:
        return (char*)"wgan";
    }
    return (char*)"sse";
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    // fprintf(stderr, (char*)"cost                                           %4d\n",  inputs);
    cost_layer l;
    memset(&l,0,sizeof(cost_layer));
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;

    return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = (float*)realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, inputs*l->batch*sizeof(float));
}

void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED)
    {
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i)
        {
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    if(l.cost_type == SMOOTH)
    {
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    else if(l.cost_type == L1)
    {
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    else
    {
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer(const cost_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;
}

layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize)
{
    // fprintf(stderr, (char*)"CRNN Layer: %d x %d x %d image, %d filters\n", h,w,c,output_filters);
    batch = batch / steps;
    layer l;
    memset(&l,0,sizeof(layer));
    l.batch = batch;
    l.type = CRNN;
    l.steps = steps;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    l.out_c = output_filters;
    l.inputs = h*w*c;
    l.hidden = h * w * hidden_filters;
    l.outputs = l.out_h * l.out_w * l.out_c;

    l.state = (float*)calloc(l.hidden*batch*(steps+1), sizeof(float));

    l.input_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.input_layer) = make_convolutional_layer(batch*steps, h, w, c, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.input_layer->batch = batch;

    l.self_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.self_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.self_layer->batch = batch;

    l.output_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.output_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l.output_layer->batch = batch;

    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_crnn_layer;
    l.backward = backward_crnn_layer;
    l.update = update_crnn_layer;

    return l;
}

void update_crnn_layer(layer l, update_args a)
{
    update_convolutional_layer(*(l.input_layer),  a);
    update_convolutional_layer(*(l.self_layer),   a);
    update_convolutional_layer(*(l.output_layer), a);
}

void forward_crnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
    if(net.train) fill_cpu(l.hidden * l.batch, 0, l.state, 1);

    for (i = 0; i < l.steps; ++i)
    {
        s.input = net.input;
        forward_convolutional_layer(input_layer, s);

        s.input = l.state;
        forward_convolutional_layer(self_layer, s);

        float *old_state = l.state;
        if(net.train) l.state += l.hidden*l.batch;
        if(l.shortcut)
        {
            copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
        }
        else
        {
            fill_cpu(l.hidden * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_convolutional_layer(output_layer, s);

        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_crnn_layer(layer l, network net)
{
    network s = net;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.hidden*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i)
    {
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_convolutional_layer(output_layer, s);

        l.state -= l.hidden*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
           axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.hidden * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.hidden*l.batch;
        if (i == 0) s.delta = 0;
        backward_convolutional_layer(self_layer, s);

        copy_cpu(l.hidden*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.hidden*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden*l.batch, 1);
        s.input = net.input + i*l.inputs*l.batch;
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_convolutional_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

image get_crop_image(crop_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}

void backward_crop_layer(const crop_layer l, network net) {}
void backward_crop_layer_gpu(const crop_layer l, network net) {}

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    // fprintf(stderr, (char*)"Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer l;
    memset(&l,0,sizeof(crop_layer));
    l.type = CROP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = (float)crop_height / h;
    l.flip = flip;
    l.angle = angle;
    l.saturation = saturation;
    l.exposure = exposure;
    l.out_w = crop_width;
    l.out_h = crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = (float*)calloc(l.outputs*batch, sizeof(float));
    l.forward = forward_crop_layer;
    l.backward = backward_crop_layer;

    return l;
}

void resize_crop_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));

}


void forward_crop_layer(const crop_layer l, network net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (l.flip && rand()%2);
    int dh = rand()%(l.h - l.out_h + 1);
    int dw = rand()%(l.w - l.out_w + 1);
    float scale = 2;
    float trans = -1;
    if(l.noadjust)
    {
        scale = 1;
        trans = 0;
    }
    if(!net.train)
    {
        flip = 0;
        dh = (l.h - l.out_h)/2;
        dw = (l.w - l.out_w)/2;
    }
    for(b = 0; b < l.batch; ++b)
    {
        for(c = 0; c < l.c; ++c)
        {
            for(i = 0; i < l.out_h; ++i)
            {
                for(j = 0; j < l.out_w; ++j)
                {
                    if(flip)
                    {
                        col = l.w - dw - j - 1;
                    }
                    else
                    {
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+l.w*(row+l.h*(c + l.c*b));
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, (char*)"r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file)))
    {
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = (char**)calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i)
    {
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = (char**)calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i)
    {
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i)
    {
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i)
    {
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i)
    {
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center)
        {
            crop = center_crop_image(im, size, size);
        }
        else
        {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, (char*)"r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = (box_label*)calloc(size, sizeof(box_label));
    while(fscanf(file, (char*)"%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
    {
        if(count == size)
        {
            size = size * 2;
            boxes = (box_label*)realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        if(boxes[i].x == 0 && boxes[i].y == 0)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip)
        {
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"labels", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 90; ++i)
    {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"labels", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);

    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i)
    {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i)
    {
        for(j = 0; j < rle[i]; ++j)
        {
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count)
    {
        im.data[count] = curr;
    }
}

void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i)
    {
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}

void exclusive_image(image src)
{
    int k, j, i;
    int s = src.w*src.h;
    for(k = 0; k < src.c-1; ++k)
    {
        for(i = 0; i < s; ++i)
        {
            if (src.data[k*s + i])
            {
                for(j = k+1; j < src.c; ++j)
                {
                    src.data[j*s + i] = 0;
                }
            }
        }
    }
}

box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y)
    {
        for(x = 0; x < im.w; ++x)
        {
            if(im.data[y*im.w + x])
            {
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
    return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    FILE *file = fopen(labelpath, (char*)"r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    int j;
    image part = make_image(w, h, 1);
    while((fscanf(file, (char*)"%d %s", &id, buff) == 2) && i < num_boxes)
    {
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);

        image mask = resize_image(sized, mw, mh);
        truth[i*(mw*mh+1)] = id;
        for(j = 0; j < mw*mh; ++j)
        {
            truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
        }
        ++i;

        free_image(mask);
        free_image(sized);
        free(rle);
    }
    if(i < num_boxes) truth[i*(mw*mh+1)] = -1;
    fclose(file);
    free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    FILE *file = fopen(labelpath, (char*)"r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, (char*)"%d %s", &id, buff) == 2) && i < num_boxes)
    {
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);
        box b = bound_image(sized);
        if(b.w > 0)
        {
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j)
            {
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}


void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"labels", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);

    find_replace(labelpath, (char*)"raw", (char*)"labels", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;
    int sub = 0;

    for (i = 0; i < count; ++i)
    {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001))
        {
            ++sub;
            continue;
        }

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
    }
    free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i)
    {
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(; i < n; ++i)
    {
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i)
    {
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i)
    {
        if(strstr(path, labels[i]))
        {
            truth[i] = 1;
            ++count;
            //printf("%s %s %d\n", path, labels[i], i);
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j)
    {
        if(truth[j])
        {
            int parent = hierarchy->parent[j];
            while(parent >= 0)
            {
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j)
    {
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i)
        {
            if(truth[count + i])
            {
                mask = 0;
                break;
            }
        }
        if (mask)
        {
            for(i = 0; i < hierarchy->group_size[j]; ++i)
            {
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i,j;
    for(i = 0; i < n; ++i)
    {
        char labelpath[4096];
        find_replace(paths[i], (char*)"images", (char*)"labels", labelpath);
        find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);
        find_replace(labelpath, (char*)".BMP", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".JPeG", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".Jpeg", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".PNG", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".TIF", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".bmp", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".jpeg", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
        find_replace(labelpath, (char*)".tif", (char*)".txt", labelpath);

        FILE *file = fopen(labelpath, (char*)"r");
        for(j = 0; j < k; ++j)
        {
            fscanf(file, (char*)"%f", &(y.vals[i][j]));
        }
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i)
    {
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy)
        {
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    //int count = 0;
    for(i = 0; i < n; ++i)
    {
        char label[4096];
        find_replace(paths[i], (char*)"images", (char*)"labels", label);
        find_replace(label, (char*)".jpg", (char*)".txt", label);
        FILE *file = fopen(label, (char*)"r");
        if (!file) continue;
        //++count;
        int tag;
        while(fscanf(file, (char*)"%d", &tag) == 1)
        {
            if(tag < k)
            {
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    //printf("%d/%d\n", count, n);
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow)
    {
        free_matrix(d.X);
        free_matrix(d.y);
    }
    else
    {
        free(d.X.vals);
        free(d.y.vals);
    }
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, (char*)"r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, (char*)"%d %s", &id, buff) == 2)
    {
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, (char*)"images", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
    find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
    find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i)
    {
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, (char*)"r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, (char*)"%d %s", &id, buff) == 2)
    {
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i)
        {
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    d.y.rows = n;
    d.y.cols = h*w*classes/div/div;
    d.y.vals = (float**)calloc(d.X.rows, sizeof(float*));

    for(i = 0; i < n; ++i)
    {
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
        //image mask = make_image(orig.w, orig.h, classes+1);
        image sized_m = rotate_crop_image(mask, a.rad, a.scale/div, a.w/div, a.h/div, a.dx/div, a.dy/div, a.aspect);

        if(flip) flip_image(sized_m);
        d.y.vals[i] = sized_m.data;

        free_image(orig);
        free_image(mask);

    }
    free(random_paths);
    return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (((w/div)*(h/div))+1)*boxes);

    for(i = 0; i < n; ++i)
    {
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, (char*)"image");

        fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, w/div, h/div);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, (char*)"part");
           show_image(sized, (char*)"orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (coords+1)*boxes);

    for(i = 0; i < n; ++i)
    {
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, (char*)"image");

        fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, 14, 14);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, (char*)"part");
           show_image(sized, (char*)"orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i)
    {
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i)
    {
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = (float*)calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2], (char*)"imgs", (char*)"labels", imlabel1);
        find_replace(imlabel1, (char*)"jpg", (char*)"txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, (char*)"r");

        while(fscanf(fp1, (char*)"%d %f", &id, &iou) == 2)
        {
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], (char*)"imgs", (char*)"labels", imlabel2);
        find_replace(imlabel2, (char*)"jpg", (char*)"txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, (char*)"r");

        while(fscanf(fp2, (char*)"%d %f", &id, &iou) == 2)
        {
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j)
        {
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5)
            {
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            }
            else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5)
            {
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            }
            else
            {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*90;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, 5*boxes);
    for(i = 0; i < n; ++i)
    {
        image orig = load_image_color(random_paths[i], 0, 0);
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        float dw = jitter * orig.w;
        float dh = jitter * orig.h;

        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
        //float scale = rand_uniform(.25, 2);
        float scale = 1;

        float nw, nh;

        if(new_ar < 1)
        {
            nh = scale * h;
            nw = nh * new_ar;
        }
        else
        {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(orig, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;


        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h);

        free_image(orig);
    }
    free(random_paths);
    return d;
}

void *load_thread(void *ptr)
{
    //printf("Loading data: %d\n", rand());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA)
    {
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    }
    else if (a.type == REGRESSION_DATA)
    {
        *a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    else if (a.type == CLASSIFICATION_DATA)
    {
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    }
    else if (a.type == SUPER_DATA)
    {
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    }
    else if (a.type == WRITING_DATA)
    {
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    }
    else if (a.type == ISEG_DATA)
    {
        *a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    else if (a.type == INSTANCE_DATA)
    {
        *a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    else if (a.type == SEGMENTATION_DATA)
    {
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale);
    }
    else if (a.type == REGION_DATA)
    {
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    }
    else if (a.type == DETECTION_DATA)
    {
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    }
    else if (a.type == SWAG_DATA)
    {
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    }
    else if (a.type == COMPARE_DATA)
    {
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    }
    else if (a.type == IMAGE_DATA)
    {
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    }
    else if (a.type == LETTERBOX_DATA)
    {
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    }
    else if (a.type == TAG_DATA)
    {
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data *buffers = (data*)calloc(args.threads, sizeof(data));
    pthread_t *threads = (pthread_t*)calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i)
    {
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i)
    {
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i)
    {
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

void load_data_blocking(load_args args)
{
    struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
    *ptr = args;
    load_thread(ptr);
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, (char*)".png", (char*)"-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = (float**)calloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = (float**)calloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i)
    {
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = rand()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_regression_labels_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

data select_data(data *orig, int *inds)
{
    data d = {0};
    d.shallow = 1;
    d.w = orig[0].w;
    d.h = orig[0].h;

    d.X.rows = orig[0].X.rows;
    d.y.rows = orig[0].X.rows;

    d.X.cols = orig[0].X.cols;
    d.y.cols = orig[0].y.cols;

    d.X.vals = (float**)calloc(orig[0].X.rows, sizeof(float *));
    d.y.vals = (float**)calloc(orig[0].y.rows, sizeof(float *));
    int i;
    for(i = 0; i < d.X.rows; ++i)
    {
        d.X.vals[i] = orig[inds[i]].X.vals[i];
        d.y.vals[i] = orig[inds[i]].y.vals[i];
    }
    return d;
}

data *tile_data(data orig, int divs, int size)
{
    data *ds = (data*)calloc(divs*divs, sizeof(data));
    int i, j;
    #pragma omp parallel for
    for(i = 0; i < divs*divs; ++i)
    {
        data d;
        d.shallow = 0;
        d.w = orig.w/divs * size;
        d.h = orig.h/divs * size;
        d.X.rows = orig.X.rows;
        d.X.cols = d.w*d.h*3;
        d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));

        d.y = copy_matrix(orig.y);
        #pragma omp parallel for
        for(j = 0; j < orig.X.rows; ++j)
        {
            int x = (i%divs) * orig.w / divs - (d.w - orig.w/divs)/2;
            int y = (i/divs) * orig.h / divs - (d.h - orig.h/divs)/2;
            image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
            d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
        }
        ds[i] = d;
    }
    return ds;
}

data resize_data(data orig, int w, int h)
{
    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;
    int i;
    d.X.rows = orig.X.rows;
    d.X.cols = w*h*3;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));

    d.y = copy_matrix(orig.y);
    #pragma omp parallel for
    for(i = 0; i < orig.X.rows; ++i)
    {
        image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
        d.X.vals[i] = resize_image(im, w, h).data;
    }
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = (float**)calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i)
    {
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i)
    {
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i)
    {
        data neww = concat_data(d[i], out);
        free_data(out);
        out = neww;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, (char*)"rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i)
    {
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int classs = bytes[0];
        y.vals[i][classs] = 1;
        for(j = 0; j < X.cols; ++j)
        {
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j)
    {
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j)
    {
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i)
    {
        for(j = 0; j < d.y.cols; ++j)
        {
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b)
    {
        char buff[256];
        sprintf(buff, (char*)"data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, (char*)"rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i)
        {
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int classs = bytes[0];
            y.vals[i+b*10000][classs] = 1;
            for(j = 0; j < X.cols; ++j)
            {
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, (char*)"rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp)))
    {
        int i;
        if(count == X.rows)
        {
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, (char*)"%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i)
        {
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i)
    {
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i)
    {
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i)
    {
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i)
    {
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = (float**)calloc(num, sizeof(float *));
    r.y.vals = (float**)calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i)
    {
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = (data*)calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = (float**)calloc(train.X.rows, sizeof(float*));
    test.X.vals = (float**)calloc(test.X.rows, sizeof(float*));
    train.y.vals = (float**)calloc(train.y.rows, sizeof(float*));
    test.y.vals = (float**)calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i)
    {
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i)
    {
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i)
    {
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

static size_t get_workspace_size_1(layer l)
{
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}

void bilinear_init(layer l)
{
    int i,j,f;
    float center = (l.size-1) / 2.;
    for(f = 0; f < l.n; ++f)
    {
        for(j = 0; j < l.size; ++j)
        {
            for(i = 0; i < l.size; ++i)
            {
                float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
                int c = f%l.c;
                int ind = f*l.size*l.size*l.c + c*l.size*l.size + j*l.size + i;
                l.weights[ind] = val;
            }
        }
    }
}


layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.nweights = c*n*size*size;
    l.nbiases = n;

    l.weights = (float*)calloc(c*n*size*size, sizeof(float));
    l.weight_updates = (float*)calloc(c*n*size*size, sizeof(float));

    l.biases = (float*)calloc(n, sizeof(float));
    l.bias_updates = (float*)calloc(n, sizeof(float));
    //float scale = n/(size*size*c);
    //printf("scale: %f\n", scale);
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    //bilinear_init(l);
    for(i = 0; i < n; ++i)
    {
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);

    l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize)
    {
        l.scales = (float*)calloc(n, sizeof(float));
        l.scale_updates = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(n, sizeof(float));
        l.variance = (float*)calloc(n, sizeof(float));

        l.mean_delta = (float*)calloc(n, sizeof(float));
        l.variance_delta = (float*)calloc(n, sizeof(float));

        l.rolling_mean = (float*)calloc(n, sizeof(float));
        l.rolling_variance = (float*)calloc(n, sizeof(float));
        l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam)
    {
        l.m = (float*)calloc(c*n*size*size, sizeof(float));
        l.v = (float*)calloc(c*n*size*size, sizeof(float));
        l.bias_m = (float*)calloc(n, sizeof(float));
        l.scale_m = (float*)calloc(n, sizeof(float));
        l.bias_v = (float*)calloc(n, sizeof(float));
        l.scale_v = (float*)calloc(n, sizeof(float));
    }

    l.activation = activation;
    l.workspace_size = get_workspace_size_1(l);

    // fprintf(stderr, (char*)"deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_deconvolutional_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i)
    {
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j)
        {
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = (float*)realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize)
    {
        l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    l->workspace_size = get_workspace_size_1(*l);
}

void forward_deconvolutional_layer(const layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i)
    {
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    }
    else
    {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer(layer l, network net)
{
    int i;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i)
    {
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = net.input + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        im2col_cpu(l.delta + i*l.outputs, l.out_c, l.out_h, l.out_w,
                   l.size, l.stride, l.pad, b);
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta)
        {
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;

            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales)
    {
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l;
    memset(&l,0,sizeof(detection_layer));
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = (float*)calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));
    l.delta = (float*)calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;

    // fprintf(stderr, (char*)"Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network net)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax)
    {
        for(b = 0; b < l.batch; ++b)
        {
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i)
            {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    if(net.train)
    {
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b)
        {
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i)
            {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
                for (j = 0; j < l.n; ++j)
                {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj)
                {
                    continue;
                }

                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j)
                {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j)
                {
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt)
                    {
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0)
                    {
                        if(iou > best_iou)
                        {
                            best_iou = iou;
                            best_index = j;
                        }
                    }
                    else
                    {
                        if(rmse < best_rmse)
                        {
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced)
                {
                    if(truth.w*truth.h < .1)
                    {
                        best_index = 1;
                    }
                    else
                    {
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000)
                {
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt)
                {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore)
                {
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt)
                {
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
        }

        if(0)
        {
            float *costs = (float*)calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b)
            {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i)
                {
                    for (j = 0; j < l.n; ++j)
                    {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b)
            {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i)
                {
                    for (j = 0; j < l.n; ++j)
                    {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }


        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i)
    {
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n)
        {
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j)
            {
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l;
    memset(&l,0,sizeof(dropout_layer));
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    // fprintf(stderr, (char*)"dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = (float*)realloc(l->rand, l->inputs*l->batch*sizeof(float));
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i)
    {
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i)
    {
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}

void gemm_bin(int M, int N, int K, float ALPHA,
              char  *A, int lda,
              float *B, int ldb,
              float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i)
    {
        for(k = 0; k < K; ++k)
        {
            char A_PART = A[i*lda+k];
            if(A_PART)
            {
                for(j = 0; j < N; ++j)
                {
                    C[i*ldc+j] += B[k*ldb+j];
                }
            }
            else
            {
                for(j = 0; j < N; ++j)
                {
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i)
    {
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i)
    {
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i)
    {
        for(k = 0; k < K; ++k)
        {
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j)
            {
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i)
    {
        for(j = 0; j < N; ++j)
        {
            register float sum = 0;
            for(k = 0; k < K; ++k)
            {
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i)
    {
        for(k = 0; k < K; ++k)
        {
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j)
            {
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i)
    {
        for(j = 0; j < N; ++j)
        {
            register float sum = 0;
            for(k = 0; k < K; ++k)
            {
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i)
    {
        for(j = 0; j < N; ++j)
        {
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    // fprintf(stderr, (char*)"GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l;
    memset(&l,0,sizeof(layer));
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;

    l.uz = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.uz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uz->batch = batch;

    l.wz = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wz->batch = batch;

    l.ur = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.ur) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ur->batch = batch;

    l.wr = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wr) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wr->batch = batch;



    l.uh = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.uh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uh->batch = batch;

    l.wh = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wh->batch = batch;

    l.batch_normalize = batch_normalize;


    l.outputs = outputs;
    l.output = (float*)calloc(outputs*batch*steps, sizeof(float));
    l.delta = (float*)calloc(outputs*batch*steps, sizeof(float));
    l.state = (float*)calloc(outputs*batch, sizeof(float));
    l.prev_state = (float*)calloc(outputs*batch, sizeof(float));
    l.forgot_state = (float*)calloc(outputs*batch, sizeof(float));
    l.forgot_delta = (float*)calloc(outputs*batch, sizeof(float));

    l.r_cpu = (float*)calloc(outputs*batch, sizeof(float));
    l.z_cpu = (float*)calloc(outputs*batch, sizeof(float));
    l.h_cpu = (float*)calloc(outputs*batch, sizeof(float));

    l.forward = forward_gru_layer;
    l.backward = backward_gru_layer;
    l.update = update_gru_layer;

    return l;
}

void update_gru_layer(layer l, update_args a)
{
    update_connected_layer(*(l.ur), a);
    update_connected_layer(*(l.uz), a);
    update_connected_layer(*(l.uh), a);
    update_connected_layer(*(l.wr), a);
    update_connected_layer(*(l.wz), a);
    update_connected_layer(*(l.wh), a);
}

void forward_gru_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
    if(net.train)
    {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
        copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
    }

    for (i = 0; i < l.steps; ++i)
    {
        s.input = l.state;
        forward_connected_layer(wz, s);
        forward_connected_layer(wr, s);

        s.input = net.input;
        forward_connected_layer(uz, s);
        forward_connected_layer(ur, s);
        forward_connected_layer(uh, s);


        copy_cpu(l.outputs*l.batch, uz.output, 1, l.z_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wz.output, 1, l.z_cpu, 1);

        copy_cpu(l.outputs*l.batch, ur.output, 1, l.r_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wr.output, 1, l.r_cpu, 1);

        activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
        mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);

        s.input = l.forgot_state;
        forward_connected_layer(wh, s);

        copy_cpu(l.outputs*l.batch, uh.output, 1, l.h_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, wh.output, 1, l.h_cpu, 1);

        if(l.tanh)
        {
            activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        }
        else
        {
            activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
        }

        weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

        net.input += l.inputs*l.batch;
        l.output += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer(layer l, network net)
{
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
            row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

image mask_to_rgb(image mask)
{
    int n = mask.c;
    image im = make_image(mask.w, mask.h, 3);
    int i, j;
    for(j = 0; j < n; ++j)
    {
        int offset = j*123457 % n;
        float red = get_color(2,offset,n);
        float green = get_color(1,offset,n);
        float blue = get_color(0,offset,n);
        for(i = 0; i < im.w*im.h; ++i)
        {
            im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
            im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
            im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
        }
    }
    return im;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

static float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
                dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
                (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
                dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}


void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k)
    {
        for(y = 0; y < source.h; ++y)
        {
            for(x = 0; x < source.w; ++x)
            {
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k)
    {
        for(y = 0; y < b.h; ++y)
        {
            for(x = 0; x < b.w; ++x)
            {
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

image tile_images(image a, image b, int dx)
{
    if(a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0);
    composite_image(b, c, a.w + dx, 0);
    return c;
}

image get_label(image **characters, char *string, int size)
{
    size = size/10;
    if(size > 7) size = 7;
    image label = make_empty_image(0,0,0);
    while(*string)
    {
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j)
    {
        for(i = 0; i < w && i + c < a.w; ++i)
        {
            for(k = 0; k < label.c; ++k)
            {
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i)
    {
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i)
    {
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i)
    {
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i)
    {
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = (image**)calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j)
    {
        alphabets[j] = (image*)calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i)
        {
            char buff[256];
            sprintf(buff, (char*)"data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i)
    {
        char labelstr[4096] = {0};
        int classs = -1;
        for(j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                if (classs < 0)
                {
                    strcat(labelstr, names[j]);
                    classs = j;
                }
                else
                {
                    strcat(labelstr, (char*)", (char*)");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if(classs >= 0)
        {
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = classs*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet)
            {
                image label = get_label(alphabet, labelstr, (im.h*.03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (dets[i].mask)
            {
                image mask = float_to_image(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }
}

void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c)
    {
        for(n = 0; n < im.w-1; ++n)
        {
            for(m = n + 1; m < im.w; ++m)
            {
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}

void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i)
    {
        for(c = 0; c < im.c; ++c)
        {
            for(x = 0; x < n/2; ++x)
            {
                for(y = 0; y < (n-1)/2 + 1; ++y)
                {
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}

void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k)
    {
        for(i = 0; i < a.h; ++i)
        {
            for(j = 0; j < a.w/2; ++j)
            {
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i)
    {
        for(j = 0; j < a.h*a.w; ++j)
        {
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j)
    {
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}

void ghost_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    float max_dist = sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
    for(k = 0; k < source.c; ++k)
    {
        for(y = 0; y < source.h; ++y)
        {
            for(x = 0; x < source.w; ++x)
            {
                float dist = sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                float alpha = (1 - dist/max_dist);
                if(alpha < 0) alpha = 0;
                float v1 = get_pixel(source, x,y,k);
                float v2 = get_pixel(dest, dx+x,dy+y,k);
                float val = alpha*v1 + (1-alpha)*v2;
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void blocky_image(image im, int s)
{
    int i,j,k;
    for(k = 0; k < im.c; ++k)
    {
        for(j = 0; j < im.h; ++j)
        {
            for(i = 0; i < im.w; ++i)
            {
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
            }
        }
    }
}

void censor_image(image im, int dx, int dy, int w, int h)
{
    int i,j,k;
    int s = 32;
    if(dx < 0) dx = 0;
    if(dy < 0) dy = 0;

    for(k = 0; k < im.c; ++k)
    {
        for(j = dy; j < dy + h && j < im.h; ++j)
        {
            for(i = dx; i < dx + w && i < im.w; ++i)
            {
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
                //im.data[i + j*im.w + k*im.w*im.h] = 0;
            }
        }
    }
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k)
    {
        for(y = 0; y < source.h; ++y)
        {
            for(x = 0; x < source.w; ++x)
            {
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i)
    {
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i)
    {
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i)
    {
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001)
    {
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i)
    {
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

void normalize_image2(image p)
{
    float *min = (float*)calloc(p.c, sizeof(float));
    float *max = (float*)calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j)
    {
        for(i = 0; i < p.h*p.w; ++i)
        {
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i)
    {
        if(max[i] - min[i] < .000000001)
        {
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j)
    {
        for(i = 0; i < p.w*p.h; ++i)
        {
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}

void copy_image_into(image src, image dest)
{
    memcpy(dest.data, src.data, src.h*src.w*src.c*sizeof(float));
}

image copy_image(image p)
{
    image copy = p;
    copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

void save_image_options(image im, const char *name, IMTYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, (char*)"%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, (char*)"%s.png", name);
    else if (f == BMP) sprintf(buff, (char*)"%s.bmp", name);
    else if (f == TGA) sprintf(buff, (char*)"%s.tga", name);
    else if (f == JPG) sprintf(buff, (char*)"%s.jpg", name);
    else               sprintf(buff, (char*)"%s.png", name);
    unsigned char *data = (unsigned char *)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k)
    {
        for(i = 0; i < im.w*im.h; ++i)
        {
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    // if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    // else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    // else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    // else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    // if(!success) fprintf(stderr, (char*)"Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
    save_image_options(im, name, JPG, 80);
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

image make_random_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    int i;
    for(i = 0; i < w*h*c; ++i)
    {
        out.data[i] = (rand_normal() * .25) + .5;
    }
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}

void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
    int x, y, c;
    for(c = 0; c < im.c; ++c)
    {
        for(y = 0; y < h; ++y)
        {
            for(x = 0; x < w; ++x)
            {
                float rx = ((float)x / w) * im.w;
                float ry = ((float)y / h) * im.h;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}

image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}

image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c)
    {
        for(y = 0; y < h; ++y)
        {
            for(x = 0; x < w; ++x)
            {
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

image rotate_image(image im, float rad)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(im.w, im.h, im.c);
    for(c = 0; c < im.c; ++c)
    {
        for(y = 0; y < im.h; ++y)
        {
            for(x = 0; x < im.w; ++x)
            {
                float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void translate_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k)
    {
        for(j = 0; j < h; ++j)
        {
            for(i = 0; i < w; ++i)
            {
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2)
    {
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance)
        {
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0)
    {
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else
    {
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i)
    {
        c.data[i] = a.data[i];
    }
    save_image(c, out);
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h))
    {
        new_w = w;
        new_h = (im.h * w)/im.w;
    }
    else
    {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h))
    {
        new_w = w;
        new_h = (im.h * w)/im.w;
    }
    else
    {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);
    return boxed;
}

image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h)
    {
        h = (h * max) / w;
        w = max;
    }
    else
    {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if(w < h)
    {
        h = (h * min) / w;
        w = min;
    }
    else
    {
        w = (w * min) / h;
        h = min;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}

augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = {0};
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - w) / 2.;
    float dy = (im.h*scale - w) / 2.;
    //if(dx < 0) dx = 0;
    //if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    a.rad = rad;
    a.scale = scale;
    a.w = w;
    a.h = h;
    a.dx = dx;
    a.dy = dy;
    a.aspect = aspect;
    return a;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
    image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    return crop;
}

float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void yuv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            y = get_pixel(im, i, j, 0);
            u = get_pixel(im, i, j, 1);
            v = get_pixel(im, i, j, 2);

            r = y + 1.13983*v;
            g = y + -.39465*u + -.58060*v;
            b = y + 2.03211*u;

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void rgb_to_yuv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            r = get_pixel(im, i, j, 0);
            g = get_pixel(im, i, j, 1);
            b = get_pixel(im, i, j, 2);

            y = .299*r + .587*g + .114*b;
            u = -.14713*r + -.28886*g + .436*b;
            v = .615*r + -.51499*g + -.10001*b;

            set_pixel(im, i, j, 0, y);
            set_pixel(im, i, j, 1, u);
            set_pixel(im, i, j, 2, v);
        }
    }
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            r = get_pixel(im, i, j, 0);
            g = get_pixel(im, i, j, 1);
            b = get_pixel(im, i, j, 2);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0)
            {
                s = 0;
                h = 0;
            }
            else
            {
                s = delta/max;
                if(r == max)
                {
                    h = (g - b) / delta;
                }
                else if (g == max)
                {
                    h = 2 + (b - r) / delta;
                }
                else
                {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            h = 6 * get_pixel(im, i, j, 0);
            s = get_pixel(im, i, j, 1);
            v = get_pixel(im, i, j, 2);
            if (s == 0)
            {
                r = g = b = v;
            }
            else
            {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0)
                {
                    r = v;
                    g = t;
                    b = p;
                }
                else if(index == 1)
                {
                    r = q;
                    g = v;
                    b = p;
                }
                else if(index == 2)
                {
                    r = p;
                    g = v;
                    b = t;
                }
                else if(index == 3)
                {
                    r = p;
                    g = q;
                    b = v;
                }
                else if(index == 4)
                {
                    r = t;
                    g = p;
                    b = v;
                }
                else
                {
                    r = v;
                    g = p;
                    b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void grayscale_image_3c(image im)
{
    assert(im.c == 3);
    int i, j, k;
    float scale[] = {0.299, 0.587, 0.114};
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            float val = 0;
            for(k = 0; k < 3; ++k)
            {
                val += scale[k]*get_pixel(im, i, j, k);
            }
            im.data[0*im.h*im.w + im.w*j + i] = val;
            im.data[1*im.h*im.w + im.w*j + i] = val;
            im.data[2*im.h*im.w + im.w*j + i] = val;
        }
    }
}

image grayscale_image(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k)
    {
        for(j = 0; j < im.h; ++j)
        {
            for(i = 0; i < im.w; ++i)
            {
                gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
            }
        }
    }
    return gray;
}

image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i)
    {
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}

image blend_image(image fore, image back, float alpha)
{
    assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
    image blend = make_image(fore.w, fore.h, fore.c);
    int i, j, k;
    for(k = 0; k < fore.c; ++k)
    {
        for(j = 0; j < fore.h; ++j)
        {
            for(i = 0; i < fore.w; ++i)
            {
                float val = alpha * get_pixel(fore, i, j, k) +
                            (1 - alpha)* get_pixel(back, i, j, k);
                set_pixel(blend, i, j, k, val);
            }
        }
    }
    return blend;
}

void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j)
    {
        for(i = 0; i < im.w; ++i)
        {
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

image binarize_image(image im)
{
    image c = copy_image(im);
    int i;
    for(i = 0; i < im.w * im.h * im.c; ++i)
    {
        if(c.data[i] > .5) c.data[i] = 1;
        else c.data[i] = 0;
    }
    return c;
}

void saturate_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void exposure_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void distort_image(image im, float hue, float sat, float val)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void random_distort_image(image im, float hue, float saturation, float exposure)
{
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k)
    {
        for(r = 0; r < im.h; ++r)
        {
            for(c = 0; c < w; ++c)
            {
                float val = 0;
                if(c == w-1 || im.w == 1)
                {
                    val = get_pixel(im, im.w-1, r, k);
                }
                else
                {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k)
    {
        for(r = 0; r < h; ++r)
        {
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c)
            {
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c)
            {
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    // unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    unsigned char *data = NULL;
    if (!data)
    {
        // fprintf(stderr, (char*)"Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k)
    {
        for(j = 0; j < h; ++j)
        {
            for(i = 0; i < w; ++i)
            {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(char *filename, int w, int h, int c)
{
    image out = load_image_stb(filename, c);

    if((h && w) && (h != out.h || w != out.w))
    {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i)
    {
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}
void print_image(image m)
{
    int i, j, k;
    for(i =0 ; i < m.c; ++i)
    {
        for(j =0 ; j < m.h; ++j)
        {
            for(k = 0; k < m.w; ++k)
            {
                printf("%.2lf, (char*)", m.data[i*m.h*m.w + j*m.w + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color)
    {
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i)
    {
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color)
        {
            embed_image(copy, filters, 0, h_offset);
        }
        else
        {
            for(j = 0; j < copy.c; ++j)
            {
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
}

image collapse_images_horz(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = ims[0].h;
    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color)
    {
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i)
    {
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color)
        {
            embed_image(copy, filters, w_offset, 0);
        }
        else
        {
            for(j = 0; j < copy.c; ++j)
            {
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
}

void free_image(image m)
{
    if(m.data)
    {
        free(m.data);
    }
}

layer make_iseg_layer(int batch, int w, int h, int classes, int ids)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = ISEG;

    l.h = h;
    l.w = w;
    l.c = classes + ids;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.batch = batch;
    l.extra = ids;
    l.cost = (float*)calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    l.truths = 90*(l.w*l.h+1);
    l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));

    l.counts = (int*)calloc(90, sizeof(int));
    l.sums = (float**)calloc(90, sizeof(float*));
    if(ids)
    {
        int i;
        for(i = 0; i < 90; ++i)
        {
            l.sums[i] = (float*)calloc(ids, sizeof(float));
        }
    }

    l.forward = forward_iseg_layer;
    l.backward = backward_iseg_layer;

    // fprintf(stderr, (char*)"iseg\n");
    srand(0);

    return l;
}

void resize_iseg_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->c;
    l->inputs = l->outputs;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

}

void forward_iseg_layer(const layer l, network net)
{

    double time = what_time_is_it_now();
    int i,b,j,k;
    int ids = l.extra;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

    for (b = 0; b < l.batch; ++b)
    {
        int index = b*l.outputs;
        activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
    }

    for (b = 0; b < l.batch; ++b)
    {
        // a priori, each pixel has no class
        for(i = 0; i < l.classes; ++i)
        {
            for(k = 0; k < l.w*l.h; ++k)
            {
                int index = b*l.outputs + i*l.w*l.h + k;
                l.delta[index] = 0 - l.output[index];
            }
        }

        // a priori, embedding should be small magnitude
        for(i = 0; i < ids; ++i)
        {
            for(k = 0; k < l.w*l.h; ++k)
            {
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] = .1 * (0 - l.output[index]);
            }
        }


        memset(l.counts, 0, 90*sizeof(int));
        for(i = 0; i < 90; ++i)
        {
            fill_cpu(ids, 0, l.sums[i], 1);

            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            // add up metric embeddings for each instance
            for(k = 0; k < l.w*l.h; ++k)
            {
                int index = b*l.outputs + c*l.w*l.h + k;
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v)
                {
                    l.delta[index] = v - l.output[index];
                    axpy_cpu(ids, 1, l.output + b*l.outputs + l.classes*l.w*l.h + k, l.w*l.h, l.sums[i], 1);
                    ++l.counts[i];
                }
            }
        }

        float *mse = (float*)calloc(90, sizeof(float));
        for(i = 0; i < 90; ++i)
        {
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            for(k = 0; k < l.w*l.h; ++k)
            {
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v)
                {
                    int z;
                    float sum = 0;
                    for(z = 0; z < ids; ++z)
                    {
                        int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                        sum += pow(l.sums[i][z]/l.counts[i] - l.output[index], 2);
                    }
                    mse[i] += sum;
                }
            }
            mse[i] /= l.counts[i];
        }

        // Calculate average embedding
        for(i = 0; i < 90; ++i)
        {
            if(!l.counts[i]) continue;
            scal_cpu(ids, 1.f/l.counts[i], l.sums[i], 1);
            if(b == 0 && net.gpu_index == 0)
            {
                printf("%4d, %6.3f, (char*)", l.counts[i], mse[i]);
                for(j = 0; j < ids; ++j)
                {
                    printf("%6.3f,", l.sums[i][j]);
                }
                printf("\n");
            }
        }
        free(mse);

        // Calculate embedding loss
        for(i = 0; i < 90; ++i)
        {
            if(!l.counts[i]) continue;
            for(k = 0; k < l.w*l.h; ++k)
            {
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v)
                {
                    for(j = 0; j < 90; ++j)
                    {
                        if(!l.counts[j])continue;
                        int z;
                        for(z = 0; z < ids; ++z)
                        {
                            int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                            float diff = l.sums[j][z] - l.output[index];
                            if (j == i) l.delta[index] +=   diff < 0? -.1 : .1;
                            else        l.delta[index] += -(diff < 0? -.1 : .1);
                        }
                    }
                }
            }
        }

        for(i = 0; i < ids; ++i)
        {
            for(k = 0; k < l.w*l.h; ++k)
            {
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] *= .01;
            }
        }
    }

    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("took %lf sec\n", what_time_is_it_now() - time);
}

void backward_iseg_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

layer make_l2norm_layer(int batch, int inputs)
{
    // fprintf(stderr, (char*)"l2norm                                         %4d\n",  inputs);
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.scales = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = (float*)calloc(inputs*batch, sizeof(float));

    l.forward = forward_l2norm_layer;
    l.backward = backward_l2norm_layer;

    return l;
}

void forward_l2norm_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer(const layer l, network net)
{
    //axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

void free_layer(layer l)
{
    if(l.type == DROPOUT)
    {
        if(l.rand)           free(l.rand);
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);
}

list *make_list()
{
    list *l = (list*)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l)
{
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}

void list_insert(list *l, void *val)
{
    node *neww = (node*)malloc(sizeof(node));
    neww->val = val;
    neww->next = 0;

    if(!l->back)
    {
        l->front = neww;
        neww->prev = 0;
    }
    else
    {
        l->back->next = neww;
        neww->prev = l->back;
    }
    l->back = neww;
    ++l->size;
}

void free_node(node *n)
{
    node *next;
    while(n)
    {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

void free_list_contents(list *l)
{
    node *n = l->front;
    while(n)
    {
        free(n->val);
        n = n->next;
    }
}

void **list_to_array(list *l)
{
    void **a = (void**)calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n)
    {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

int local_out_height(local_layer l)
{
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
}

int local_out_width(local_layer l)
{
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;
}

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation)
{
    int i;
    local_layer l;
    memset(&l,0,sizeof(local_layer));
    l.type = LOCAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;

    int out_h = local_out_height(l);
    int out_w = local_out_width(l);
    int locations = out_h*out_w;
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.weights = (float*)calloc(c*n*size*size*locations, sizeof(float));
    l.weight_updates = (float*)calloc(c*n*size*size*locations, sizeof(float));

    l.biases = (float*)calloc(l.outputs, sizeof(float));
    l.bias_updates = (float*)calloc(l.outputs, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,1);

    l.output = (float*)calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = (float*)calloc(l.batch*out_h * out_w * n, sizeof(float));

    l.workspace_size = out_h*out_w*size*size*c;

    l.forward = forward_local_layer;
    l.backward = backward_local_layer;
    l.update = update_local_layer;

    l.activation = activation;

    // fprintf(stderr, (char*)"Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void forward_local_layer(const local_layer l, network net)
{
    int out_h = local_out_height(l);
    int out_w = local_out_width(l);
    int i, j;
    int locations = out_h * out_w;

    for(i = 0; i < l.batch; ++i)
    {
        copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1);
    }

    for(i = 0; i < l.batch; ++i)
    {
        float *input = net.input + i*l.w*l.h*l.c;
        im2col_cpu(input, l.c, l.h, l.w,
                   l.size, l.stride, l.pad, net.workspace);
        float *output = l.output + i*l.outputs;
        for(j = 0; j < locations; ++j)
        {
            float *a = l.weights + j*l.size*l.size*l.c*l.n;
            float *b = net.workspace + j;
            float *c = output + j;

            int m = l.n;
            int n = 1;
            int k = l.size*l.size*l.c;

            gemm(0,0,m,n,k,1,a,k,b,locations,1,c,locations);
        }
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_local_layer(local_layer l, network net)
{
    int i, j;
    int locations = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    for(i = 0; i < l.batch; ++i)
    {
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }

    for(i = 0; i < l.batch; ++i)
    {
        float *input = net.input + i*l.w*l.h*l.c;
        im2col_cpu(input, l.c, l.h, l.w,
                   l.size, l.stride, l.pad, net.workspace);

        for(j = 0; j < locations; ++j)
        {
            float *a = l.delta + i*l.outputs + j;
            float *b = net.workspace + j;
            float *c = l.weight_updates + j*l.size*l.size*l.c*l.n;
            int m = l.n;
            int n = l.size*l.size*l.c;
            int k = 1;

            gemm(0,1,m,n,k,1,a,locations,b,locations,1,c,n);
        }

        if(net.delta)
        {
            for(j = 0; j < locations; ++j)
            {
                float *a = l.weights + j*l.size*l.size*l.c*l.n;
                float *b = l.delta + i*l.outputs + j;
                float *c = net.workspace + j;

                int m = l.size*l.size*l.c;
                int n = 1;
                int k = l.n;

                gemm(1,0,m,n,k,1,a,m,b,locations,0,c,locations);
            }

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_local_layer(local_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int locations = l.out_w*l.out_h;
    int size = l.size*l.size*l.c*l.n*locations;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}

layer make_logistic_layer(int batch, int inputs)
{
    // fprintf(stderr, (char*)"logistic x entropy                             %4d\n",  inputs);
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)calloc(inputs*batch, sizeof(float));
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;

    return l;
}

void forward_logistic_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    if(net.truth)
    {
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_logistic_layer(const layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
    // fprintf(stderr, (char*)"LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l;
    memset(&l,0,sizeof(layer));
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;

    l.uf = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.uf) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uf->batch = batch;

    l.ui = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.ui) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ui->batch = batch;

    l.ug = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.ug) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.ug->batch = batch;

    l.uo = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.uo) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l.uo->batch = batch;

    l.wf = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wf) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wf->batch = batch;

    l.wi = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wi) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wi->batch = batch;

    l.wg = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wg) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wg->batch = batch;

    l.wo = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.wo) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l.wo->batch = batch;

    l.batch_normalize = batch_normalize;
    l.outputs = outputs;

    l.output = (float*)calloc(outputs*batch*steps, sizeof(float));
    l.state = (float*)calloc(outputs*batch, sizeof(float));

    l.forward = forward_lstm_layer;
    l.update = update_lstm_layer;

    l.prev_state_cpu =  (float*)calloc(batch*outputs, sizeof(float));
    l.prev_cell_cpu =   (float*)calloc(batch*outputs, sizeof(float));
    l.cell_cpu =        (float*)calloc(batch*outputs*steps, sizeof(float));

    l.f_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.i_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.g_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.o_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.c_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.h_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l.temp_cpu =        (float*)calloc(batch*outputs, sizeof(float));
    l.temp2_cpu =       (float*)calloc(batch*outputs, sizeof(float));
    l.temp3_cpu =       (float*)calloc(batch*outputs, sizeof(float));
    l.dc_cpu =          (float*)calloc(batch*outputs, sizeof(float));
    l.dh_cpu =          (float*)calloc(batch*outputs, sizeof(float));

    return l;
}

void update_lstm_layer(layer l, update_args a)
{
    update_connected_layer(*(l.wf), a);
    update_connected_layer(*(l.wi), a);
    update_connected_layer(*(l.wg), a);
    update_connected_layer(*(l.wo), a);
    update_connected_layer(*(l.uf), a);
    update_connected_layer(*(l.ui), a);
    update_connected_layer(*(l.ug), a);
    update_connected_layer(*(l.uo), a);
}

void forward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
    if (state.train)
    {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }

    for (i = 0; i < l.steps; ++i)
    {
        s.input = l.h_cpu;
        forward_connected_layer(wf, s);
        forward_connected_layer(wi, s);
        forward_connected_layer(wg, s);
        forward_connected_layer(wo, s);

        s.input = state.input;
        forward_connected_layer(uf, s);
        forward_connected_layer(ui, s);
        forward_connected_layer(ug, s);
        forward_connected_layer(uo, s);

        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);

        state.input += l.inputs*l.batch;
        l.output    += l.outputs*l.batch;
        l.cell_cpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_lstm_layer(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps - 1);
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

    l.output += l.outputs*l.batch*(l.steps - 1);
    l.cell_cpu += l.outputs*l.batch*(l.steps - 1);
    l.delta += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i)
    {
        if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);

        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;

        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);

        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);

        copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);

        gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
        axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);
        mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wo, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uo, s);

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wg, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ug, s);

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wi, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(ui, s);

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input = l.prev_state_cpu;
        s.delta = l.dh_cpu;
        backward_connected_layer(wf, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer(uf, s);

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output -= l.outputs*l.batch;
        l.cell_cpu -= l.outputs*l.batch;
        l.delta -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}

void free_matrix(matrix m)
{
    int i;
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    free(m.vals);
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
    int *indexes = (int*)calloc(k, sizeof(int));
    int n = truth.cols;
    int i,j;
    int correct = 0;
    for(i = 0; i < truth.rows; ++i)
    {
        top_k(guess.vals[i], n, k, indexes);
        for(j = 0; j < k; ++j)
        {
            int classs = indexes[j];
            if(truth.vals[i][classs])
            {
                ++correct;
                break;
            }
        }
    }
    free(indexes);
    return (float)correct/truth.rows;
}

void scale_matrix(matrix m, float scale)
{
    int i,j;
    for(i = 0; i < m.rows; ++i)
    {
        for(j = 0; j < m.cols; ++j)
        {
            m.vals[i][j] *= scale;
        }
    }
}

matrix resize_matrix(matrix m, int size)
{
    int i;
    if (m.rows == size) return m;
    if (m.rows < size)
    {
        m.vals = (float**)realloc(m.vals, size*sizeof(float*));
        for (i = m.rows; i < size; ++i)
        {
            m.vals[i] = (float*)calloc(m.cols, sizeof(float));
        }
    }
    else if (m.rows > size)
    {
        for (i = size; i < m.rows; ++i)
        {
            free(m.vals[i]);
        }
        m.vals = (float**)realloc(m.vals, size*sizeof(float*));
    }
    m.rows = size;
    return m;
}

void matrix_add_matrix(matrix from, matrix to)
{
    assert(from.rows == to.rows && from.cols == to.cols);
    int i,j;
    for(i = 0; i < from.rows; ++i)
    {
        for(j = 0; j < from.cols; ++j)
        {
            to.vals[i][j] += from.vals[i][j];
        }
    }
}

matrix copy_matrix(matrix m)
{
    matrix c = {0};
    c.rows = m.rows;
    c.cols = m.cols;
    c.vals = (float**)calloc(c.rows, sizeof(float *));
    int i;
    for(i = 0; i < c.rows; ++i)
    {
        c.vals[i] = (float*)calloc(c.cols, sizeof(float));
        copy_cpu(c.cols, m.vals[i], 1, c.vals[i], 1);
    }
    return c;
}

matrix make_matrix(int rows, int cols)
{
    int i;
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = (float**)calloc(m.rows, sizeof(float *));
    for(i = 0; i < m.rows; ++i)
    {
        m.vals[i] = (float*)calloc(m.cols, sizeof(float));
    }
    return m;
}

matrix hold_out_matrix(matrix *m, int n)
{
    int i;
    matrix h;
    h.rows = n;
    h.cols = m->cols;
    h.vals = (float**)calloc(h.rows, sizeof(float *));
    for(i = 0; i < n; ++i)
    {
        int index = rand()%m->rows;
        h.vals[i] = m->vals[index];
        m->vals[index] = m->vals[--(m->rows)];
    }
    return h;
}

float *pop_column(matrix *m, int c)
{
    float *col = (float*)calloc(m->rows, sizeof(float));
    int i, j;
    for(i = 0; i < m->rows; ++i)
    {
        col[i] = m->vals[i][c];
        for(j = c; j < m->cols-1; ++j)
        {
            m->vals[i][j] = m->vals[i][j+1];
        }
    }
    --m->cols;
    return col;
}

matrix csv_to_matrix(char *filename)
{
    FILE *fp = fopen(filename, (char*)"r");
    if(!fp) file_error(filename);

    matrix m;
    m.cols = -1;

    char *line;

    int n = 0;
    int size = 1024;
    m.vals = (float**)calloc(size, sizeof(float*));
    while((line = fgetl(fp)))
    {
        if(m.cols == -1) m.cols = count_fields(line);
        if(n == size)
        {
            size *= 2;
            m.vals = (float**)realloc(m.vals, size*sizeof(float*));
        }
        m.vals[n] = parse_fields(line, m.cols);
        free(line);
        ++n;
    }
    m.vals = (float**)realloc(m.vals, n*sizeof(float*));
    m.rows = n;
    return m;
}

void matrix_to_csv(matrix m)
{
    int i, j;

    for(i = 0; i < m.rows; ++i)
    {
        for(j = 0; j < m.cols; ++j)
        {
            if(j > 0) printf(",");
            printf("%.17g", m.vals[i][j]);
        }
        printf("\n");
    }
}

void print_matrix(matrix m)
{
    int i, j;
    printf("%d X %d Matrix:\n",m.rows, m.cols);
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i)
    {
        printf("|  ");
        for(j = 0; j < m.cols; ++j)
        {
            printf("%15.7f ", m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l;
    memset(&l,0,sizeof(maxpool_layer));
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*)calloc(output_size, sizeof(int));
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    // fprintf(stderr, (char*)"max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = (int*)realloc(l->indexes, output_size * sizeof(int));
    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b)
    {
        for(k = 0; k < c; ++k)
        {
            for(i = 0; i < h; ++i)
            {
                for(j = 0; j < w; ++j)
                {
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n)
                    {
                        for(m = 0; m < l.size; ++m)
                        {
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i)
    {
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0)
    {
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy)
    {
    case CONSTANT:
        return net->learning_rate;
    case STEP:
        return net->learning_rate * pow(net->scale, batch_num/net->step);
    case STEPS:
        rate = net->learning_rate;
        for(i = 0; i < net->num_steps; ++i)
        {
            if(net->steps[i] > batch_num) return rate;
            rate *= net->scales[i];
        }
        return rate;
    case EXP:
        return net->learning_rate * pow(net->gamma, batch_num);
    case POLY:
        return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
    case RANDOM:
        return net->learning_rate * pow(rand_uniform(0,1), net->power);
    case SIG:
        return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
    default:
        // fprintf(stderr, (char*)"Policy is weird!\n");
        return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a)
    {
    case CONVOLUTIONAL:
        return (char*)"convolutional";
    case ACTIVE:
        return (char*)"activation";
    case LOCAL:
        return (char*)"local";
    case DECONVOLUTIONAL:
        return (char*)"deconvolutional";
    case CONNECTED:
        return (char*)"connected";
    case RNN:
        return (char*)"rnn";
    case GRU:
        return (char*)"gru";
    case LSTM:
        return (char*)"lstm";
    case CRNN:
        return (char*)"crnn";
    case MAXPOOL:
        return (char*)"maxpool";
    case REORG:
        return (char*)"reorg";
    case AVGPOOL:
        return (char*)"avgpool";
    case SOFTMAX:
        return (char*)"softmax";
    case DETECTION:
        return (char*)"detection";
    case REGION:
        return (char*)"region";
    case YOLO:
        return (char*)"yolo";
    case DROPOUT:
        return (char*)"dropout";
    case CROP:
        return (char*)"crop";
    case COST:
        return (char*)"cost";
    case ROUTE:
        return (char*)"route";
    case SHORTCUT:
        return (char*)"shortcut";
    case NORMALIZATION:
        return (char*)"normalization";
    case BATCHNORM:
        return (char*)"batchnorm";
    default:
        break;
    }
    return (char*)"none";
}

network *make_network(int n)
{
    network *net = (network*)calloc(1, sizeof(network));
    net->n = n;
    net->layers = (layer*)calloc(net->n, sizeof(layer));
    net->seen = (size_t*)calloc(1, sizeof(size_t));
    net->t    = (int*)calloc(1, sizeof(int));
    net->cost = (float*)calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i)
    {
        net.index = i;
        layer l = net.layers[i];
        if(l.delta)
        {
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth)
        {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i)
    {
        layer l = net.layers[i];
        if(l.update)
        {
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i)
    {
        if(net.layers[i].cost)
        {
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i)
    {
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0)
        {
            net = orig;
        }
        else
        {
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i)
    {
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i)
    {
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i)
    {
        net->layers[i].temperature = t;
    }
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i)
    {
        net->layers[i].batch = b;
    }
}

int resize_network(network *net, int w, int h)
{
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    // fprintf(stderr, (char*)"Resizing to %d x %d...\n", w, h);
    // fflush(stderr);
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL)
        {
            resize_convolutional_layer(&l, w, h);
        }
        else if(l.type == CROP)
        {
            resize_crop_layer(&l, w, h);
        }
        else if(l.type == MAXPOOL)
        {
            resize_maxpool_layer(&l, w, h);
        }
        else if(l.type == REGION)
        {
            resize_region_layer(&l, w, h);
        }
        else if(l.type == YOLO)
        {
            resize_yolo_layer(&l, w, h);
        }
        else if(l.type == ROUTE)
        {
            resize_route_layer(&l, net);
        }
        else if(l.type == SHORTCUT)
        {
            resize_shortcut_layer(&l, w, h);
        }
        else if(l.type == UPSAMPLE)
        {
            resize_upsample_layer(&l, w, h);
        }
        else if(l.type == REORG)
        {
            resize_reorg_layer(&l, w, h);
        }
        else if(l.type == AVGPOOL)
        {
            resize_avgpool_layer(&l, w, h);
        }
        else if(l.type == NORMALIZATION)
        {
            resize_normalization_layer(&l, w, h);
        }
        else if(l.type == COST)
        {
            resize_cost_layer(&l, inputs);
        }
        else
        {
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
    free(net->workspace);
    net->workspace = (float*)calloc(1, workspace_size);
    // fprintf(stderr, (char*)" Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i)
    {
        if(net->layers[i].type == DETECTION)
        {
            return net->layers[i];
        }
    }
    // fprintf(stderr, (char*)"Detection layer not found!!\n");
    layer l;
    memset(&l,0,sizeof(layer));
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
    if (l.out_w && l.out_h && l.out_c)
    {
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i)
    {
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if(l.type == YOLO)
        {
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION)
        {
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i)
    {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        if(l.coords > 4)
        {
            dets[i].mask = (float*)calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j)
    {
        layer l = net->layers[j];
        if(l.type == YOLO)
        {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION)
        {
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION)
        {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net)
{
    return net->w;
}
int network_height(network *net)
{
    return net->h;
}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = (float*)calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch)
    {
        for(b = 0; b < net->batch; ++b)
        {
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m)
        {
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b)
            {
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j)
                {
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = (float*)calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch)
    {
        for(b = 0; b < net->batch; ++b)
        {
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b)
        {
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j)
            {
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        // fprintf(stderr, (char*)"Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        // if(n > 100) n = 100;
        // for(j = 0; j < n; ++j) fprintf(stderr, (char*)"%f, (char*)", output[j]);
        // if(n == 100)fprintf(stderr,".....\n");
        // fprintf(stderr, (char*)"\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i)
    {
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth)
        {
            if(p2 == truth) ++d;
            else ++c;
        }
        else
        {
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i)
    {
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i)
    {
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i)
    {
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    // fprintf(stderr, (char*)"Local Response Normalization Layer: %d x %d x %d image, %d size\n", w,h,c,size);
    layer layer;
    memset(&layer,0,sizeof(layer));
    layer.type = NORMALIZATION;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.kappa = kappa;
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.output = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.delta = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.squared = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.norms = (float*)calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.forward = forward_normalization_layer;
    layer.backward = backward_normalization_layer;

    return layer;
}

void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
    layer->output = (float*)realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = (float*)realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = (float*)realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = (float*)realloc(layer->norms, h * w * c * batch * sizeof(float));

}

void forward_normalization_layer(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_cpu(w*h*c*layer.batch, 0, layer.squared, 1);

    for(b = 0; b < layer.batch; ++b)
    {
        float *squared = layer.squared + w*h*c*b;
        float *norms   = layer.norms + w*h*c*b;
        float *input   = net.input + w*h*c*b;
        pow_cpu(w*h*c, 2, input, 1, squared, 1);

        const_cpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k)
        {
            axpy_cpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k)
        {
            copy_cpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_cpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_cpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    mul_cpu(w*h*c*layer.batch, net.input, 1, layer.output, 1);
}

void backward_normalization_layer(const layer layer, network net)
{
    // TODO This is approximate ;-)
    // Also this should add in to delta instead of overwritting.

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    mul_cpu(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);
}

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, (char*)"r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0)
    {
        ++ nu;
        strip(line);
        switch(line[0])
        {
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if(!read_option(line, options))
            {
                // fprintf(stderr, (char*)"Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, (char*)"names", 0);
    if(!name_list) name_list = option_find_str(options, (char*)"labels", 0);
    if(!name_list)
    {
        // fprintf(stderr, (char*)"No names or labels found\n");
    }
    else
    {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, (char*)"classes", 2);
    free_list(options);
    return m;
}

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i)
    {
        if(s[i] == '=')
        {
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = (kvp*)malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n)
    {
        kvp *p = (kvp *)n->val;
        //if(!p->used){
        //fprintf(stderr, (char*)"Unused field: '%s = %s'\n", p->key, p->val);
        //}
        n = n->next;
    }
}

char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n)
    {
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0)
        {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    // if(def) fprintf(stderr, (char*)"%s: Using default '%s'\n", key, def);
    return def;
}

int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    // fprintf(stderr, (char*)"%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    // fprintf(stderr, (char*)"%s: Using default '%lf'\n", key, def);
    return def;
}

typedef struct
{
    char *type;
    list *options;
} section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, (char*)"[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, (char*)"[crop]")==0) return CROP;
    if (strcmp(type, (char*)"[cost]")==0) return COST;
    if (strcmp(type, (char*)"[detection]")==0) return DETECTION;
    if (strcmp(type, (char*)"[region]")==0) return REGION;
    if (strcmp(type, (char*)"[yolo]")==0) return YOLO;
    if (strcmp(type, (char*)"[iseg]")==0) return ISEG;
    if (strcmp(type, (char*)"[local]")==0) return LOCAL;
    if (strcmp(type, (char*)"[conv]")==0
            || strcmp(type, (char*)"[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, (char*)"[deconv]")==0
            || strcmp(type, (char*)"[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, (char*)"[activation]")==0) return ACTIVE;
    if (strcmp(type, (char*)"[logistic]")==0) return LOGXENT;
    if (strcmp(type, (char*)"[l2norm]")==0) return L2NORM;
    if (strcmp(type, (char*)"[net]")==0
            || strcmp(type, (char*)"[network]")==0) return NETWORK;
    if (strcmp(type, (char*)"[crnn]")==0) return CRNN;
    if (strcmp(type, (char*)"[gru]")==0) return GRU;
    if (strcmp(type, (char*)"[lstm]") == 0) return LSTM;
    if (strcmp(type, (char*)"[rnn]")==0) return RNN;
    if (strcmp(type, (char*)"[conn]")==0
            || strcmp(type, (char*)"[connected]")==0) return CONNECTED;
    if (strcmp(type, (char*)"[max]")==0
            || strcmp(type, (char*)"[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, (char*)"[reorg]")==0) return REORG;
    if (strcmp(type, (char*)"[avg]")==0
            || strcmp(type, (char*)"[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, (char*)"[dropout]")==0) return DROPOUT;
    if (strcmp(type, (char*)"[lrn]")==0
            || strcmp(type, (char*)"[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, (char*)"[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, (char*)"[soft]")==0
            || strcmp(type, (char*)"[softmax]")==0) return SOFTMAX;
    if (strcmp(type, (char*)"[route]")==0) return ROUTE;
    if (strcmp(type, (char*)"[upsample]")==0) return UPSAMPLE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n)
    {
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i)
    {
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, (char*)"%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params
{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, (char*)"filters",1);
    int size = option_find_int(options, (char*)"size",1);
    int stride = option_find_int(options, (char*)"stride",1);
    int pad = option_find_int(options, (char*)"pad",0);
    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, (char*)"filters",1);
    int size = option_find_int(options, (char*)"size",1);
    int stride = option_find_int(options, (char*)"stride",1);

    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);
    int pad = option_find_int_quiet(options, (char*)"pad",0);
    int padding = option_find_int_quiet(options, (char*)"padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
}


convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, (char*)"filters",1);
    int size = option_find_int(options, (char*)"size",1);
    int stride = option_find_int(options, (char*)"stride",1);
    int pad = option_find_int_quiet(options, (char*)"pad",0);
    int padding = option_find_int_quiet(options, (char*)"padding",0);
    int groups = option_find_int_quiet(options, (char*)"groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);
    int binary = option_find_int_quiet(options, (char*)"binary", 0);
    int xnor = option_find_int_quiet(options, (char*)"xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, (char*)"flipped", 0);
    layer.dot = option_find_float_quiet(options, (char*)"dot", 0);

    return layer;
}

layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, (char*)"output_filters",1);
    int hidden_filters = option_find_int(options, (char*)"hidden_filters",1);
    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, (char*)"shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, (char*)"output",1);
    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, (char*)"shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, (char*)"output",1);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, (char*)"tanh", 0);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, (char*)"output", 1);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, (char*)"output",1);
    char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, (char*)"groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, (char*)"temperature", 1);
    char *tree_file = option_find_str(options, (char*)"tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, (char*)"spatial", 0);
    l.noloss =  option_find_int_quiet(options, (char*)"noloss", 0);
    return l;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i)
        {
            if (a[i] == ',') ++n;
        }
        mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i)
        {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, (char*)"classes", 20);
    int total = option_find_int(options, (char*)"num", 1);
    int num = total;

    char *a = option_find_str(options, (char*)"mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, (char*)"max",90);
    l.jitter = option_find_float(options, (char*)"jitter", .2);

    l.ignore_thresh = option_find_float(options, (char*)"ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, (char*)"truth_thresh", 1);
    l.random = option_find_int_quiet(options, (char*)"random", 0);

    char *map_file = option_find_str(options, (char*)"map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, (char*)"anchors", 0);
    if(a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i)
        {
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i)
        {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, (char*)"classes", 20);
    int ids = option_find_int(options, (char*)"ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, (char*)"coords", 4);
    int classes = option_find_int(options, (char*)"classes", 20);
    int num = option_find_int(options, (char*)"num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, (char*)"log", 0);
    l.sqrt = option_find_int_quiet(options, (char*)"sqrt", 0);

    l.softmax = option_find_int(options, (char*)"softmax", 0);
    l.background = option_find_int_quiet(options, (char*)"background", 0);
    l.max_boxes = option_find_int_quiet(options, (char*)"max",30);
    l.jitter = option_find_float(options, (char*)"jitter", .2);
    l.rescore = option_find_int_quiet(options, (char*)"rescore",0);

    l.thresh = option_find_float(options, (char*)"thresh", .5);
    l.classfix = option_find_int_quiet(options, (char*)"classfix", 0);
    l.absolute = option_find_int_quiet(options, (char*)"absolute", 0);
    l.random = option_find_int_quiet(options, (char*)"random", 0);

    l.coord_scale = option_find_float(options, (char*)"coord_scale", 1);
    l.object_scale = option_find_float(options, (char*)"object_scale", 1);
    l.noobject_scale = option_find_float(options, (char*)"noobject_scale", 1);
    l.mask_scale = option_find_float(options, (char*)"mask_scale", 1);
    l.class_scale = option_find_float(options, (char*)"class_scale", 1);
    l.bias_match = option_find_int_quiet(options, (char*)"bias_match",0);

    char *tree_file = option_find_str(options, (char*)"tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, (char*)"map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, (char*)"anchors", 0);
    if(a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i)
        {
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i)
        {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, (char*)"coords", 1);
    int classes = option_find_int(options, (char*)"classes", 1);
    int rescore = option_find_int(options, (char*)"rescore", 0);
    int num = option_find_int(options, (char*)"num", 1);
    int side = option_find_int(options, (char*)"side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, (char*)"softmax", 0);
    layer.sqrt = option_find_int(options, (char*)"sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, (char*)"max",90);
    layer.coord_scale = option_find_float(options, (char*)"coord_scale", 1);
    layer.forced = option_find_int(options, (char*)"forced", 0);
    layer.object_scale = option_find_float(options, (char*)"object_scale", 1);
    layer.noobject_scale = option_find_float(options, (char*)"noobject_scale", 1);
    layer.class_scale = option_find_float(options, (char*)"class_scale", 1);
    layer.jitter = option_find_float(options, (char*)"jitter", .2);
    layer.random = option_find_int_quiet(options, (char*)"random", 0);
    layer.reorg = option_find_int_quiet(options, (char*)"reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, (char*)"type", (char*)"sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, (char*)"scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, (char*)"ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, (char*)"noobj", 1);
    layer.thresh =  option_find_float_quiet(options, (char*)"thresh",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, (char*)"crop_height",1);
    int crop_width = option_find_int(options, (char*)"crop_width",1);
    int flip = option_find_int(options, (char*)"flip",0);
    float angle = option_find_float(options, (char*)"angle",0);
    float saturation = option_find_float(options, (char*)"saturation",1);
    float exposure = option_find_float(options, (char*)"exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, (char*)"noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, (char*)"shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, (char*)"stride",1);
    int reverse = option_find_int_quiet(options, (char*)"reverse",0);
    int flatten = option_find_int_quiet(options, (char*)"flatten",0);
    int extra = option_find_int_quiet(options, (char*)"extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, (char*)"stride",1);
    int size = option_find_int(options, (char*)"size",stride);
    int padding = option_find_int_quiet(options, (char*)"padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, (char*)"probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, (char*)"alpha", .0001);
    float beta =  option_find_float(options, (char*)"beta", .75);
    float kappa = option_find_float(options, (char*)"kappa", 1);
    int size = option_find_int(options, (char*)"size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, (char*)"from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, (char*)"activation", (char*)"linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, (char*)"alpha", 1);
    s.beta = option_find_float_quiet(options, (char*)"beta", 1);
    return s;
}


layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}


layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, (char*)"activation", (char*)"linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, (char*)"stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, (char*)"scale", 1);
    return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, (char*)"layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i)
    {
        if (l[i] == ',') ++n;
    }

    int *layers = (int*)calloc(n, sizeof(int));
    int *sizes = (int*)calloc(n, sizeof(int));
    for(i = 0; i < n; ++i)
    {
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i)
    {
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h)
        {
            layer.out_c += next.out_c;
        }
        else
        {
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, (char*)"random")==0) return RANDOM;
    if (strcmp(s, (char*)"poly")==0) return POLY;
    if (strcmp(s, (char*)"constant")==0) return CONSTANT;
    if (strcmp(s, (char*)"step")==0) return STEP;
    if (strcmp(s, (char*)"exp")==0) return EXP;
    if (strcmp(s, (char*)"sigmoid")==0) return SIG;
    if (strcmp(s, (char*)"steps")==0) return STEPS;
    // fprintf(stderr, (char*)"Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, (char*)"batch",1);
    net->learning_rate = option_find_float(options, (char*)"learning_rate", .001);
    net->momentum = option_find_float(options, (char*)"momentum", .9);
    net->decay = option_find_float(options, (char*)"decay", .0001);
    int subdivs = option_find_int(options, (char*)"subdivisions",1);
    net->time_steps = option_find_int_quiet(options, (char*)"time_steps",1);
    net->notruth = option_find_int_quiet(options, (char*)"notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, (char*)"random", 0);

    net->adam = option_find_int_quiet(options, (char*)"adam", 0);
    if(net->adam)
    {
        net->B1 = option_find_float(options, (char*)"B1", .9);
        net->B2 = option_find_float(options, (char*)"B2", .999);
        net->eps = option_find_float(options, (char*)"eps", .0000001);
    }

    net->h = option_find_int_quiet(options, (char*)"height",0);
    net->w = option_find_int_quiet(options, (char*)"width",0);
    net->c = option_find_int_quiet(options, (char*)"channels",0);
    net->inputs = option_find_int_quiet(options, (char*)"inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, (char*)"max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, (char*)"min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, (char*)"max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, (char*)"min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, (char*)"center",0);
    net->clip = option_find_float_quiet(options, (char*)"clip", 0);

    net->angle = option_find_float_quiet(options, (char*)"angle", 0);
    net->aspect = option_find_float_quiet(options, (char*)"aspect", 1);
    net->saturation = option_find_float_quiet(options, (char*)"saturation", 1);
    net->exposure = option_find_float_quiet(options, (char*)"exposure", 1);
    net->hue = option_find_float_quiet(options, (char*)"hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, (char*)"policy", (char*)"constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, (char*)"burn_in", 0);
    net->power = option_find_float_quiet(options, (char*)"power", 4);
    if(net->policy == STEP)
    {
        net->step = option_find_int(options, (char*)"step", 1);
        net->scale = option_find_float(options, (char*)"scale", 1);
    }
    else if (net->policy == STEPS)
    {
        char *l = option_find(options, (char*)"steps");
        char *p = option_find(options, (char*)"scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i)
        {
            if (l[i] == ',') ++n;
        }
        int *steps = (int*)calloc(n, sizeof(int));
        float *scales = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i)
        {
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP)
    {
        net->gamma = option_find_float(options, (char*)"gamma", 1);
    }
    else if (net->policy == SIG)
    {
        net->gamma = option_find_float(options, (char*)"gamma", 1);
        net->step = option_find_int(options, (char*)"step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM)
    {
    }
    net->max_batches = option_find_int(options, (char*)"max_batches", 0);
}

int is_network(section *s)
{
    return (strcmp(s->type, (char*)"[net]")==0
            || strcmp(s->type, (char*)"[network]")==0);
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    // fprintf(stderr, (char*)"layer     filters    size              input                output\n");
    while(n)
    {
        params.index = count;
        // fprintf(stderr, (char*)"%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l;
        memset(&l,0,sizeof(layer));
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL)
        {
            l = parse_convolutional(options, params);
        }
        else if(lt == DECONVOLUTIONAL)
        {
            l = parse_deconvolutional(options, params);
        }
        else if(lt == LOCAL)
        {
            l = parse_local(options, params);
        }
        else if(lt == ACTIVE)
        {
            l = parse_activation(options, params);
        }
        else if(lt == LOGXENT)
        {
            l = parse_logistic(options, params);
        }
        else if(lt == L2NORM)
        {
            l = parse_l2norm(options, params);
        }
        else if(lt == RNN)
        {
            l = parse_rnn(options, params);
        }
        else if(lt == GRU)
        {
            l = parse_gru(options, params);
        }
        else if (lt == LSTM)
        {
            l = parse_lstm(options, params);
        }
        else if(lt == CRNN)
        {
            l = parse_crnn(options, params);
        }
        else if(lt == CONNECTED)
        {
            l = parse_connected(options, params);
        }
        else if(lt == CROP)
        {
            l = parse_crop(options, params);
        }
        else if(lt == COST)
        {
            l = parse_cost(options, params);
        }
        else if(lt == REGION)
        {
            l = parse_region(options, params);
        }
        else if(lt == YOLO)
        {
            l = parse_yolo(options, params);
        }
        else if(lt == ISEG)
        {
            l = parse_iseg(options, params);
        }
        else if(lt == DETECTION)
        {
            l = parse_detection(options, params);
        }
        else if(lt == SOFTMAX)
        {
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }
        else if(lt == NORMALIZATION)
        {
            l = parse_normalization(options, params);
        }
        else if(lt == BATCHNORM)
        {
            l = parse_batchnorm(options, params);
        }
        else if(lt == MAXPOOL)
        {
            l = parse_maxpool(options, params);
        }
        else if(lt == REORG)
        {
            l = parse_reorg(options, params);
        }
        else if(lt == AVGPOOL)
        {
            l = parse_avgpool(options, params);
        }
        else if(lt == ROUTE)
        {
            l = parse_route(options, params, net);
        }
        else if(lt == UPSAMPLE)
        {
            l = parse_upsample(options, params, net);
        }
        else if(lt == SHORTCUT)
        {
            l = parse_shortcut(options, params, net);
        }
        else if(lt == DROPOUT)
        {
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
        }
        else
        {
            // fprintf(stderr, (char*)"Type not recognized: %s\n", s->type);
        }
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, (char*)"truth", 0);
        l.onlyforward = option_find_int_quiet(options, (char*)"onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, (char*)"stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, (char*)"dontsave", 0);
        l.dontload = option_find_int_quiet(options, (char*)"dontload", 0);
        l.numload = option_find_int_quiet(options, (char*)"numload", 0);
        l.dontloadscales = option_find_int_quiet(options, (char*)"dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, (char*)"learning_rate", 1);
        l.smooth = option_find_float_quiet(options, (char*)"smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
    if(workspace_size)
    {
        //printf("%ld\n", workspace_size);
        net->workspace = (float*)calloc(1, workspace_size);
    }
    return net;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, (char*)"r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0)
    {
        ++ nu;
        strip(line);
        switch(line[0])
        {
        case '[':
            current = (section*)malloc(sizeof(section));
            list_insert(options, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if(!read_option(line, current->options))
            {
                // fprintf(stderr, (char*)"Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
    binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
    int size = l.c*l.size*l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i)
    {
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j)
        {
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k)
            {
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary)
    {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
    // fprintf(stderr, (char*)"Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, (char*)"wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i)
    {
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL)
        {
            save_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED)
        {
            save_connected_weights(l, fp);
        }
        if(l.type == BATCHNORM)
        {
            save_batchnorm_weights(l, fp);
        }
        if(l.type == RNN)
        {
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        }
        if (l.type == LSTM)
        {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        }
        if (l.type == GRU)
        {
            if(1)
            {
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }
            else
            {
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }
        if(l.type == CRNN)
        {
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == LOCAL)
        {
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = (float*)calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x)
    {
        for(y = 0; y < cols; ++y)
        {
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose)
    {
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i)
    {
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j)
        {
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k)
            {
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary)
    {
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
        if(0)
        {
            int i;
            for(i = 0; i < l.n; ++i)
            {
                printf("%g, (char*)", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i)
            {
                printf("%g, (char*)", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0)
        {
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
        if(0)
        {
            int i;
            for(i = 0; i < l.n; ++i)
            {
                printf("%g, (char*)", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i)
            {
                printf("%g, (char*)", l.rolling_variance[i]);
            }
            printf("\n");
        }
    }
    fread(l.weights, sizeof(float), num, fp);
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped)
    {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
}


void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    // fprintf(stderr, (char*)"Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, (char*)"rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000)
    {
        fread(net->seen, sizeof(size_t), 1, fp);
    }
    else
    {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i)
    {
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL)
        {
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED)
        {
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM)
        {
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN)
        {
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN)
        {
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM)
        {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU)
        {
            if(1)
            {
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }
            else
            {
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if(l.type == LOCAL)
        {
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
        }
    }
    // fprintf(stderr, (char*)"Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = (float*)calloc(1, sizeof(float));
    l.biases = (float*)calloc(n*2, sizeof(float));
    l.bias_updates = (float*)calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i)
    {
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;

    // fprintf(stderr, (char*)"detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int classs, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier)
    {
        float pred = 1;
        while(classs >= 0)
        {
            pred *= output[index + stride*classs];
            int g = hier->group[classs];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i)
            {
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*classs] = scale * (1 - output[index + stride*classs]);

            classs = hier->parent[classs];
        }
        *avg_cat += pred;
    }
    else
    {
        if (delta[index] && tag)
        {
            delta[index + stride*classs] = scale * (1 - output[index + stride*classs]);
            return;
        }
        for(n = 0; n < classes; ++n)
        {
            delta[index + stride*n] = scale * (((n == classs)?1 : 0) - output[index + stride*n]);
            if(n == classs) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index_1(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    for (b = 0; b < l.batch; ++b)
    {
        for(n = 0; n < l.n; ++n)
        {
            int index = entry_index_1(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index_1(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            index = entry_index_1(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree)
    {
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i)
        {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    }
    else if (l.softmax)
    {
        int index = entry_index_1(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b)
    {
        if(l.softmax_tree)
        {
            int onlyclass = 0;
            for(t = 0; t < 30; ++t)
            {
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int classs = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000)
                {
                    for(n = 0; n < l.n*l.w*l.h; ++n)
                    {
                        int class_index = entry_index_1(l, b, n, l.coords + 1);
                        int obj_index = entry_index_1(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, classs, l.w*l.h);
                        if(p > maxp)
                        {
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index_1(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index_1(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, classs, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j)
        {
            for (i = 0; i < l.w; ++i)
            {
                for (n = 0; n < l.n; ++n)
                {
                    int box_index = entry_index_1(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    for(t = 0; t < 30; ++t)
                    {
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou)
                        {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index_1(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh)
                    {
                        l.delta[obj_index] = 0;
                    }

                    if(*(net.seen) < 12800)
                    {
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t)
        {
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for(n = 0; n < l.n; ++n)
            {
                int box_index = entry_index_1(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match)
                {
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index_1(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(l.coords > 4)
            {
                int mask_index = entry_index_1(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

            int obj_index = entry_index_1(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore)
            {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background)
            {
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }

            int classs = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) classs = l.map[classs];
            int class_index = entry_index_1(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, classs, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            ++count;
            ++class_count;
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h))
    {
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2)
    {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j)
        {
            for (i = 0; i < l.w/2; ++i)
            {
                for (n = 0; n < l.n; ++n)
                {
                    for(z = 0; z < l.classes + l.coords + 1; ++z)
                    {
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0)
                        {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i)
        {
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n)
        {
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j)
            {
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index_1(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index_1(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index_1(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask)
            {
                for(j = 0; j < l.coords - 4; ++j)
                {
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index_1(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree)
            {

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map)
                {
                    for(j = 0; j < 200; ++j)
                    {
                        int class_index = entry_index_1(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
                else
                {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            }
            else
            {
                if(dets[index].objectness)
                {
                    for(j = 0; j < l.classes; ++j)
                    {
                        int class_index = entry_index_1(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i)
    {
        for(n = 0; n < l.n; ++n)
        {
            int obj_index = entry_index_1(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten;
    if(reverse)
    {
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }
    else
    {
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra)
    {
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    // if(extra){
    // fprintf(stderr, (char*)"reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    // } else {
    // fprintf(stderr, (char*)"reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    // }
    int output_size = l.outputs * batch;
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));

    l.forward = forward_reorg_layer;
    l.backward = backward_reorg_layer;

    return l;
}

void resize_reorg_layer(layer *l, int w, int h)
{
    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse)
    {
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }
    else
    {
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

}

void forward_reorg_layer(const layer l, network net)
{
    int i;
    if(l.flatten)
    {
        memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
        if(l.reverse)
        {
            flatten(l.output, l.w*l.h, l.c, l.batch, 0);
        }
        else
        {
            flatten(l.output, l.w*l.h, l.c, l.batch, 1);
        }
    }
    else if (l.extra)
    {
        for(i = 0; i < l.batch; ++i)
        {
            copy_cpu(l.inputs, net.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
        }
    }
    else if (l.reverse)
    {
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    }
    else
    {
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
    }
}

void backward_reorg_layer(const layer l, network net)
{
    int i;
    if(l.flatten)
    {
        memcpy(net.delta, l.delta, l.outputs*l.batch*sizeof(float));
        if(l.reverse)
        {
            flatten(net.delta, l.w*l.h, l.c, l.batch, 1);
        }
        else
        {
            flatten(net.delta, l.w*l.h, l.c, l.batch, 0);
        }
    }
    else if(l.reverse)
    {
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta);
    }
    else if (l.extra)
    {
        for(i = 0; i < l.batch; ++i)
        {
            copy_cpu(l.inputs, l.delta + i*l.outputs, 1, net.delta + i*l.inputs, 1);
        }
    }
    else
    {
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
    }
}

layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
{
    // fprintf(stderr, (char*)"RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l;
    memset(&l,0,sizeof(layer));
    l.batch = batch;
    l.type = RNN;
    l.steps = steps;
    l.inputs = inputs;

    l.state = (float*)calloc(batch*outputs, sizeof(float));
    l.prev_state = (float*)calloc(batch*outputs, sizeof(float));

    l.input_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.input_layer) = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
    l.input_layer->batch = batch;

    l.self_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.self_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.self_layer->batch = batch;

    l.output_layer = (layer*)malloc(sizeof(layer));
    // fprintf(stderr, (char*)"\t\t");
    *(l.output_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l.output_layer->batch = batch;

    l.outputs = outputs;
    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_rnn_layer;
    l.backward = backward_rnn_layer;
    l.update = update_rnn_layer;

    return l;
}

void update_rnn_layer(layer l, update_args a)
{
    update_connected_layer(*(l.input_layer),  a);
    update_connected_layer(*(l.self_layer),   a);
    update_connected_layer(*(l.output_layer), a);
}

void forward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_layer.delta, 1);
    if(net.train) fill_cpu(l.outputs * l.batch, 0, l.state, 1);

    for (i = 0; i < l.steps; ++i)
    {
        s.input = net.input;
        forward_connected_layer(input_layer, s);

        s.input = l.state;
        forward_connected_layer(self_layer, s);

        float *old_state = l.state;
        if(net.train) l.state += l.outputs*l.batch;
        if(l.shortcut)
        {
            copy_cpu(l.outputs * l.batch, old_state, 1, l.state, 1);
        }
        else
        {
            fill_cpu(l.outputs * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.outputs * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_connected_layer(output_layer, s);

        net.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_rnn_layer(layer l, network net)
{
    network s = net;
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.outputs*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i)
    {
        copy_cpu(l.outputs * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_connected_layer(output_layer, s);

        l.state -= l.outputs*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.outputs * l.batch, input_layer.output - l.outputs*l.batch, 1, l.state, 1);
           axpy_cpu(l.outputs * l.batch, 1, self_layer.output - l.outputs*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.outputs * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.outputs*l.batch;
        if (i == 0) s.delta = 0;
        backward_connected_layer(self_layer, s);

        copy_cpu(l.outputs*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.outputs*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.outputs*l.batch, 1);
        s.input = net.input + i*l.inputs*l.batch;
        if(net.delta) s.delta = net.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_connected_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    // fprintf(stderr,"route ");
    route_layer l;
    memset(&l,0,sizeof(route_layer));
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i)
    {
        // fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    // fprintf(stderr, (char*)"\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  (float*)calloc(outputs*batch, sizeof(float));
    l.output = (float*)calloc(outputs*batch, sizeof(float));;

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i)
    {
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h)
        {
            l->out_c += next.out_c;
        }
        else
        {
            // printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));

}

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i)
    {
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j)
        {
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i)
    {
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j)
        {
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    // fprintf(stderr, (char*)"res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  (float*)calloc(l.outputs*batch, sizeof(float));
    l.output = (float*)calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;

    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));
}


void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    // fprintf(stderr, (char*)"softmax                                        %4d\n",  inputs);
    softmax_layer l;
    memset(&l,0,sizeof(softmax_layer));
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)calloc(inputs*batch, sizeof(float));
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree)
    {
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i)
        {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    }
    else
    {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss)
    {
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

void change_leaves(tree *t, char *leaf_list)
{
    list *llist = get_paths(leaf_list);
    char **leaves = (char **)list_to_array(llist);
    int n = llist->size;
    int i,j;
    int found = 0;
    for(i = 0; i < t->n; ++i)
    {
        t->leaf[i] = 0;
        for(j = 0; j < n; ++j)
        {
            if (0==strcmp(t->name[i], leaves[j]))
            {
                t->leaf[i] = 1;
                ++found;
                break;
            }
        }
    }
    // fprintf(stderr, (char*)"Found %d leaves.\n", found);
}

float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
    float p = 1;
    while(c >= 0)
    {
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for(j = 0; j < n; ++j)
    {
        int parent = hier->parent[j];
        if(parent >= 0)
        {
            predictions[j*stride] *= predictions[parent*stride];
        }
    }
    if(only_leaves)
    {
        for(j = 0; j < n; ++j)
        {
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while(1)
    {
        float max = 0;
        int max_i = 0;

        for(i = 0; i < hier->group_size[group]; ++i)
        {
            int index = i + hier->group_offset[group];
            float val = predictions[(i + hier->group_offset[group])*stride];
            if(val > max)
            {
                max_i = index;
                max = val;
            }
        }
        if(p*max > thresh)
        {
            p = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0) return max_i;
        }
        else if (group == 0)
        {
            return max_i;
        }
        else
        {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

tree *read_tree(char *filename)
{
    tree t = {0};
    FILE *fp = fopen(filename, (char*)"r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while((line=fgetl(fp)) != 0)
    {
        char *id = (char*)calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, (char*)"%s %d", id, &parent);
        t.parent = (int*)realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;

        t.child = (int*)realloc(t.child, (n+1)*sizeof(int));
        t.child[n] = -1;

        t.name = (char**)realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;
        if(parent != last_parent)
        {
            ++groups;
            t.group_offset = (int*)realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = (int*)realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = (int*)realloc(t.group, (n+1)*sizeof(int));
        t.group[n] = groups;
        if (parent >= 0)
        {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = (int*)realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = (int*)realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = (int*)calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = (tree*)calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}

layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if(stride < 0)
    {
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  (float*)calloc(l.outputs*batch, sizeof(float));
    l.output = (float*)calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;

    // if(l.reverse) fprintf(stderr, (char*)"downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    // else fprintf(stderr, (char*)"upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse)
    {
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));

}

void forward_upsample_layer(const layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.reverse)
    {
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }
    else
    {
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

void backward_upsample_layer(const layer l, network net)
{
    if(l.reverse)
    {
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta);
    }
    else
    {
        upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}

double what_time_is_it_now()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

int *read_intlist(char *gpu_list, int *ngpus, int d)
{
    int *gpus = 0;
    if(gpu_list)
    {
        int len = strlen(gpu_list);
        *ngpus = 1;
        int i;
        for(i = 0; i < len; ++i)
        {
            if (gpu_list[i] == ',') ++*ngpus;
        }
        gpus = (int*)calloc(*ngpus, sizeof(int));
        for(i = 0; i < *ngpus; ++i)
        {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    }
    else
    {
        gpus = (int*)calloc(1, sizeof(int));
        *gpus = d;
        *ngpus = 1;
    }
    return gpus;
}

int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, (char*)"r");
    if(!file) file_error(filename);
    while((str=fgetl(file)))
    {
        ++n;
        map = (int*)realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}

void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
    size_t i;
    for(i = 0; i < sections; ++i)
    {
        size_t start = n*i/sections;
        size_t end = n*(i+1)/sections;
        size_t num = end-start;
        shuffle(arr+(start*size), num, size);
    }
}

void shuffle(void *arr, size_t n, size_t size)
{
    size_t i;
    void *swp = calloc(1, size);
    for(i = 0; i < n-1; ++i)
    {
        size_t j = i + rand()/(RAND_MAX / (n-i)+1);
        memcpy(swp,          arr+(j*size), size);
        memcpy(arr+(j*size), arr+(i*size), size);
        memcpy(arr+(i*size), swp,          size);
    }
}

int *random_index_order(int min, int max)
{
    int *inds = (int*)calloc(max-min, sizeof(int));
    int i;
    for(i = min; i < max; ++i)
    {
        inds[i] = i;
    }
    for(i = min; i < max-1; ++i)
    {
        int swap = inds[i];
        int index = i + rand()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i)
    {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg))
        {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i)
    {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg))
        {
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for(i = 0; i < argc-1; ++i)
    {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg))
        {
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i)
    {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg))
        {
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}


char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}

int alphanum_to_int(char c)
{
    return (c < 58) ? c - 48 : c-87;
}
char int_to_alphanum(int i)
{
    if (i == 36) return '.';
    return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i)
    {
        printf("%d ", i+1);
        for(j = 0; j < N; ++j)
        {
            printf("%2.4f, (char*)", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, (char*)"%s", str);
    if(!(p = strstr(buffer, orig)))   // Is 'orig' even in 'str'?
    {
        sprintf(output, (char*)"%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, (char*)"%s%s%s", buffer, rep, p+strlen(orig));
}

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i)
    {
        int curr = i;
        for(j = 0; j < k; ++j)
        {
            if((index[j] < 0) || a[curr] > a[index[j]])
            {
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

unsigned char *read_file(char *filename)
{
    FILE *fp = fopen(filename, (char*)"rb");
    size_t size;

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *text = (unsigned char *)calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);
    return text;
}

void malloc_error()
{
    // fprintf(stderr, (char*)"Malloc error\n");
    exit(-1);
}

void file_error(char *s)
{
    // fprintf(stderr, (char*)"Couldn't open file: %s\n", s);
    exit(0);
}

list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i)
    {
        if(s[i] == delim)
        {
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i)
    {
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i)
    {
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = (char*)malloc(size*sizeof(char));
    if(!fgets(line, size, fp))
    {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp))
    {
        if(curr == size-1)
        {
            size *= 2;
            line = (char*)realloc(line, size*sizeof(char));
            if(!line)
            {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

int read_int(int fd)
{
    int n = 0;
    int next = read(fd, &n, sizeof(int));
    if(next <= 0) return -1;
    return n;
}

void write_int(int fd, int n)
{
    int next = write(fd, &n, sizeof(int));
    if(next <= 0) error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes)
    {
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes)
    {
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes)
    {
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) error("read failed");
        n += next;
    }
}

void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes)
    {
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) error("write failed");
        n += next;
    }
}


char *copy_string(char *s)
{
    char *copy = (char*)malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}

list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c)
    {
        if(*c == '"') in = !in;
        else if(*c == ',' && !in)
        {
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}

int count_fields(char *line)
{
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c)
    {
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}

float *parse_fields(char *line, int n)
{
    float *field = (float*)calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c)
    {
        done = (*c == '\0');
        if(*c == ',' || done)
        {
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
            p = c+1;
            ++count;
        }
    }
    return field;
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
    int i;
    int j;
    memset(avg, 0, els*sizeof(float));
    for(j = 0; j < n; ++j)
    {
        for(i = 0; i < els; ++i)
        {
            avg[i] += a[j][i];
        }
    }
    for(i = 0; i < els; ++i)
    {
        avg[i] /= n;
    }
}

void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}

float mse_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i]*a[i];
    return sqrt(sum/n);
}

void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i)
    {
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        a[i] += s;
    }
}

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i)
    {
        sum += a[i]*a[i];
    }
    return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        a[i] *= s;
    }
}

int sample_array(float *a, int n)
{
    float sum = sum_array(a, n);
    scale_array(a, n, 1./sum);
    float r = rand_uniform(0, 1);
    int i;
    for(i = 0; i < n; ++i)
    {
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}

int max_int_index(int *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    int max = a[0];
    for(i = 1; i < n; ++i)
    {
        if(a[i] > max)
        {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i)
    {
        if(a[i] > max)
        {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

int int_index(int *a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        if(a[i] == val) return i;
    }
    return -1;
}

int rand_int(int min, int max)
{
    if (max < min)
    {
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand()%(max - min + 1)) + min;
    return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)rand()/RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
    return  ((size_t)(rand()&0xff) << 56) |
            ((size_t)(rand()&0xff) << 48) |
            ((size_t)(rand()&0xff) << 40) |
            ((size_t)(rand()&0xff) << 32) |
            ((size_t)(rand()&0xff) << 24) |
            ((size_t)(rand()&0xff) << 16) |
            ((size_t)(rand()&0xff) << 8) |
            ((size_t)(rand()&0xff) << 0);
}

float rand_uniform(float min, float max)
{
    if(max < min)
    {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if(rand()%2) return scale;
    return 1./scale;
}

float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = (float**)calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i)
    {
        t[i] = (float*)calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)calloc(1, sizeof(float));
    l.biases = (float*)calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else
    {
        l.mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i)
        {
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i)
    {
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;

    // fprintf(stderr, (char*)"yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int classs, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index])
    {
        delta[index + stride*classs] = 1 - output[index + stride*classs];
        if(avg_cat) *avg_cat += output[index + stride*classs];
        return;
    }
    for(n = 0; n < classes; ++n)
    {
        delta[index + stride*n] = ((n == classs)?1 : 0) - output[index + stride*n];
        if(n == classs && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

    for (b = 0; b < l.batch; ++b)
    {
        for(n = 0; n < l.n; ++n)
        {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b)
    {
        for (j = 0; j < l.h; ++j)
        {
            for (i = 0; i < l.w; ++i)
            {
                for (n = 0; n < l.n; ++n)
                {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t)
                    {
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou)
                        {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh)
                    {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh)
                    {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int classs = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) classs = l.map[classs];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, classs, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t)
        {
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n)
            {
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0)
            {
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int classs = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) classs = l.map[classs];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, classs, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h))
    {
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i)
    {
        for(n = 0; n < l.n; ++n)
        {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh)
            {
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j)
    {
        for (i = 0; i < l.w/2; ++i)
        {
            for (n = 0; n < l.n; ++n)
            {
                for(z = 0; z < l.classes + 4 + 1; ++z)
                {
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0)
                    {
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i)
    {
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n)
        {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j)
            {
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

using namespace std;

const short AI = 1; // AI的棋子
const short OP = -1; // 对手的棋子
const short BLK = 0; // 空白

class AlphaPig
{
public:
    short board[81] = { 0 };
    bool air_vis[81];
    bool ai_first = false;
    int round = 0;
    int mcts_cnt = 0, pl_cnt = 0;
    network* net;

    // 蒙特卡洛树节点
    struct treeNode
    {
        // 棋盘及颜色
        short board[81] = { 0 };
        short color;

        // 下一步可行位置
        short available_me[81], available_his[81];
        int available_me_size = 0, available_his_size = 0;
        bool available_me_map[81] = { false }, available_his_map[81] = { false };

        // 节点输赢统计
        double value = 0;
        int total = 0;
        int win = 0;
        bool fail = false;

        // 父节点
        treeNode* father = NULL;

        // 孩子节点
        treeNode* children[81];
        int children_size = 0;

        // 孩子探索策略
        short policy[81];
        int policy_size = 0;

        // 分支遍历完成数量
        int complete = 0;

        // 最后一手位置
        short last_p = -1;

        // 深度
        int depth = 0;
    };

    treeNode* root = NULL;

    // 策略权重
    struct Weight
    {
        short p;
        float w;
    };

    // 权重排序
    static bool weightCMP(const Weight& a, const  Weight& b)
    {
        return a.w > b.w;
    }

    // 移动位置
    inline short moveTo(short p, short dir)
    {
        switch (dir)
        {
        case 0:
            return (p += 9) < 81 ? p : -1;
        case 1:
            return (p -= 9) >= 0 ? p : -1;
        case 2:
            return p % 9 < 8 ? p + 1 : -1;
        case 3:
            return p % 9 > 0 ? p - 1 : -1;
        }
        return p;
    }

    // 判断是否有气
    bool hasAir(short mBoard[], short p)
    {
        air_vis[p] = true;
        bool flag = false;
        for (short dir = 0; dir < 4; dir++)
        {
            short dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (mBoard[dp] == BLK)
                    flag = true;
                if (mBoard[dp] == mBoard[p] && !air_vis[dp])
                    if (hasAir(mBoard, dp))
                        flag = true;
            }
        }
        return flag;
    }

    // 判断是否可以下子
    bool judgeAvailable(short mBoard[], short p, short col)
    {
        if (mBoard[p]) return false;
        mBoard[p] = col;
        memset(air_vis, 0, sizeof(air_vis));
        if (!hasAir(mBoard, p))
        {
            mBoard[p] = 0;
            return false;
        }
        for (short dir = 0; dir < 4; dir++)
        {
            short dp = moveTo(p, dir);
            if (dp >= 0)
            {
                if (mBoard[dp] && !air_vis[dp])
                    if (!hasAir(mBoard, dp))
                    {
                        mBoard[p] = 0;
                        return false;
                    }
            }
        }
        mBoard[p] = 0;
        return true;
    }

    // 扫描可以下子的位置
    void scanAvailable(treeNode* node)
    {
        short* board = node->board;
        bool ban_his[81] = { false }, ban_me[81] = { false }; // 禁下
        bool vis[81] = { false };

        for (short dir = 0; dir < 4; dir++)
        {
            short p = moveTo(node->last_p, dir);
            if(p == -1) continue;
            if (board[p] == BLK)
            {
                ban_me[p] = !judgeAvailable(board, p, node->color);
                ban_his[p] = !judgeAvailable(board, p, -node->color);
            }
            else if (!vis[p])
            {
                short queue[81], q_left = 0, q_right = 0;
                bool tgas_vis[81] = { false };
                short tgas = 0;
                int tgas_size = 0;
                queue[q_right++] = p;
                while (q_left != q_right)
                {
                    short pq = queue[q_left++];
                    q_left %= 81;
                    vis[pq] = true;
                    for (short dir = 0; dir < 4; dir++)
                    {
                        short dp = moveTo(pq, dir);
                        if (dp >= 0)
                        {
                            if (board[dp] == BLK && !tgas_vis[dp])
                            {
                                tgas_vis[dp] = true;
                                tgas_size++;
                                tgas = dp;
                            }
                            else if (board[dp] == board[pq] && !vis[dp])
                            {
                                queue[q_right++] = dp;
                                q_right %= 81;
                            }
                        }
                    }
                }
                if (tgas_size == 1)
                {
                    ban_me[tgas] = !judgeAvailable(board, tgas, node->color);
                    ban_his[tgas] = !judgeAvailable(board, tgas, -node->color);
                }
            }
        }

        for (int i = 0; i < node->father->available_me_size; i++)
        {
            short p = node->father->available_me[i];
            if (board[p] == BLK && !ban_his[p])
            {
                node->available_his[(node->available_his_size)++] = p;
                node->available_his_map[p] = true;
            }
        }

        for (int i = 0; i < node->father->available_his_size; i++)
        {
            short p = node->father->available_his[i];
            if (board[p] == BLK && !ban_me[p])
            {
                node->available_me[(node->available_me_size)++] = p;
                node->available_me_map[p] = true;
            }
        }
    }

    // 策略函数
    void makePolicy(treeNode* node)
    {
        // 优先不走眼
        short eye[81] = { 0 }, no_eye[81] = { 0 };
        int eye_size = 0, no_eye_size = 0;
        short col = -node->color;

        for (int i = 0; i < node->available_his_size; i++)
        {
            short p = node->available_his[i];
            bool is_eye = true;
            for (short dir = 0; dir < 4; dir++)
            {
                short dp = moveTo(p, dir);
                if (dp >= 0 && node->board[dp] != col)
                {
                    is_eye = false;
                    break;
                }
            }
            if (is_eye)
            {
                eye[eye_size++] = p;
            }
            else
            {
                no_eye[no_eye_size++] = p;
            }
        }

        // 只剩下眼，直接返回
        if (no_eye_size == 0)
        {
            memcpy(node->policy, eye, sizeof(node->policy));
            node->policy_size = eye_size;
            return;
        }

        // 直接打乱后返回
        if (no_eye_size <= 15 || node->depth > 2)
        {
            memcpy(node->policy, no_eye, sizeof(node->policy));
            node->policy_size = no_eye_size;
            // 打乱
            for (int i = node->policy_size - 1; i >= 0; i--)
                swap(node->policy[i], node->policy[rand() % (i + 1)]);
            return;
        }

        // 标记出哪些地方可以走
        bool available[81] = { false };
        for (int i = 0; i < no_eye_size; i++)
            available[no_eye[i]] = true;

        // 策略网络
        float input[4 * 81] = { 0 };
        for (int i = 0; i < 81; i++)
        {
            if (node->board[i] == col)
                input[0 * 81 + i] = 1;
            else if (node->board[i] == -col)
                input[1 * 81 + i] = 1;
            else if (node->board[i] == BLK)
                input[2 * 81 + i] = 1;
        }
        if (node->last_p >= 0)
            input[3 * 81 + node->last_p] = 1;
        float* output = network_predict(net, input);

        // 按照权值排序
        Weight weight[81];
        for (int i = 0; i < 81; i++)
        {
            weight[i].p = i;
            weight[i].w = output[i];
        }
        std::sort(weight, weight + 81, &weightCMP);

        // 挑选出权值高的点
        double minw;
        for (int i = 0; i < 81; i++)
        {
            if (available[weight[i].p])
            {
                minw = weight[i].w * 0.05;
                break;
            }
        }

        bool selected[81] = { false };
        for (int i = 0; i < 81 && (minw < weight[i].w || node->policy_size < 2) && node->policy_size < 8; i++)
        {
            if (available[weight[i].p])
            {
                node->policy[node->policy_size++] = weight[i].p;
                selected[weight[i].p] = true;
            }
        }

        // 单独扫描可能被策略网络漏掉的眼
        eye_size == 0;
        for (short p = 0; p < 81; p++)
        {
            if (node->board[p])
                continue;
            bool weye = true;
            short col = 0, air = 0, air_p = 0;
            for (short dir = 0; dir < 4; dir++)
            {
                short dp = moveTo(p, dir);
                if (dp == -1) continue;
                if (node->board[dp])
                {
                    if (col)
                    {
                        if (node->board[dp] != col)
                        {
                            weye = false;
                            break;
                        }
                    }
                    else
                    {
                        col = node->board[dp];
                    }
                }
                else
                {
                    if (air++ > 1)
                        break;
                    air_p = dp;
                }
            }
            if (weye)
            {
                if (air == 1)
                {
                    if (col == node->color)
                    {
                        if (node->available_me_map[air_p])
                        {
                            if (!selected[p] && node->available_his_map[p])
                            {
                                node->policy[node->policy_size++] = p;
                                selected[p] = true;
                            }

                            if (!selected[air_p] && node->available_his_map[air_p])
                            {
                                node->policy[node->policy_size++] = air_p;
                                selected[air_p] = true;
                            }
                        }
                    }
                    else
                    {
                        if (!selected[air_p] && node->available_his_map[air_p])
                        {
                            node->policy[node->policy_size++] = air_p;
                            selected[air_p] = true;
                        }
                    }
                }
            }
        }

        for (int i = node->policy_size - 1; i >= 0; i--)
            swap(node->policy[i], node->policy[rand() % (i + 1)]);
    }

    // 估值函数
    inline double calcValue(treeNode* node)
    {
        // 暂时用可行下子位置估值
        double a = node->available_me_size;
        double b = node->available_his_size;
        if (a == 0 && b == 0 && node->father != NULL)
        {
            return -calcValue(node->father);
        }
        return 1 / (1 + pow(2.7182818284590452354, b - a)) * 2 - 1;
    }

    // 新建节点
    inline treeNode* newNode(treeNode* father, short p)
    {
        treeNode* newNode = new treeNode();
        memcpy(newNode->board, father->board, sizeof(board));
        newNode->color = -father->color;
        newNode->last_p = p;
        newNode->board[p] = newNode->color;
        newNode->father = father;
        newNode->depth = father->depth + 1;
        scanAvailable(newNode);
        makePolicy(newNode);
        father->children[father->children_size++] = newNode;
        return newNode;
    }

    // 删除分支
    void deleteTree(treeNode* node)
    {
        if (node != NULL)
        {
            while (node->children_size > 0)
                deleteTree(node->children[--node->children_size]);
            delete node;
        }
    }

    // 节点搜索完成
    inline bool finishNode(treeNode* node)
    {
        return (node->available_his_size > 0 && node->policy_size == 0 && node->complete == node->children_size) || (node->available_his_size == 0 && node->complete > 0);
    }

    // 选择最优子节点
    treeNode* bestChild(treeNode* node)
    {
        treeNode* max_node = NULL;
        bool Allcomplete = true;
        double max = -1e10;
        for (int i = 0; i < node->children_size; i++)
        {
            treeNode* t_node = node->children[i];
            if (finishNode(t_node))
                continue;

            // 上限置信区间算法
            double probability = t_node->value / t_node->total + 1.4142135623731 * sqrt(log(t_node->father->total) / t_node->total);
            if (probability > max)
            {
                max = probability;
                max_node = t_node;
                Allcomplete = false;
            }
        }
        return Allcomplete ? NULL : max_node;
    }

    // 选择&模拟&回溯
    bool select(treeNode* node)
    {
        // 选择
        while (node->available_his_size > 0) // 这个节点的游戏没有结束
        {
            if (node->policy_size > 0) // 这个节点有可行动作还未被拓展过
            {
                // 拓展
                node = newNode(node, node->policy[--node->policy_size]);
                break;
            }
            else   // 这个节点所有可行动作都被拓展过
            {
                node = bestChild(node);
                if (node == NULL)
                    return false;
            }
        }
        double value;

        // 模拟
        if (node->available_his_size == 0)   // 是否结束
        {
            node->complete = 1;
            treeNode* father = node->father;
            father->complete++;
            father->fail = true;
            while (father != NULL)
            {
                if (father->father == NULL)
                    break;
                if (finishNode(father))
                {
                    father->father->complete++;
                    if (father->fail == true)
                        father->father->win++;
                    if (father->win == father->complete)
                        father->father->fail = true;
                }
                father = father->father;
            }
            value = 1;
        }
        else
        {
            value = calcValue(node);
        }

        // 回溯
        while (node != NULL)
        {
            node->total += 1;
            node->value += value;
            node = node->father;
            value = -value;
        }

        return true;
    }

    // 初始化树
    void initRoot(short last_p)
    {
        root = new treeNode();
        memcpy(root, board, sizeof(board));
        root->color = OP;
        root->last_p = last_p;
        for (int i = 0; i < 81; i++)
        {
            if (judgeAvailable(root->board, i, root->color))
            {
                root->available_me[(root->available_me_size)++] = i;
                root->available_me_map[i] = true;
            }

            if (judgeAvailable(root->board, i, -root->color))
            {
                root->available_his[(root->available_his_size)++] = i;
                root->available_his_map[i] = true;
            }

        }
        makePolicy(root);
    }

    AlphaPig()
    {
        srand((unsigned)time(NULL));
        char* cfgfile = (char*)"data/nogo/policy_network.cfg";
        char* weightfile = (char*)"data/nogo/policy_network.weights";
        net = load_network(cfgfile, weightfile, 0);
    }

    ~AlphaPig()
    {
        free_network(net);
    }

    short choose(short last_p)
    {
        initRoot(last_p);

        // 搜索0.9秒
        double endClock = 0.9 * CLOCKS_PER_SEC + clock();
        while (clock() < endClock && select(root));
        mcts_cnt = root->total;
        pl_cnt = root->children_size;

        // 选择最好的下法
        short ai = 0;
        treeNode* max_node = root->children[0];
        bool fail = finishNode(root) && root->win == root->complete; // 必败
        double max = -1e10;
        for (int i = 0; i < root->children_size; i++)
        {
            treeNode* t_node = root->children[i];
            if (finishNode(t_node) && t_node->win == t_node->complete)
            {
                // 必胜节点
                max_node = t_node;
                break;
            }
            if (!fail && t_node->fail)
            {
                // 假设没有必败，就跳过必败节点
                continue;
            }
            double probability = t_node->value / t_node->total;
            if (probability > max)
            {
                max = probability;
                max_node = t_node;
            }
        }

        ai = max_node->last_p;

        if(round <= 6)
        {
            // 白子占内圈
            /*
            if (!ai_first)
            {
                if (last_p == 0 && board[1] == BLK && board[9] == BLK && root->available_his_map[10])
                    ai = 10;
                if (last_p == 8 && board[7] == BLK && board[17] == BLK && root->available_his_map[16])
                    ai = 16;
                if (last_p == 72 && board[73] == BLK && board[63] == BLK && root->available_his_map[64])
                    ai = 64;
                if (last_p == 80 && board[79] == BLK && board[71] == BLK && root->available_his_map[70])
                    ai = 70;
            }
            */

            // 反制占内圈
            int n = 0;
            n += (board[0] == AI && board[10] == OP);
            n += (board[8] == AI && board[16] == OP);
            n += (board[72] == AI && board[64] == OP);
            n += (board[80] == AI && board[70] == OP);

            if(n)
            {
                if(ai == 0 && root->available_his_map[10])
                    ai = 10;
                if(ai == 8 && root->available_his_map[16])
                    ai = 16;
                if(ai == 72 && root->available_his_map[64])
                    ai = 64;
                if(ai == 80 && root->available_his_map[70])
                    ai = 70;
            }
        }

        deleteTree(root);

        return ai;
    }
};

int main()
{
    AlphaPig alphaPig;
    Json::Reader reader;
    Json::Value input;
    Json::Value ret;
    Json::Value debug;
    Json::FastWriter writer;
    string str;

    bool fst_turn = true;
    while (cin >> str)
    {
        reader.parse(str, input);

        int x = 0, y = 0;
        if (fst_turn)
        {
            x = input["requests"][(Json::Value::UInt) 0]["x"].asInt();
            y = input["requests"][(Json::Value::UInt) 0]["y"].asInt();
            alphaPig.ai_first = x == -1;
        }
        else
        {
            x = input["x"].asInt();
            y = input["y"].asInt();
        }

        int rp = x * 9 + y;
        if (x != -1) alphaPig.board[rp] = OP;
        alphaPig.round++;
        int p = alphaPig.choose(rp);
        alphaPig.board[p] = AI;

        debug["mcts_cnt"] = alphaPig.mcts_cnt;
        debug["pl_cnt"] = alphaPig.pl_cnt;

        ret["response"]["x"] = p / 9;
        ret["response"]["y"] = p % 9;
        ret["response"]["debug"] = debug;

        cout << writer.write(ret) << endl;
        cout << ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << endl;
        cout << flush;
        fst_turn = false;
    }

    return 0;
}