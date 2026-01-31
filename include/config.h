#ifndef CONFIG_H
#define CONFIG_H
#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <argp.h>
#include <pthread.h>

#include "pvt.h"
#include "sim.h"
#include "path.h"
#include "gis.h"
#include "array.h"

extern struct config config;
enum {
    DP_MODE_NONE = 0,      // No direct path
    DP_MODE_FIXED = 1,     // Fixed amplitude direct path
    DP_MODE_STEERED = 2    // Direct path with steering vector consideration
};

struct config {
    struct GPSPOS   tx_location;
    char*           tx_source_file;

    struct GPSPOS   rx_location;
    struct array*   rx_array;
    
    // New: Incident angle of TX relative to RX (global constant)
    double          tx_azimuth;      // [rad] Azimuth angle of TX as seen from RX
    double          tx_elevation;    // [rad] Elevation angle of TX as seen from RX
    cf32*           tx_steering;     // Steering vector of RX array towards TX direction
    
    double          center_freq;
    double          sample_rate;
    int             sample_count;
    int             interactive;
    int             verbosity;
    int             direct_path_mode;
    char*           dump_trackpoints;

    double          max_distance;
    double          min_altitude;
    int             max_slowdelay;

    char*           output_file;
    char*           output_timestamp;
    char*           truth_file;   // Path for ground truth CSV output (optional)

    time_t          start_time;
    time_t          end_time;
    time_t          step;

    struct PVTDB**  targets;
    int             target_len;

    time_t          min_time;
    time_t          max_time;

    double          main_axel;
};

void init_config();
void prepare_config(struct config* config);
void print_config(struct config* config);

extern struct argp argp;
error_t parse_opt(int key, char *arg, struct argp_state *state);

#endif

