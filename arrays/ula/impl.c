#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "array.h"

double deg2rad(double degrees);
double rad2deg(double radians);
double calc_angle_to_rx(double heading, double azi, double elev);

static cf32* array_steering_impl(struct array* self, double theta_deg, double elev) {
    cf32* a = calloc(self->channel_count, sizeof(cf32));
    if (!a) return NULL;

    float dphi = 2.0 * M_PI * self->ant_geom * sin(deg2rad(theta_deg));
    for(int i=0; i < self->channel_count; i++) {
        a[i] = cexp((float)(i) * dphi * I);
    }

    return a;
}

static int array_blindspot_impl(struct array* self, double azi) {
    (void)self;

    double theta_deg = calc_angle_to_rx(self->heading, azi, 0.0);

    if(theta_deg > 80 && theta_deg < 280) {
        return 1;
    }

    return 0;
}

static void array_free_impl(struct array* self) {
    free(self);
}

DLL_EXPORT struct array* create_array_model(void) {
    struct array* a = calloc(1, sizeof(*a));
    if (!a) return NULL;

     strncpy(a->id, "SIMPLE ULA", 32);

    a->channel_count  = 8;
    a->observe_count  = 7;
    a->heading        = 0.0;
    a->ant_geom       = 0.33;
    a->frontend_gain  = 20.0;
    a->geom           = ULA;

    a->steering  = array_steering_impl;
    a->blindspot = array_blindspot_impl;
    a->destroy   = array_free_impl;

    return a;
}
