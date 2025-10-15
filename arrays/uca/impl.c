#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "array.h"

double deg2rad(double degrees);
double rad2deg(double radians);
double calc_angle_to_rx(double heading, double azi, double elev);


static cf32 array_element_char_for_dir(double azi) {
    if(azi < 0) azi += 360;
    if(azi < 0) azi += 360;

    //printf("azi: %.2f\n", azi);

    if(azi > 300 || azi < 60) {
        return 1.0;
    } else if(azi > 120 && azi < 240) {
        return 1.0;
    } else {
        return 0.0;
    }
}

static cf32* array_steering_impl(struct array* self, double theta_deg, double elev) {
    cf32* a = calloc(self->channel_count, sizeof(cf32));
    if (!a) return NULL;

    double alpha = 2.0*M_PI / (self->channel_count-1);
    double theta = deg2rad(theta_deg);

    cf32* ant = calloc(self->channel_count, sizeof(cf32));

    ant[0] = 0 + 0*I; // reference channel in center
    for(int ch = 1 ; ch<self->channel_count; ch++)
        ant[ch] = self->ant_geom * cexp((ch-1) * alpha * I); // other elements on circle

    for(int ch = 0; ch<self->channel_count; ch++) {
        a[ch] = cexp( I * 2.0 * M_PI * ( creal(ant[ch]) * cos(theta) + cimag(ant[ch]) * sin(theta) ) );

        double rel_azi = theta_deg - rad2deg(alpha) * ch;
        if(rel_azi < 0) rel_azi += 360;
        if(rel_azi < 0) rel_azi += 360;
        //printf("direction = %.2f deg, relative to channel %d -> %.2f\n", theta_deg, ch, rel_azi);
        a[ch] *= array_element_char_for_dir(rel_azi);
    }

    free(ant);

    return a;
}


static int array_blindspot_impl(struct array* self, double azi) {
    (void)self;

    return 0;
}

static void array_free_impl(struct array* self) {
    free(self);
}

DLL_EXPORT struct array* create_array_model(void) {
    struct array* a = calloc(1, sizeof(*a));
    if (!a) return NULL;

    strncpy(a->id, "SIMPLE UCA", 32);

    a->channel_count  = 8;
    a->observe_count  = 7;
    a->heading        = 0.0;
    a->ant_geom       = 0.33;
    a->frontend_gain  = 20.0;
    a->geom           = UCA;

    a->steering  = array_steering_impl;
    a->blindspot = array_blindspot_impl;
    a->destroy   = array_free_impl;

    return a;
}
