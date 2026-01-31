#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "array.h"

double deg2rad(double degrees);
double rad2deg(double radians);
double calc_angle_to_rx(double heading, double azi, double elev);

/**
 * Steering vector calculation (elevation angle ignored)
 *
 * Parameter conventions:
 *  - theta_enu_rad : Azimuth angle (in array plane), unit: radians
 *  - elev          : Not used in current version, interface reserved
 */
static cf32* array_steering_impl(struct array* self, double theta_enu_rad, double elev)
{
    (void)elev;  // Ignore elevation angle

    if (!self || self->channel_count <= 0)
        return NULL;

    cf32* a = calloc(self->channel_count, sizeof(cf32));
    if (!a) return NULL;

    if (self->channel_count < 2) {
        a[0] = 1.0f + 0.0f * I;
        return a;
    }

    // 🔑 Key: ENU angle (0rad=North) → Mathematical angle (0rad=East)
    // 90 degrees converted to radians: M_PI/2
    double theta_math_rad = M_PI / 2.0 - theta_enu_rad;
    
    // Normalize to [0, 2π) range
    while (theta_math_rad < 0.0) 
        theta_math_rad += 2 * M_PI;
    while (theta_math_rad >= 2 * M_PI) 
        theta_math_rad -= 2 * M_PI;

    // No need to convert angle to radians anymore, use converted radian value directly
    double theta_rad = theta_math_rad;

    double alpha = 2.0 * M_PI / (self->channel_count - 1);
    cf32* ant = calloc(self->channel_count, sizeof(cf32));
    if (!ant) {
        free(a);
        return NULL;
    }

    ant[0] = 0.0f + 0.0f * I;  // Central reference channel
    for (int ch = 1; ch < self->channel_count; ch++) {
        ant[ch] = self->ant_geom * cexpf((ch - 1) * alpha * I);  // ch=1 at East (0 rad)
    }

    for (int ch = 0; ch < self->channel_count; ch++) {
        double x = crealf(ant[ch]);
        double y = cimagf(ant[ch]);
        double phase = 2.0 * M_PI * (x * cos(theta_rad) + y * sin(theta_rad));
        a[ch] = cexpf(I * phase);  // Omnidirectional antenna
    }

    free(ant);
    return a;
}

static int array_blindspot_impl(struct array* self, double azi)
{
    (void)self;
    (void)azi;
    return 0;
}

static void array_free_impl(struct array* self)
{
    free(self);
}

DLL_EXPORT struct array* create_array_model(void)
{
    struct array* a = calloc(1, sizeof(*a));
    if (!a) return NULL;

    strncpy(a->id, "SIMPLE UCA", 32);

    a->channel_count  = 9;      // 1 ref + 8 surv
    a->observe_count  = 8;
    a->heading        = 0.0;
    a->ant_geom       = 0.33;   // Unit: λ (wavelength)
    a->frontend_gain  = 20.0;
    a->geom           = UCA;

    a->steering  = array_steering_impl;
    a->blindspot = array_blindspot_impl;
    a->destroy   = array_free_impl;

    return a;
}

