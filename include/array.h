#ifndef ARRAY_H
#define ARRAY_H

#include <complex.h>
typedef complex float cf32;

#ifdef __GNUC__
#define DLL_EXPORT __attribute__((visibility("default")))
#else
#define DLL_EXPORT
#endif

enum array_geom {
    ULA,
    UCA,
    OCA   // Other / Custom Array
};

struct array;

typedef cf32* (*array_steering_fn)(struct array* self, double azi, double elev);
typedef int   (*array_blindspot_fn)(struct array* self, double azi);
typedef void  (*array_free_fn)(struct array* self);

struct array {
    char id[32];
    
    int channel_count;
    int observe_count;

    double heading;
    double ant_geom;
    double frontend_gain;

    enum array_geom geom;

    array_steering_fn   steering;
    array_blindspot_fn  blindspot;
    array_free_fn       destroy;

    void* priv;
};

DLL_EXPORT struct array* create_array(void);

#endif
