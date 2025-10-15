#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double rad2deg(double radians) {
    return radians / (M_PI / 180.0);
}

double calc_angle_to_rx(double heading, double azi, double elev) {
    double azi_deg = rad2deg(azi);

    char* azi_override = getenv("OVERRIDE_AZI");
    if(azi_override) {
        azi_deg = atof(azi_override);
        printf("OVERRIDE AZI TO %.2f\n", atof(azi_override));
    }

    double theta_deg = azi_deg - heading;

    //printf("\tazi_deg: %.1f theta_deg: %.1f\n", azi_deg, theta_deg);

    if(theta_deg >= 3.0 * 90.0) {
        theta_deg -= 360;
    } else if(theta_deg > 90) {
        theta_deg -= 180;
        printf("target behind array\n");
    }

    return theta_deg;
}