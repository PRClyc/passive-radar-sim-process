#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "gis.h"
#include "threadpool.h"

struct config config;

int main(int argc, char *argv[]) {
    init_config(argc, argv);

    prepare_config(&config);

    print_config(&config);

    if(config.rx_array == NULL) {
        fprintf(stderr, "ERR: cannot use antenna definition library\n");
        exit(-1);
    }

    simulate();

    return 0;
}


void simulate(void) {
    printf("SIMULATION STARTING NOW... (%.2fm main)\n", config.main_axel);

    int target_cnt = 0, valid_cnt = 0;
    char _tick[20];

    // ===== Array output file =====
    FILE* output = NULL;
    if (config.output_file != NULL) {
        output = fopen(config.output_file, "wb");
        if (output == NULL) {
            printf("ERR: cannot open output file\n");
            exit(2);
        }
        printf("opened output path for writing '%s'\n", config.output_file);
    }

    // ===== Timestamp output file =====
    FILE* output_timestamp = NULL;
    if (config.output_timestamp != NULL) {
        output_timestamp = fopen(config.output_timestamp, "w");
        if (output_timestamp == NULL) {
            printf("ERR: cannot open output timestamp file\n");
            exit(2);
        }
        printf("opened timestamp path for writing '%s'\n", config.output_timestamp);
    }

    // ===== Reference signal output file (New: Fixed write to ref.cf32) =====
    FILE* output_ref = fopen("ref.cf32", "wb");
    if (!output_ref) {
        printf("ERR: cannot open ref.cf32 for writing\n");
        exit(2);
    }
    printf("opened reference path 'ref.cf32'\n");

    // ===== New: Ground truth CSV output file (Optional: Specified by -q/--truth-data) =====
    FILE* output_truth = NULL;
    if (config.truth_file != NULL) {
        output_truth = fopen(config.truth_file, "w");
        if (!output_truth) {
            printf("ERR: cannot open truth csv '%s'\n", config.truth_file);
            exit(2);
        }
        printf("opened truth csv for writing '%s'\n", config.truth_file);

        fprintf(output_truth,
            "tick_unix,target_idx,"
            "tx_azimuth_deg,tx_elevation_deg,"
            "target_lat,target_lon,target_alt,"
            "target_speed_mps,target_heading_deg,"
            "tx_dist_m,rx_dist_m,route_diff_m,bistatic_speed_mps,"
            "delay_samp,doppler_hz,azimuth_deg,elevation_deg,attenuation_db\n"
        );
        fflush(output_truth);
    }

    // paths[i] corresponds one-to-one with targets[i]
    struct PVT** targets = calloc(config.target_len, sizeof(struct PVT*));
    struct path* paths   = calloc(config.target_len, sizeof(struct path));

    init_rf();

    tpool_t* workers = tpool_create(4);

    time_t tick = config.start_time;
    do {
        clock_t begin = clock();
        (void)begin; // Avoid unused variable warning if begin is not used

        strftime(_tick, 20, "%Y-%m-%d %H:%M:%S", localtime(&tick));

        target_cnt = get_pvts_at(tick, config.targets, targets, config.target_len);
        valid_cnt  = filter_pvts(targets, config.target_len,
                                 config.max_distance,
                                 config.min_altitude,
                                 10.0);

        printf("[%s] %d in air, %d valid\n", _tick, target_cnt, valid_cnt);

        // Use i directly as index
        for (int i = 0; i < config.target_len; i++) {

            if (targets[i] == NULL) continue;
            struct PVT* t = targets[i];

            paths[i].target = t;
            paths[i].rcs    = config.targets[i]->rcs;

            if (solve_path(&paths[i])) {
                // Mark as invalid on failure and skip
                targets[i] = NULL;
                valid_cnt--;
                continue;
            }

            if (config.targets[i]->dump_trackpoints != NULL) {
                fprintf(config.targets[i]->dump_trackpoints,
                        "\t{\"timestamp\": \"%s\", \"lat\": %.6f, \"lon\": %.7f}\n",
                        _tick, t->pos.lat, t->pos.lon);
                fflush(config.targets[i]->dump_trackpoints);
            }

            // Output path information
            printf("\t#%d -> %f,%f,%7.2f %5.1fm/s %5.1fdeg "
                   "[%8.2f+%8.2f, %9.1f ds, %6.1f m/s, %4d dt, %7.1fHz df, "
                   "az %5.1f, el %2.1f, %5.2fdB] \n",
                   i,
                   t->pos.lat, t->pos.lon, t->pos.alt,
                   t->speed, t->heading,
                   paths[i].tx_dist,
                   paths[i].rx_dist,
                   paths[i].route_diff,
                   paths[i].bistatic_speed,
                   paths[i].delay,
                   paths[i].doppler,
                   rad2deg(paths[i].azimuth),
                   rad2deg(paths[i].elevation),
                   10 * log10(paths[i].attenuation));

            // ===== New: Write to truth csv (one line per target per frame) =====
            if (output_truth != NULL) {
                fprintf(output_truth,
                    "%ld,%d,"
                    "%.6f,%.6f,"
                    "%.8f,%.8f,%.2f,"
                    "%.2f,%.2f,"
                    "%.3f,%.3f,%.3f,%.3f,"
                    "%d,%.3f,%.3f,%.3f,%.3f\n",
                    (long)tick, i,
                    rad2deg(config.tx_azimuth), rad2deg(config.tx_elevation),
                    t->pos.lat, t->pos.lon, t->pos.alt,
                    t->speed, t->heading,
                    paths[i].tx_dist, paths[i].rx_dist,
                    paths[i].route_diff, paths[i].bistatic_speed,
                    paths[i].delay, paths[i].doppler,
                    rad2deg(paths[i].azimuth), rad2deg(paths[i].elevation),
                    10.0 * log10(paths[i].attenuation)
                );
            }
        }

        if (output_truth != NULL) fflush(output_truth);

        if (valid_cnt == 0) goto fastforward;

        // Render each path by index i
        for (int i = 0; i < config.target_len; i++) {
            if (targets[i] == NULL) continue;
            render_path(&paths[i]);
        }
        tpool_wait(workers);

        __clean_output();

        for (int ch = 0; ch < config.rx_array->channel_count; ch++) {
            add_noise_only(__get_channel(ch), -90.0);
        }

        // ===== Inject direct path according to direct_path_mode =====
        if (config.direct_path_mode != DP_MODE_NONE) {
            const double dp_leak_db = -50.0;
            const double dp_leak_lin = pow(10.0, dp_leak_db / 20.0);

            if (config.direct_path_mode == DP_MODE_FIXED) {
                // Inject equal amplitude to all channels
                for (int ch = 0; ch < config.rx_array->channel_count; ch++) {
                    addch_saaxpy(__get_channel(ch), dp_leak_lin, 1.0, __get_reference());
                }
            } else if (config.direct_path_mode == DP_MODE_STEERED) {
                // Use steering vector (ensure tx_steering is calculated)
                if (config.tx_steering != NULL) {
                    for (int ch = 0; ch < config.rx_array->channel_count; ch++) {
                        addch_saaxpy(__get_channel(ch),
                                    config.tx_steering[ch] * dp_leak_lin,
                                    1.0,
                                    __get_reference());
                    }
                }
            }
        }

        // ===== Finally overlay target echoes (unchanged) =====
        for (int ch = 0; ch < config.rx_array->channel_count; ch++) {
            for (int i = 0; i < config.target_len; i++) {
                if (targets[i]) {
                    addch_saaxpy(__get_channel(ch),
                                paths[i].steering[ch],
                                1.0,
                                paths[i].observation);
                }
            }
        }

        // ===== Write array data (all channels) =====
        if (output != NULL) {
            fwrite(__get_channel(0),
                   config.sample_count * config.rx_array->channel_count,
                   sizeof(cf32),
                   output);
            fflush(output);
        }

        // ===== Write reference signal (clean ref, one block per frame) =====
        if (output_ref != NULL) {
            fwrite(__get_reference(),
                   config.sample_count,
                   sizeof(cf32),
                   output_ref);
            fflush(output_ref);
        }

        // ===== Write timestamp =====
        if (output_timestamp != NULL) {
            fprintf(output_timestamp, "%ld\n", (long)tick);
            fflush(output_timestamp);
        }

fastforward:
        if (config.step == 0) break;
        else tick += config.step;

    } while (tick < config.end_time);

    free(paths);
    free(targets);

    if (output != NULL)           fclose(output);
    if (output_timestamp != NULL) fclose(output_timestamp);
    if (output_ref != NULL)       fclose(output_ref);

    // ===== New: Close truth csv =====
    if (output_truth != NULL)     fclose(output_truth);
}

