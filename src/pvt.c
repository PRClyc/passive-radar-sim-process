#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/HTMLparser.h>
#include <math.h>
#include "config.h"
#include "pvt.h"
#include "gis.h"

// Convert KML file to PVT (Position, Velocity, Time) data structure
int kml_to_pvt(const char* in, struct PVTDB* out) {
    int ret = 0;

    out->list = NULL;

    xmlDocPtr doc = NULL;
    xmlXPathContextPtr context = NULL;
    xmlXPathObjectPtr result = NULL;

    xmlInitParser();

    doc = xmlReadFile(in, NULL, 0);
    if(doc == NULL) {
        fprintf(stderr, "Failed to parse the KML file on path '%s'\n", in);
        ret = PVT_NO_FILE;
        goto cleanup;
    }

    context = xmlXPathNewContext(doc);
    if (context == NULL) {
        fprintf(stderr, "Failed to create XPath context.\n");
        ret = PVT_LIBXML_ERROR;
        goto cleanup;
    }

    xmlXPathRegisterNs(context, (xmlChar*)"kml", (xmlChar*)"http://www.opengis.net/kml/2.2");

    const xmlChar* xpathExpr = (xmlChar*)"//kml:Folder[1]/kml:Placemark";

    result = xmlXPathEvalExpression(xpathExpr, context);
    if (result == NULL) {
        ret = PVT_LIBXML_ERROR;
        goto cleanup;
    }

    if(xmlXPathNodeSetIsEmpty(result->nodesetval)) {
        printf("No path found in KML file.\n");
        ret = PVT_EMPTY;
        goto cleanup;
    }

    struct PVT* table = calloc(result->nodesetval->nodeNr, sizeof(struct PVT));
    if(table == NULL) {
        printf("No memory for PVT points.\n");
        ret = PVT_MEMERROR;
        goto cleanup;
    }
    out->list = table;

    for (int i = 0; i < result->nodesetval->nodeNr; i++) {
        xmlNodePtr placemark    = result->nodesetval->nodeTab[i];
        xmlNodePtr name         = xmlFirstElementChild(placemark);
        xmlNodePtr description  = xmlNextElementSibling(name);
        xmlNodePtr timestamp    = xmlNextElementSibling(description);
        xmlNodePtr style        = xmlNextElementSibling(timestamp);
        xmlNodePtr coordinates  = xmlNextElementSibling(style);

        xmlChar* timestampText      = xmlNodeGetContent(xmlFirstElementChild(timestamp));
        xmlChar* coordinatesText    = xmlNodeGetContent(xmlNextElementSibling(xmlFirstElementChild(coordinates)));
        xmlChar* headingText        = xmlNodeGetContent(xmlFirstElementChild(xmlFirstElementChild(style)));

        xmlChar* descText = xmlNodeGetContent(description);

        //printf("Timestamp: %s\n", timestampText);
        //printf("Coordinates: %s\n", coordinatesText);
        //printf("Heading: %s\n", headingText);
        //printf("Description: %s\n", descText);

        struct tm tm;

        sscanf(timestampText, "%u-%u-%uT%u:%u:%u",
            &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
            &tm.tm_hour, &tm.tm_min, &tm.tm_sec
        );

        tm.tm_mon -= 1;
        tm.tm_year -= 1900;

        table[i].time = mktime(&tm);

        sscanf(coordinatesText, "%lf,%lf,%lf", &(table[i].pos.lon), &(table[i].pos.lat), &(table[i].pos.alt));

        table[i].heading = strtod(headingText, NULL);

        table[i].real = true;
        table[i].valid = true;

        if(descText){
            xmlDocPtr desc = htmlReadDoc((const xmlChar*)descText, NULL, NULL, HTML_PARSE_RECOVER);

            if(desc != NULL) {
                xmlXPathContextPtr dcontext = xmlXPathNewContext(desc);

               const xmlChar* dxpathExpr = (xmlChar*)"//div[span/b='Speed:']/span[2]";

               xmlXPathObjectPtr dresult = xmlXPathEvalExpression(dxpathExpr, dcontext);

               if (dresult != NULL && dresult->nodesetval != NULL && dresult->nodesetval->nodeNr > 0) {
                   xmlNodePtr speedNode = dresult->nodesetval->nodeTab[0];
                   xmlChar* speedText = xmlNodeGetContent(speedNode);

                   table[i].speed = strtod(speedText, NULL) * KNOT_TO_MS;

                   xmlFree(speedText);
               }

               if(dresult != NULL) xmlFree(dresult);
               xmlFree(dcontext);
               xmlFree(desc);
            }
        }

        // Clean up memory
        xmlFree(timestampText);
        xmlFree(coordinatesText);
        xmlFree(headingText);

        xmlFree(descText);
        ret++;
    }


    if(config.verbosity > 0) {
        char _start[20];
        char _end[20];
        strftime(_start, 20, "%Y-%m-%d %H:%M:%S", localtime(&(out->list[0].time)));
        strftime(_end, 20, "%Y-%m-%d %H:%M:%S", localtime(&(out->list[ret-1].time)));

        printf("Track loaded from %s - got %d points.\n", in, ret);
        printf("\tstarts at %lu (%s)\n", (unsigned long)out->list[0].time, _start);
        printf("\tends   at %lu (%s)\n", (unsigned long)out->list[ret-1].time, _end);

    }



    cleanup:
    if(result) xmlXPathFreeObject(result);
    if(context) xmlXPathFreeContext(context);
    if(doc) xmlFreeDoc(doc);
    xmlCleanupParser();

    out->size = ret;

    return ret;
}

// Get PVT data from multiple databases at a specific timestamp
int get_pvts_at(time_t timestamp, struct PVTDB** dbs, struct PVT** curs, int len) {
    int count = 0;
    for(int i=0; i<len; i++) {
        curs[i] = calloc(1, sizeof(struct PVT));
        if(get_pvt_at(timestamp, dbs[i], curs[i]) < 0) {
            //printf("WARN: no PVT found for track #%d\n", i);
            free(curs[i]);
            curs[i] = NULL;
        } else {
            count++;
        }
    }
    return count;
}

// Get interpolated PVT data at a specific timestamp from a single PVT database
int get_pvt_at(time_t timestamp, struct PVTDB* db, struct PVT* cur) {
    if(db == NULL)      return -1;
    if(cur == NULL)     return -2;
    if(db->size <= 0)   return -3;

    if(timestamp < db->list[0].time) {
        return -4;
    } else if(timestamp > db->list[db->size-1].time) {
        return -5;
    }

    struct PVT* pre     = NULL;
    struct PVT* post    = NULL;
    for(int i=0; i < db->size-1; i++) {
        if(db->list[i].time >= timestamp && post == NULL) {
            post = &(db->list[i]);

            if(post->time != timestamp && i >= 1) pre = &(db->list[i-1]);
            else pre = post;
        }
    }
    if(post == NULL) {
        post = &(db->list[db->size-1]);
        pre = &(db->list[db->size-2]);
    }

    if(post->time == timestamp) {
        *cur = *post;
        return 0;
    }

    interpolate_pvt(*pre, *post, timestamp, cur );

    return 1;
}

// Interpolate PVT data between two PVT points at a specific timestamp
int interpolate_pvt(struct PVT pvt1, struct PVT pvt2, double timestamp, struct PVT* result) {
    double t = (timestamp - pvt1.time) / (pvt2.time - pvt1.time);

    // Interpolate position, altitude and speed (these are linear quantities, no issues)
    result->pos.lat = pvt1.pos.lat + t * (pvt2.pos.lat - pvt1.pos.lat);
    result->pos.lon = pvt1.pos.lon + t * (pvt2.pos.lon - pvt1.pos.lon);
    result->pos.alt = pvt1.pos.alt + t * (pvt2.pos.alt - pvt1.pos.alt);
    result->speed = pvt1.speed + t * (pvt2.speed - pvt1.speed);

    // === Fix: Correctly interpolate heading ===
    double heading1 = pvt1.heading;
    double heading2 = pvt2.heading;

    // Calculate the shortest difference between two angles (in range [-180, +180))
    double diff = heading2 - heading1;
    if (diff > 180.0) {
        diff -= 360.0;
    } else if (diff < -180.0) {
        diff += 360.0;
    }

    // Interpolate based on the shortest difference
    double interpolated_heading = heading1 + t * diff;

    // Normalize the result to [0, 360)
    interpolated_heading = fmod(interpolated_heading, 360.0);
    if (interpolated_heading < 0) {
        interpolated_heading += 360.0;
    }

    result->heading = interpolated_heading;
    // === End of angle interpolation fix ===

    result->time = timestamp;
    result->real = false;

    return 0;
}

// Filter PVT points based on radius, altitude and speed thresholds
int filter_pvts(struct PVT** pvts, int len, double max_radius, double min_altitude, double min_speed) {
    int count = 0;

    for(int i=0; i<len; i++) {
        int drop = 0;
        if(pvts[i] == NULL) continue;
        double dist = distance(config.rx_location, pvts[i]->pos);
        if(dist > max_radius)               drop += 1;
        if(pvts[i]->pos.alt < min_altitude) drop += 2;
        if(pvts[i]->speed < min_speed)      drop += 4;

        if(drop) {
            free(pvts[i]);
            pvts[i] = NULL;
        } else {
            count++;
        }
    }
    return count;
}

