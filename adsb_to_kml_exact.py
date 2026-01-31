# adsb_to_kml_exact.py — Generate KML files compatible with C language parser
# Usage examples:
#   python adsb_to_kml_exact.py --callsign AF1294 --out AF1294.kml
#   python adsb_to_kml_exact.py --hex 78067B --out track_78067B.kml
import time
import requests
import argparse
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from datetime import datetime, timezone

# Synchronize with dump1090 port (modify according to your actual configuration)
JSON_URL_DEFAULT = "http://127.0.0.1:8080/data/aircraft.json"

def iso_when_with_offset(ts):
    """Return time format parsed by C code: YYYY-MM-DDTHH:MM:SS+00:00 (precision to second)"""
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.isoformat(timespec="seconds")  # e.g., 2023-10-18T07:26:24+00:00

def name_utc(ts):
    """Return Placemark name <expected by C code>: YYYY-MM-DD HH:MM:SS UTC"""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def write_kml_matched(rows, kml_path, name="Flight Track"):
    """
    Generate KML file fully compatible with C language parser:
    1. Contains the first Folder tag, all Placemarks inside Folder
    2. Placemark child element order: name → description → TimeStamp → Style → Point
    3. Strictly match the nested structure and content format of each tag
    """
    # Register KML namespace (ensure C code's XPath can recognize kml: prefix)
    ElementTree.register_namespace("", "http://www.opengis.net/kml/2.2")
    kml = Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    doc = SubElement(kml, "Document")
    SubElement(doc, "name").text = name

    # Core requirement: Add the first Folder tag (C code XPath depends on //kml:Folder[1])
    folder = SubElement(doc, "Folder")
    SubElement(folder, "name").text = "Track Points"  # Folder name doesn't affect parsing, only for identification

    # Generate each Placemark (strictly follow child element order required by C code)
    for r in rows:
        pm = SubElement(folder, "Placemark")  # Placemark must be inside Folder

        # 1. First child element: name (optionally add brief altitude display)
        name_text = f"{name_utc(r['ts'])} | Alt: {r['alt_ft']:.0f} ft"
        SubElement(pm, "name").text = name_text

        # 2. Second child element: description (fixed HTML structure, add altitude display in meters)
        desc = SubElement(pm, "description")
        altitude_m = int(round(r["alt_ft"] * 0.3048))  # Pre-calculate altitude in meters
        desc.text = (
            "<div>"
            "<div><span><b>Altitude (ft):</b></span> <span>{:.0f} ft</span></div>"
            "<div><span><b>Altitude (m):</b></span> <span>{:.0f} m</span></div>"
            "<div><span><b>Speed:</b></span> <span>{:.0f} kt</span></div>"
            "<div><span><b>Heading:</b></span> <span>{:.0f}&deg;</span></div>"
            "</div>"
        ).format(r["alt_ft"], altitude_m, r["speed_kt"], r["heading_deg"])

        # 3. Third child element: TimeStamp (contains only when sub-tag)
        ts = SubElement(pm, "TimeStamp")
        SubElement(ts, "when").text = iso_when_with_offset(r["ts"])

        # 4. Fourth child element: Style (nested IconStyle→heading, match C code parsing hierarchy)
        st = SubElement(pm, "Style")
        ic = SubElement(st, "IconStyle")
        # Round heading to integer, match C code's strtod parsing
        SubElement(ic, "heading").text = f"{int(round(r['heading_deg']))}"
        icon = SubElement(ic, "Icon")
        SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/airports.png"

        # 5. Fifth child element: Point (contains altitudeMode and coordinates)
        pt = SubElement(pm, "Point")
        SubElement(pt, "altitudeMode").text = "absolute"
        # Convert feet to meters and round, write as third value of coordinates (altitude)
        SubElement(pt, "coordinates").text = f"{r['lon']},{r['lat']},{altitude_m}"

    # Write KML file (UTF-8 encoding + XML declaration, ensure C code's libxml2 can parse)
    ElementTree.ElementTree(kml).write(kml_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Generated C parser-compatible KML file: {kml_path}")
    print(f"[INFO] Contains {len(rows)} track points, all Placemarks inside first Folder")

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=JSON_URL_DEFAULT, help="dump1090 aircraft.json URL")
    ap.add_argument("--callsign", help="Target callsign (priority)")
    ap.add_argument("--hex", help="Or specify ICAO Hex (e.g., 78067B)")
    ap.add_argument("--interval", type=float, default=2.0, help="Capture interval (seconds)")
    ap.add_argument("--out", default="track_matched.kml", help="Output KML filename")
    ap.add_argument("--name", default="Flight Track", help="KML document name")
    args = ap.parse_args()

    if not (args.callsign or args.hex):
        print("[ERROR] Please specify target aircraft with --callsign or --hex", file=sys.stderr)
        sys.exit(1)

    rows = []
    print("[INFO] Starting ADS-B data collection, press Ctrl+C to stop...")
    try:
        while True:
            try:
                # Fetch dump1090 aircraft.json data
                response = requests.get(args.url, timeout=2)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Failed to fetch data: {e}, retrying in {args.interval} seconds")
                time.sleep(args.interval)
                continue

            now = time.time()
            matched = False
            for ac in data.get("aircraft", []):
                # Match callsign or ICAO Hex
                if args.callsign:
                    cs = (ac.get("flight") or "").strip()
                    matched = (cs == args.callsign)
                elif args.hex:
                    matched = (ac.get("hex") or "").lower() == args.hex.lower()

                if not matched:
                    continue

                # Extract latitude and longitude (skip if missing)
                lat, lon = ac.get("lat"), ac.get("lon")
                if lat is None or lon is None:
                    continue

                # Extract flight parameters (core fix: read altitude field instead of alt_baro, enhance robustness)
                # Altitude: handle numeric/string types, set to 0 if missing
                alt_raw = ac.get("altitude")
                if isinstance(alt_raw, (int, float)):
                    alt_ft = alt_raw
                elif isinstance(alt_raw, str) and alt_raw.isdigit():
                    alt_ft = float(alt_raw)
                else:
                    alt_ft = 0.0

                # Speed: prioritize gs (ground speed), then speed, set to 0 if missing
                spd_raw = ac.get("gs") or ac.get("speed")
                if isinstance(spd_raw, (int, float)):
                    spd_kt = spd_raw
                elif isinstance(spd_raw, str) and spd_raw.isdigit():
                    spd_kt = float(spd_raw)
                else:
                    spd_kt = 0.0

                # Heading: get track field, set to 0 if missing
                hdg_raw = ac.get("track")
                if isinstance(hdg_raw, (int, float)):
                    heading_deg = hdg_raw
                elif isinstance(hdg_raw, str) and hdg_raw.isdigit():
                    heading_deg = float(hdg_raw)
                else:
                    heading_deg = 0.0

                # Add to data rows
                rows.append({
                    "ts": int(now),
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt_ft": float(alt_ft),
                    "speed_kt": float(spd_kt),
                    "heading_deg": float(heading_deg),
                })
                break  # Exit loop after finding target aircraft

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Collection terminated by user, starting KML file generation...")

    # Check if data was collected
    if not rows:
        print("[WARN] No aircraft data captured, please check callsign/hex or dump1090 connection")
        return

    # Generate C parser-compatible KML file
    write_kml_matched(rows, args.out, name=args.name)
    print(f"[OK] Completed! KML file path: {args.out}")

if __name__ == "__main__":
    main()

