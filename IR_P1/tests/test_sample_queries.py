"""Validate hardcoded sample Boolean query expectations."""

import os
import subprocess
import sys
import tempfile


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_SCRIPT = os.path.join(PROJECT_ROOT, "search-o-SAURS_search.py")


# Expected values from sample_queries.md. AND checks use counts and postings;
# OR checks use counts only because their postings lists are intentionally long.
SAMPLE_QUERIES = [
    (1, "aeroelastic AND aircraft", "aeroelastic OR aircraft", 5, 84,
     [12, 14, 78, 184, 202]),
    (2, "dynamics AND effects", "dynamics OR effects", 34, 586,
     [110, 140, 190, 201, 210, 286, 290, 297, 328, 342, 395, 531, 650, 714,
      766, 783, 792, 858, 859, 905, 939, 953, 1001, 1004, 1008, 1009, 1066,
      1144, 1165, 1203, 1289, 1296, 1321, 1331]),
    (3, "hypersonic AND wake", "hypersonic OR wake", 5, 211,
     [17, 85, 536, 976, 1183]),
    (4, "flutter AND steady", "flutter OR steady", 11, 151,
     [14, 52, 363, 380, 444, 704, 753, 894, 899, 1272, 1339]),
    (5, "viscosity AND reynolds", "viscosity OR reynolds", 23, 239,
     [7, 43, 50, 54, 55, 62, 73, 80, 115, 132, 151, 185, 255, 328, 417, 630,
      671, 941, 962, 1007, 1082, 1159, 1226]),
    (6, "heat AND stagnation", "heat OR stagnation", 80, 360,
     [24, 54, 62, 82, 84, 101, 110, 123, 142, 272, 283, 294, 303, 328, 329,
      349, 352, 353, 354, 364, 366, 369, 375, 437, 438, 486, 522, 524, 539,
      553, 555, 559, 564, 565, 571, 572, 576, 603, 628, 629, 635, 645, 662,
      667, 668, 689, 707, 773, 876, 899, 983, 1003, 1040, 1096, 1097, 1098,
      1099, 1100, 1101, 1104, 1143, 1158, 1159, 1161, 1191, 1198, 1204, 1215,
      1222, 1236, 1258, 1279, 1295, 1307, 1344, 1348, 1386, 1393, 1394, 1395]),
    (7, "oscillatory AND transonic", "oscillatory OR transonic", 1, 83, [815]),
    (8, "creep AND buckling", "creep OR buckling", 26, 136,
     [833, 950, 951, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
      1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034,
      1035, 1052]),
    (9, "pressure AND wing", "pressure OR wing", 91, 687,
     [14, 76, 92, 97, 147, 189, 191, 205, 222, 225, 226, 227, 230, 247, 250,
      252, 287, 289, 311, 364, 379, 415, 416, 431, 464, 465, 466, 486, 513,
      545, 599, 601, 612, 633, 636, 637, 652, 671, 672, 673, 675, 679, 680,
      681, 692, 693, 694, 695, 696, 704, 714, 757, 781, 783, 791, 794, 798,
      801, 808, 809, 811, 901, 917, 921, 924, 927, 970, 971, 991, 1062, 1064,
      1074, 1075, 1090, 1091, 1092, 1111, 1144, 1184, 1188, 1207, 1208, 1229,
      1233, 1239, 1246, 1271, 1276, 1277, 1336, 1355]),
    (10, "transonic AND nozzle", "transonic OR nozzle", 3, 140, [118, 157, 750]),
    (11, "excitation AND noise", "excitation OR noise", 5, 47,
     [209, 722, 725, 909, 911]),
    (12, "mass AND flutter", "mass OR flutter", 6, 121,
     [380, 442, 701, 747, 874, 1290]),
]


def parse_docids(postings_text):
    """Convert a comma-separated postings list into document IDs."""
    return [int(docid.strip()) for docid in postings_text.split(",") if docid.strip()]


def parse_search_output(output_path):
    """Read and validate the count-and-postings result file format."""
    with open(output_path, "r") as output_file:
        output = output_file.read().strip()

    count_text, separator, postings_text = output.partition("|")
    if not separator:
        raise ValueError("Output must use the format: count | docid1, docid2, ...")

    count = int(count_text.strip())
    docids = parse_docids(postings_text)
    if count != len(docids):
        raise ValueError(f"Output count {count} does not match {len(docids)} document IDs.")
    return count, docids


def run_query(query, output_path):
    """Run one query and return the count and postings written by the searcher."""
    result = subprocess.run(
        [sys.executable, SEARCH_SCRIPT, query, output_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    return parse_search_output(output_path)


def main():
    """Check all hardcoded AND and OR sample queries."""
    failures = 0

    with tempfile.TemporaryDirectory() as temporary_directory:
        for number, and_query, or_query, and_count_expected, or_count_expected, and_docids_expected in SAMPLE_QUERIES:
            and_output = os.path.join(temporary_directory, f"sample_{number}_and.txt")
            or_output = os.path.join(temporary_directory, f"sample_{number}_or.txt")

            and_count, and_docids = run_query(and_query, and_output)
            or_count, _ = run_query(or_query, or_output)

            and_matches = (
                and_count == and_count_expected
                and
                and_docids == and_docids_expected
            )
            or_matches = or_count == or_count_expected

            if and_matches:
                print(f"OK  #{number} AND: count and postings match")
            else:
                failures += 1
                print(
                    f"FAIL #{number} AND: expected count {and_count_expected} "
                    f"and {and_docids_expected}; got count {and_count} and {and_docids}"
                )

            if or_matches:
                print(f"OK  #{number} OR: count matches")
            else:
                failures += 1
                print(
                    f"FAIL #{number} OR: expected count {or_count_expected}; "
                    f"got {or_count}"
                )

    if failures:
        print(f"Sample query validation failed: {failures} mismatch(es).")
        return 1

    print(f"Sample query validation passed: {len(SAMPLE_QUERIES)} AND and OR pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
