# PA1 - Sample Boolean Queries to Verify Your Implementation

Use the queries below to sanity-check your preprocessing, indexing, and Boolean search programs before
submission. Each query has a known expected result, computed from `cran.all` using the standard
pipeline: tokenize -> remove stopwords (the provided `stopwords.txt`) -> stem (Porter) -> normalize.

## How to use this

1. Build your index (`groupname_cran.index`) from `cran.all` as specified in the assignment.
2. For each query below, stem/normalize the two terms the same way your program stems query terms
   (this must be the same pipeline you used for indexing).
3. Run your Boolean search for both the `AND` and the `OR` version.
4. Compare your resulting docid list against the "Expected result" column.

**A note on small mismatches:** if a couple of docids (+/-2-3) differ from the expected list, that's
usually fine - it typically comes from small, defensible differences in tokenization, such as how you
split hyphenated words (e.g. `re-entry` -> `re`, `entry` vs. `reentry`), whether you drop pure numbers,
or how you handle slashes (e.g. `/slip flow/`). If your results are wildly different (empty when a
result is expected, or off by a large margin), first check your stopword removal and stemming - these
two steps cause the majority of mismatches.

## Sample queries and expected results

| # | Query (AND) | Query (OR) | \|AND\| | \|OR\| | Expected AND result (docids) |
|---|---|---|---|---|---|
| 1 | `aeroelastic AND aircraft` | `aeroelastic OR aircraft` | 5 | 84 | 12, 14, 78, 184, 202 |
| 2 | `dynamics AND effects` | `dynamics OR effects` | 34 | 586 | 110, 140, 190, 201, 210, 286, 290, 297, 328, 342, 395, 531, 650, 714, 766, 783, 792, 858, 859, 905, 939, 953, 1001, 1004, 1008, 1009, 1066, 1144, 1165, 1203, 1289, 1296, 1321, 1331 |
| 3 | `hypersonic AND wake` | `hypersonic OR wake` | 5 | 211 | 17, 85, 536, 976, 1183 |
| 4 | `flutter AND steady` | `flutter OR steady` | 11 | 151 | 14, 52, 363, 380, 444, 704, 753, 894, 899, 1272, 1339 |
| 5 | `viscosity AND reynolds` | `viscosity OR reynolds` | 23 | 239 | 7, 43, 50, 54, 55, 62, 73, 80, 115, 132, 151, 185, 255, 328, 417, 630, 671, 941, 962, 1007, 1082, 1159, 1226 |
| 6 | `heat AND stagnation` | `heat OR stagnation` | 80 | 360 | 24, 54, 62, 82, 84, 101, 110, 123, 142, 272, 283, 294, 303, 328, 329, 349, 352, 353, 354, 364, 366, 369, 375, 437, 438, 486, 522, 524, 539, 553, 555, 559, 564, 565, 571, 572, 576, 603, 628, 629, 635, 645, 662, 667, 668, 689, 707, 773, 876, 899, 983, 1003, 1040, 1096, 1097, 1098, 1099, 1100, 1101, 1104, 1143, 1158, 1159, 1161, 1191, 1198, 1204, 1215, 1222, 1236, 1258, 1279, 1295, 1307, 1344, 1348, 1386, 1393, 1394, 1395 |
| 7 | `oscillatory AND transonic` | `oscillatory OR transonic` | 1 | 83 | 815 |
| 8 | `creep AND buckling` | `creep OR buckling` | 26 | 136 | 833, 950, 951, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 1052 |
| 9 | `pressure AND wing` | `pressure OR wing` | 91 | 687 | 14, 76, 92, 97, 147, 189, 191, 205, 222, 225, 226, 227, 230, 247, 250, 252, 287, 289, 311, 364, 379, 415, 416, 431, 464, 465, 466, 486, 513, 545, 599, 601, 612, 633, 636, 637, 652, 671, 672, 673, 675, 679, 680, 681, 692, 693, 694, 695, 696, 704, 714, 757, 781, 783, 791, 794, 798, 801, 808, 809, 811, 901, 917, 921, 924, 927, 970, 971, 991, 1062, 1064, 1074, 1075, 1090, 1091, 1092, 1111, 1144, 1184, 1188, 1207, 1208, 1229, 1233, 1239, 1246, 1271, 1276, 1277, 1336, 1355 |
| 10 | `transonic AND nozzle` | `transonic OR nozzle` | 3 | 140 | 118, 157, 750 |
| 11 | `excitation AND noise` | `excitation OR noise` | 5 | 47 | 209, 722, 725, 909, 911 |
| 12 | `mass AND flutter` | `mass OR flutter` | 6 | 121 | 380, 442, 701, 747, 874, 1290 |

For the OR queries, the expected docid lists are long - it's easiest to just check that your result
**count** matches the `|OR|` column rather than comparing every id by hand.
