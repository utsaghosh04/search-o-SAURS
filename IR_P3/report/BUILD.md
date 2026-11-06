# Building the report

The editable document is `Report_PA3_search-o-SAURS.tex`. Its numerical tables
are in `report_data.tex`, which is already included. Figures are drawn directly
in LaTeX with TikZ and PGFPlots; there are no external image assets.

From this folder, compile with either:

```text
tectonic -X compile Report_PA3_search-o-SAURS.tex
```

or a standard LaTeX installation:

```text
pdflatex -interaction=nonstopmode -halt-on-error Report_PA3_search-o-SAURS.tex
pdflatex -interaction=nonstopmode -halt-on-error Report_PA3_search-o-SAURS.tex
```

Two pdflatex passes resolve the table of contents. The document uses standard
packages: geometry, fontenc, lmodern, amsmath, amssymb, booktabs, array, tabularx,
graphicx, xcolor, float, fancyhdr, enumitem, listings, tikz, pgfplots and hyperref.

The checked final PDF is in `output/pdf/Report_PA3_search-o-SAURS.pdf`, relative
to the project root. It contains 12 A4 pages: cover, contents and ten report pages.

To regenerate the tables from the project's current experiment outputs, run
`python -B report/prepare_report.py` from the project root before recompiling.
This also recomputes the supplemental common-cutoff analysis using the fixed
saved models. It does not retrain or modify the original experiment outputs.
The statistics script requires the surrounding project data and saved weights;
compiling the included TeX files alone does not.
