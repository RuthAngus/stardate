LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out
CHECK_RERUN = grep Rerun $*.log

NAME = ms

all: ${NAME}.pdf

%.pdf: %.tex *.tex
	${LATEX} $<
	bibtex ${NAME}
	${LATEX} $<
	${LATEX} $<
	open ${NAME}.pdf

clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
