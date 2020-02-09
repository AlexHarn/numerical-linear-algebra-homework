all: build/homework.pdf
	cp build/homework.pdf homework.pdf
	cp build/homework.pdf ../hw05_submission.pdf

TeXOptions = -pdf \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/homework.pdf: FORCE | build
	latexmk $(TeXOptions) homework.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
