all: build/exam.pdf
	cp build/exam.pdf final_exam.pdf
	cp build/exam.pdf ../final_exam_submission.pdf

TeXOptions = -pdf \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/exam.pdf: FORCE | build
	latexmk $(TeXOptions) exam.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
