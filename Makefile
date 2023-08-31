all:
	nvcc -O3 -lnuma cuts.cu -o cuts

debug:
	nvcc -g -lnuma cuts.cu -o cuts

clean:
	@rm -f cuts
