all:
	nvcc -O3 cuts.cu -o cuts

debug:
	nvcc -g cuts.cu -o cuts

clean:
	@rm -f cuts
