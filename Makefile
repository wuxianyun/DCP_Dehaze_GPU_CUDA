#WORK BY WXYUN 5/15/2013

CUFLAGS= -O3 -use_fast_math -m64#-arch sm_20# -m64 --ptxas-options=-v -use_fast_math#-arch sm_20 -gencode=arch=compute_20 #-use_fast_math -m64 #-maxrregcount 31 #_fmad_false_

LIBS = -lcudart -lcuda -lcurand -lcublas
all: DeHaze_cuda


DeHaze_cuda: dehaze.cu
	nvcc $(CUFLAGS) -c dehaze.cu 
	nvcc $(CUFLAGS) -o $@ dehaze.o dehaze_kernel.cu $(LIBS) darkchannel.cu $(LIBS) boxfilter.cu $(LIBS)


	
run:
	#./DeHaze_cuda canon_400_600.raw deHazecanon.raw 400 600 3
	./DeHaze_cuda 0001_1920_1080.raw deHaze0001.raw 1080 1920 3 7
	#./DeHaze_cuda tiananmen1_270_193.raw deHaze7-1.raw 270 193 3
	#./DeHaze_cuda forest_1024_768.raw deHaze_forest.raw 1024 768 3
	#./DeHaze_cuda canyon_450_600.raw deHaze_canyon.raw 450 600 3

clean:
	rm -fr *.mod *.o DeHaze_cuda 
