###############################################################################
#
# File: Makefile
# Author: Michael Wisely
# Date: February 26, 2012
# 
# Course: CS 301
# Project: 2
# 
###############################################################################

.PHONY: all clean

# C++ compiler
CXX = /usr/bin/g++

INCLUDES = $(shell pkg-config --cflags opencv)
LIBS     = $(shell pkg-config --libs opencv)

# Generic GNU g++ compiler flags
CXXFLAGS = -g -Wall -W $(INCLUDES) -O -O2 -O3

# Define a rule for turning .cpp's into .o's
%.o: %.cpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Default target
default: all

all: stitcher

# Executable's target
stitcher: test.o
	${CXX} ${CXXFLAGS} $(LIBS) test.o -o $@
	-@echo "._. <(done)"

# Gets rid of objects and executable
clean:
	-@rm -f stitcher
	-@rm -f *~ *.o
