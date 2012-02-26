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

# Generic GNU g++ compiler flags
CXXFLAGS = -g -Wall -W -pedantic-errors \
	-I /usr/include/opencv \
	-L /usr/lib  \
	-lm -lcv -lhighgui -lcvaux 

# Files to compile
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

# Objects to be compiled
OBJECTS = $(SOURCES:%.cpp=%.o)

# Define a rule for turning .cpp's into .o's
%.o: %.cpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Default target
default: all

all: stereo

# Executable's target
stereo: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o $@
	-@echo "._. <(done)"

# Gets rid of objects and executable
clean:
	-@rm -f stereo
	-@rm -f depend
	-@rm -f *~
	-@rm -f ${OBJECTS}

# Automatically generate dependencies and include them in Makefile
depend: $(SOURCES) $(HEADERS)
	@echo "Generating dependencies"
	@$(CXX) -MM *.cpp > $@


-include depend
