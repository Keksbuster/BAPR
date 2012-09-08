# Sample Makefile
#=========================

#compiler options
CC = g++
CPPFLAGS = -g `pkg-config --cflags opencv` 
LDFLAGS = `pkg-config --libs opencv`
#project files
PROGRAM = main
OBJECTS = main.o

#rules
all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CC) -o $@ $+ $(LDFLAGS)

# generic rule for compiling *.cpp -> *.o
%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $*.cpp

clean:
	rm -f $(PROGRAM) $(OBJECTS)


