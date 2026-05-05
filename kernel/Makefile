CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
TARGET = closure_lm

all: $(TARGET)

$(TARGET): main.c closure_kernel.c closure_kernel.h
	$(CC) $(CFLAGS) -o $@ main.c closure_kernel.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
