# Makefile for Monster Process Sampler kernel module

obj-m += monster_sampler.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	sudo insmod monster_sampler.ko

uninstall:
	sudo rmmod monster_sampler

logs:
	sudo dmesg | tail -50

stats:
	sudo dmesg | grep "Monster Process Sampler Statistics" -A 20
