CC = gcc
CFLAGS = -O3 -Wall -Wextra -Werror

.PHONY: all os gui_kernel gui_iso boot-mint-gpu boot-gpu-os boot-gui-iso boot-os compare-os test-20-packages clean

all: bbc os gui_kernel gpu_os gpu_desktop

# 1. Buildbit Systems & Bare-Metal Kernel Compiler
bbc: bbc.c
	$(CC) $(CFLAGS) bbc.c -o bbc

# 2. AMDGPU Zero-Copy Graphical Desktop OS (Linux Mint Cinnamon Edition + 20-Pkg Store)
gpu_desktop: bbc os/gpu_desktop.bb os/gpu_vfs.bb os/gpu_mint_gui.h
	./bbc os/gpu_desktop.bb -lX11 -lm -o gpu_desktop

# 3. AMDGPU Wavefront Scheduler & Benchmark OS Engine
gpu_os: bbc os/gpu_os.bb os/gpu_vfs.bb
	./bbc os/gpu_os.bb -lX11 -lm -o gpu_os

# 4. Bare-Metal Ring-0 Headless Kernel (Multiboot1 + x86_64 Long Mode + Linux ABI)
os: bbc os/kernel.bb os/linux_abi.bb
	./bbc os/kernel.bb --kernel -o buildbit_os_kernel.elf

# 5. Bare-Metal Ring-0 1280x720 VBE Graphical Desktop Kernel (PS/2 Mouse IRQ12 + Keyboard IRQ1)
gui_kernel: bbc os/kernel_gui.bb os/linux_abi.bb os/baremetal_gui.h
	./bbc os/kernel_gui.bb --kernel -o buildbit_gui_kernel.elf

# 6. Bootable GRUB2 Hybrid ISO Image (QEMU/KVM or USB Flash Drive)
gui_iso: gui_kernel
	mkdir -p /tmp/bb_iso_root/boot/grub
	cp buildbit_gui_kernel.elf /tmp/bb_iso_root/boot/buildbit_gui_kernel.elf
	printf 'set timeout=0\nset default=0\nmenuentry "BuildbitOS v2.3.0 (Ring-0 1280x720 VBE Graphical Desktop)" {\n    multiboot /boot/buildbit_gui_kernel.elf\n    boot\n}\n' > /tmp/bb_iso_root/boot/grub/grub.cfg
	grub-mkrescue -o buildbit_gui_os.iso /tmp/bb_iso_root 2>/dev/null
	rm -rf /tmp/bb_iso_root

# Launch Interactive Linux Mint Cinnamon GPU-OS Desktop (with 20-Package Linux Store)
boot-mint-gpu: gpu_desktop
	./gpu_desktop

# Run Automated 20-Package Linux Compatibility Suite & Save Desktop Frame
test-20-packages: gpu_desktop
	BB_GUI_BENCH_FRAMES=2 ./gpu_desktop

# Boot Bare-Metal Ring-0 Graphical ISO in QEMU/KVM
boot-gui-iso: gui_iso
	qemu-system-x86_64 -enable-kvm -cpu host -m 256M -vga std -cdrom buildbit_gui_os.iso -serial stdio -no-reboot

# Boot Bare-Metal Ring-0 Headless Kernel in QEMU/KVM
boot-os: os
	qemu-system-x86_64 -enable-kvm -cpu host -kernel buildbit_os_kernel.elf -m 128M -display none -serial stdio -no-reboot

# Head-to-Head Benchmark: BuildbitOS Ring-0 Kernel vs. Linux 6.14 Kernel in QEMU/KVM
compare-os: os os/linux_compare_init.c
	@mkdir -p /tmp/bb_linux_initramfs
	@$(CC) -O3 -msse2 -ffast-math -static os/linux_compare_init.c -lm -o /tmp/bb_linux_initramfs/init
	@(cd /tmp/bb_linux_initramfs && echo init | cpio -o -H newc --quiet | gzip -9 > /tmp/linux_bench_initramfs.cpio.gz)
	@echo "=== 1. BOOTING BUILDBIT-OS (30 KB ELF Kernel) ON QEMU/KVM ==="
	@t0=$$(date +%s%N); \
	qemu-system-x86_64 -enable-kvm -cpu host -kernel buildbit_os_kernel.elf -m 128M -display none -serial stdio -no-reboot; \
	t1=$$(date +%s%N); \
	echo ">>> BuildbitOS Total Cold-Boot-to-Poweroff Wall Time: $$(( (t1 - t0) / 1000000 )) ms"
	@echo ""
	@echo "=== 2. BOOTING LINUX KERNEL + C11 INITRAMFS ON QEMU/KVM ==="
	@rm -rf /tmp/bb_linux_initramfs /tmp/linux_bench_initramfs.cpio.gz

# Additional Compiler & Hardware Benchmarks
bench: bbc examples/bare_metal_bench.bb
	./bbc examples/bare_metal_bench.bb -o bench

bench_freestanding: bbc examples/bare_metal_bench.bb
	./bbc examples/bare_metal_bench.bb --freestanding -o bench_freestanding

hello_elf: bbc examples/hello_elf.bb
	./bbc examples/hello_elf.bb --native-elf -o hello_elf

clean:
	rm -f bbc gpu_os gpu_desktop bench bench_freestanding hello_elf sim_c sim_bb_single sim_bb_multi sys_demo sys_demo_free ffi_demo swarm_c_o2 swarm_c_o3 swarm_bb_1core swarm_bb_full *.gen.c
