import numba.cuda
print(numba.cuda.detect())
device = numba.cuda.select_device(0)
print(f'Name: {device.name}')
print(f'Multiprocessor: {device.MULTIPROCESSOR_COUNT}')
print("Compute capability:", device.compute_capability)
print(numba.cuda.current_context().get_memory_info())
