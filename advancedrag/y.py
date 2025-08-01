import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Should print the GPU ID (0 for the first GPU)
print(torch.cuda.get_device_name(0))  # Should print the GPU name
