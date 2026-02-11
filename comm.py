import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
bit_duration = 0.01   # seconds
fs = 10000            # sampling frequency
distance = 5          # meters
alpha = 0.2           # attenuation coefficient
I0 = 1                # initial intensity

# MESSAGE
msg = "HELLO"

# TEXT -> BINARY
binary_msg = ''.join(format(ord(c), '08b') for c in msg)
binary_bits = np.array(list(binary_msg), dtype=int)

# OOK MODULATION
samples_per_bit = int(bit_duration * fs)
signal = np.repeat(binary_bits, samples_per_bit) * I0

# UNDERWATER ATTENUATION
attenuation = np.exp(-alpha * distance)
received_signal = signal * attenuation

# ADD GAUSSIAN NOISE
noise_power = 0.1
noise = np.random.normal(0, noise_power, received_signal.shape)
received_signal += noise

# DETECTION
threshold = 0.5 * attenuation
received_bits = []

for i in range(len(binary_bits)):
    start = i * samples_per_bit
    end = (i + 1) * samples_per_bit
    sample = np.mean(received_signal[start:end])
    received_bits.append(1 if sample > threshold else 0)

received_bits = np.array(received_bits)

# BINARY -> TEXT
received_binary_str = ''.join(received_bits.astype(str))
received_msg = ''.join(chr(int(received_binary_str[i:i+8], 2)) 
                       for i in range(0, len(received_binary_str), 8))

print("Original Message:", msg)
print("Received Message:", received_msg)

# PLOT
t = np.arange(len(received_signal)) / fs
plt.plot(t, received_signal)
plt.title("Received Signal After Underwater Channel")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
