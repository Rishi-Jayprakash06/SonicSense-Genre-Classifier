import os
import wave
import struct
import math
import random

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
base_dir = 'Data/genres_original'

os.makedirs(base_dir, exist_ok=True)

sample_rate = 22050
duration = 30
samples = sample_rate * duration

for genre in genres:
    genre_dir = os.path.join(base_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)
    for i in range(3):
        freq = 440 + random.randint(-200, 200)
        filename = f'{genre}.{i:05d}.wav'
        filepath = os.path.join(genre_dir, filename)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            for s in range(samples):
                value = int(32767.0 * math.sin(freq * s * 2.0 * math.pi / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframesraw(data)
                
print("Dummy dataset generated successfully!")
