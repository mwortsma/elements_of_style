
from sys import byteorder
from array import array
from struct import pack
import sys

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 3*1024
FORMAT = pyaudio.paInt16
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)
num_silent = 0
snd_started = False

r = array('h')

def is_silent(data):
    print(max(data))
    return max(data) < THRESHOLD

while(True):
    snd_data = array('h', stream.read(CHUNK_SIZE))
    if byteorder == 'big':
        snd_data.byteswap()
    r.extend(snd_data)
    # print(snd_data)

    silent = is_silent(snd_data)

    if silent and snd_started:
        num_silent += 1
        #print('silent')
    elif not silent and not snd_started:
        snd_started = True
    else:
        pass

    if snd_started and num_silent > 500:
        break

    sys.stdout.flush()

'''
print('here')
# Play back collected sound.
r = r.tostring()
stream.write(r)

# Cleanup the stream and stop PyAudio
stream.stop_stream()
stream.close()
p.terminate()
'''
