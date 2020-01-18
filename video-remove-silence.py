#!/usr/bin/env python3

import argparse
import collections
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to video')
parser.add_argument('--audio-file', type=str, default=None, help='Optional audio file - if specified use this audio instead of mp4 file. Must be .wav')
parser.add_argument('--threshold-level', type=float, default=-35, help='threshold level in dB')
parser.add_argument('--threshold-duration', type=float, default=0.4, help='threshold duration in seconds')
parser.add_argument('--constant', type=float, default=0, help='duration constant transform value')
parser.add_argument('--sublinear', type=float, default=0, help='duration sublinear transform factor')
parser.add_argument('--linear', type=float, default=0.2, help='duration linear transform factor')
parser.add_argument('--save-silence', type=str, help='filename for saving silence')
parser.add_argument('--smooth', type=int, default=1, help='Accelerated speed up. This needs to be False if youre using recalculate-time-in-description')
parser.add_argument('--recalculate-time-in-description', type=str, help='path to text file')
parser.add_argument("--initial-grace-period", type=float, default=3.0, help="Allow this much silence at the start in seconds")
args = parser.parse_args()


def _get_json(path):
    result = subprocess.run(['ffprobe', path, '-loglevel', 'quiet', '-print_format', 'json', '-show_streams'], stdout=subprocess.PIPE)
    result.check_returncode()
    return json.loads(result.stdout)

def get_resolution(path):
    for stream in _get_json(path)['streams']:
        if stream['codec_type'] == 'video':
            return stream['width'], stream['height']

def get_frames(path):
    for stream in _get_json(path)['streams']:
        if stream['codec_type'] == 'video':
            if 'nb_frames' in stream:
                return int(stream['nb_frames'])

def get_duration(path):
    for stream in _get_json(path)['streams']:
        if stream['codec_type'] == 'video':
            if 'duration' in stream:
                return float(stream['duration'])
            else:
                parts = stream['tags']['DURATION'].split(':')
                assert len(parts) == 3
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def get_frame_rate(path):
    for stream in _get_json(path)['streams']:
        if stream['codec_type'] == 'video':
            if 'avg_frame_rate' in stream:
                assert stream['avg_frame_rate'].count('/') <= 1
                parts = stream['avg_frame_rate'].split('/')
                result = float(parts[0])
                if len(parts) == 2:
                    result /= float(parts[1])
                return result


def find_silences(filename):
    global args
    blend_duration = 0.005
    with wave.open(filename) as wav:
        size = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        max_value = 1 << (8 * sample_width - 1)
        half_blend_frames = int(blend_duration * frame_rate / 2)
        blend_frames = half_blend_frames * 2
        assert size > blend_frames > 0
        square_threshold = max_value ** 2 * 10 ** (args.threshold_level / 10)
        blend_squares = collections.deque()
        blend = 0

        def get_values():
            frames_read = 0
            while frames_read < size:
                frames = wav.readframes(min(0x1000, size - frames_read))
                frames_count = len(frames) // sample_width // channels
                for frame_index in range(frames_count):
                    yield frames[frame_index*channels*sample_width:(frame_index+1)*channels*sample_width]
                frames_read += frames_count

        def get_is_silence(blend):
            results = 0
            frames = get_values()
            for index in range(half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
            for index in range(size-half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
                if index < half_blend_frames:
                    yield blend < square_threshold * channels * (half_blend_frames + index + 1)
                else:
                    result = blend < square_threshold * channels * (blend_frames + 1)
                    if result:
                        results += 1
                    yield result
                    blend -= blend_squares.popleft()
            for index in range(half_blend_frames):
                blend -= blend_squares.popleft()
                yield blend < square_threshold * channels * (blend_frames - index)

        is_silence = get_is_silence(blend)

        def to_regions(iterable):
            iterator = enumerate(iterable)
            while True:
                try:
                    index, value = next(iterator)
                except StopIteration:
                    return
                if value:
                    start = index
                    while True:
                        try:
                            index, value = next(iterator)
                            if not value:
                                yield start, index
                                break
                        except StopIteration:
                            yield start, index+1
                            return
        grace_frames = int(args.initial_grace_period * frame_rate)
        threshold_frames = int(args.threshold_duration * frame_rate)
        silence_regions = ( (start, end) for start, end in to_regions(is_silence) if end-start >= blend_duration )
        silence_regions = ( (start + (half_blend_frames if start > 0 else 0), end - (half_blend_frames if end < size else 0)) for start, end in silence_regions )
        silence_regions = [ (start, end) for start, end in silence_regions if end-start >= threshold_frames ]
        silence_regions = [ (start, end) for start, end in silence_regions if start >= grace_frames or (start < grace_frames and end > grace_frames)  ]
        for index, (start, end) in enumerate(silence_regions):
            if start < grace_frames:
                silence_regions[index] = (grace_frames, end)
        including_end = len(silence_regions) == 0 or silence_regions[-1][1] == size
        silence_regions = [ (start/frame_rate, end/frame_rate) for start, end in silence_regions ]

        if args.save_silence:
            with wave.open(args.save_silence, 'wb') as out_wav:
                out_wav.setnchannels(channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(frame_rate)
                for start, end in silence_regions:
                    s = int(start * frame_rate)
                    e = int(end * frame_rate)
                    wav.setpos(s)
                    frames = wav.readframes(e-s)
                    out_wav.writeframes(frames)

    return silence_regions, including_end


def extract_audio(input_filename, output_filename):
    command = [ 'ffmpeg', '-i', input_filename, '-acodec', 'pcm_s16le', '-f', 'wav', '-y', output_filename ]
    print(f"###### Executing command:    {' '.join(command)}\n\n")
    subprocess.run(command, stderr=subprocess.PIPE).check_returncode()


if __name__ == "__main__":
    if args.audio_file is not None:
        audio_file = args.audio_file
        remove_audio = False
        print('Have extracted audio file')
    else:
        audio_file = tempfile.NamedTemporaryFile(delete=False)
        audio_file.close()
        print('Extracting audio...')
        extract_audio(args.path, audio_file.name)
        audio_file = audio_file.name
        remove_audio = True
    if args.smooth:
        print("Doing smoothing!!!!!")

    def transform_duration(duration):
        global args
        return args.constant + args.sublinear * math.log(duration + 1) + args.linear * duration

    print('Finding gaps...')
    silences, including_end = find_silences(audio_file)

    total_duration = sum((end-start for start, end in silences))
    print(f"Total duration is {total_duration}")
    if len(silences) == 0:
        print('Everything is fine')
        sys.exit(0)

    print('Found {} gaps, {:.1f} seconds total'.format(len(silences), total_duration))

    for i, (start, end) in enumerate(silences):
        diff = end - start
        print(f"Silence {i} is {diff:0.1f} seconds from {start:0.1f} to {end:0.1f}")

    regions = []
    if silences[0][0] > 0:
        regions.append((0, silences[0][0], False))
    for silence, next_silence in zip(silences[:-1], silences[1:]):
        regions.append((silence[0], silence[1], True))
        regions.append((silence[1], next_silence[0], False))
    if including_end:
        regions.append((silences[-1][0], None, True))
    else:
        regions.append((silences[-1][0], silences[-1][1], True))
        regions.append((silences[-1][1], None, False))

    def format_offset(offset):
        return '{}:{}:{}'.format(int(offset) // 3600, int(offset) % 3600 // 60, offset % 60)

    frames = get_frames(args.path)
    duration = get_duration(args.path)
    if frames:
        frame_rate = frames / duration # N.B. Possibly we need to simply read frame rate instead of calculating it.
    else:
        frame_rate = get_frame_rate(args.path)
        frames = int(frame_rate * duration)

    width, height = get_resolution(args.path)

    def closest_frames(duration, frame_rate):
        return int((duration + 1 / frame_rate / 2) // (1 / frame_rate))

    if args.recalculate_time_in_description:
        with open(args.recalculate_time_in_description, encoding='utf-8') as description_file:
            description = description_file.read()
        time_codes = {}
        for time_code, h, m, s in re.findall('((\d+):(\d\d):(\d\d))', description):
            time_codes[time_code] = int(h)*3600+int(m)*60+int(s)
        current_position = 0
        for start, end, is_silence in regions:
            start_frame = int(start * frame_rate)
            end_frame = frames if end is None else int(end * frame_rate)
            duration = (end_frame - start_frame) / frame_rate
            if is_silence:
                new_duration = transform_duration(duration)
            else:
                new_duration = duration
            for time_code, time_value in list(time_codes.items()):
                if start <= time_value < end:
                    time_codes[time_code] = int((time_value-start)/duration*new_duration+current_position)
            current_position += new_duration
        for time_code, time_value in sorted(time_codes.items(), key=lambda x: x[1], reverse=True):
            description = description.replace(time_code, '{:d}:{:02d}:{:02d}'.format(time_value // 3600, (time_value // 60) % 60, time_value % 60))
        description_base_name, description_extension = os.path.splitext(args.recalculate_time_in_description)
        with open('{}_result{}'.format(*os.path.splitext(args.recalculate_time_in_description)), 'w', encoding='utf-8') as description_file:
            description_file.write(description)

    print('Processing {} frames...'.format(frames))
    command = [ 'ffmpeg', '-i', args.path, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-' ]
    print(f"###### Executing command:    {' '.join(command)}\n\n")
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    decoder.stderr.close()

    video_track = tempfile.NamedTemporaryFile(delete=False)
    video_track.close()
    print(f"Video track going to {video_track.name}")
    command = [ 'ffmpeg', '-framerate', str(frame_rate), '-s', '{}x{}'.format(width, height), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-i', '-' ]
    command += [ '-f', 'mp4', '-pix_fmt', 'yuv420p', '-crf', '18', '-y', video_track.name ]
    print(f"###### Executing command:    {' '.join(command)}\n\n")
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)

    audio_track = tempfile.NamedTemporaryFile(delete=False)
    audio_track.close()
    print(f"Audio track going to {audio_track.name}")

    wav = wave.open(audio_file)
    out_wav = wave.open(audio_track.name, 'wb')
    size = wav.getnframes()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    audio_frame_rate = wav.getframerate()
    out_wav.setnchannels(channels)
    out_wav.setsampwidth(sample_width)
    out_wav.setframerate(audio_frame_rate)

    def compress_audio(args, wav, start_frame, end_frame, result_frames):
        if result_frames*2 <= end_frame - start_frame:
            left_length = result_frames
            right_length = result_frames
        else:
            left_length = (end_frame - start_frame + 1) // 2
            right_length = end_frame - start_frame - left_length
        crossfade_length = right_length + left_length - result_frames
        if result_frames == 0:
            return b''
        elif result_frames == end_frame - start_frame or crossfade_length == 1:
            wav.setpos(start_frame)
            return wav.readframes(result_frames)
        else:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_width = sample_width*channels
            crossfade_start = (result_frames - crossfade_length) // 2
            wav.setpos(start_frame)
            left_frames = wav.readframes(left_length)
            wav.setpos(end_frame - right_length)
            right_frames = wav.readframes(right_length)
            result = bytearray(b'\x00'*result_frames*frame_width)
            result[:(left_length-crossfade_length)*frame_width] = left_frames[:-crossfade_length*frame_width]
            result[-(right_length-crossfade_length)*frame_width:] = right_frames[crossfade_length*frame_width:]
            for i in range(crossfade_length):
                r = i / (crossfade_length - 1)
                l = 1 - r
                for channel in range(channels):
                    signal_left = int.from_bytes(left_frames[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width], 'little', signed=True)
                    signal_right = int.from_bytes(right_frames[i*frame_width+channel*sample_width:i*frame_width+(channel+1)*sample_width], 'little', signed=True)
                    result[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width] = int(signal_left*l + signal_right*r).to_bytes(sample_width, 'little', signed=True)
            return result

    audio_remainder_frames = 0.0
    for start, end, is_silence in regions:
        start_frame = int(start * frame_rate)
        end_frame = frames if end is None else int(end * frame_rate)
        audio_start_frame = min(int(start * audio_frame_rate), size)
        audio_end_frame = size if end is None else min(int(end * audio_frame_rate), size)
        if is_silence:
            duration = (end_frame - start_frame) / frame_rate
            new_duration = transform_duration(duration)
            new_frames_count = closest_frames(new_duration, frame_rate)
            weights = np.linspace(1, 0, new_frames_count)
            new_frames = set()
            for index, weight in enumerate(weights):
                new_frame = start_frame + int((index + 0.5) * (end_frame - start_frame) / new_frames_count)
                original_frame = start_frame + index
                # This does linear interpolation between frames
                # Lets make this smooth
                if args.smooth:
                    w2 = weight ** 0.8
                    new_frame2 = (1 - w2) * new_frame + w2 * original_frame
                    new_frame = new_frame2
                new_frame = int(new_frame)
                assert not new_frame in new_frames
                assert new_frame >= start_frame
                assert new_frame < end_frame
                new_frames.add(new_frame)
            audio_delta_frames = audio_remainder_frames + (duration - new_frames_count / frame_rate) * audio_frame_rate
            audio_remainder_frames = audio_delta_frames - int(audio_delta_frames)
            if int(audio_delta_frames) > audio_end_frame - audio_start_frame:
                audio_remainder_frames += audio_delta_frames - (audio_end_frame - audio_start_frame)
                audio_delta_frames = audio_end_frame - audio_start_frame
            audio_result_frames = audio_end_frame - audio_start_frame - int(audio_delta_frames)
        else:
            new_frames = set(range(start_frame, end_frame))
            audio_result_frames = audio_end_frame-audio_start_frame
        for index in range(start_frame, end_frame):
            frame = decoder.stdout.read(width*height*3)
            if index in new_frames:
                encoder.stdin.write(frame)
        out_wav.writeframes(compress_audio(args, wav, audio_start_frame, audio_end_frame, audio_result_frames))

    wav.close()
    if remove_audio:
        os.unlink(audio_file)
    out_wav.close()

    encoder.stdin.close()

    encoder.wait()
    assert encoder.returncode == 0

    decoder.terminate()

    name, extension = os.path.splitext(args.path)
    command = [ 'ffmpeg', '-f', 'mp4', '-i', video_track.name, '-f', 'wav', '-i', audio_track.name ]
    command += [ '-c:v', 'libx264', '-crf', '23',  '-c:a', 'aac', '-ab', '128000', '-ac', '1', '-y', '{}_result{}'.format(name, extension) ]
    print(f"Executing command:    {' '.join(command)}\n\n")
    subprocess.run(command)

    os.unlink(audio_track.name)
    os.unlink(video_track.name)
