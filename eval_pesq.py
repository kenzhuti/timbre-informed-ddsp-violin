import argparse
import warnings
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import traceback
from pesq import pesq
import tempfile
import librosa
import scipy

warnings.filterwarnings("ignore", category=UserWarning)

def segment_audio(audio, sr, segment_length=4, stride=0.5):
    frame_size = int(sr * segment_length)
    step_size = int(sr * stride)
    
    segments = []
    start = 0
    while start + frame_size <= len(audio):
        segments.append(audio[start:start+frame_size])
        start += step_size
    
    return segments

def resample_audio(audio, orig_sr, target_sr):
    try:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except:
        from scipy.signal import resample_poly
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return resample_poly(audio, up, down)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir", type=str)
    parser.add_argument("pred_dir", type=str)
    parser.add_argument("--suffix", type=str, default=".wav")
    parser.add_argument("--segment_length", type=float, default=4.0)
    parser.add_argument("--stride", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=600.0)
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)

    pred_files = [
        f for f in pred_dir.rglob(f"*{args.suffix}")
        if "convert" not in str(f.relative_to(pred_dir))
    ]
    
    ref_files = []
    for pred_file in pred_files:
        relative_path = pred_file.relative_to(pred_dir)
        ref_file = ref_dir / relative_path
        if not ref_file.exists():
            print(f"Reference file not found: {ref_file}")
            continue
        ref_files.append(ref_file)
    
    valid_pairs = [(r, p) for r, p in zip(ref_files, pred_files) if r.exists() and p.exists()]
    total_pairs = len(valid_pairs)
    print(f"Found {total_pairs} valid file pairs")
    
    all_segment_scores = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for ref_path, pred_path in tqdm(valid_pairs, desc="Processing files"):
            try:
                ref_audio, ref_sr = sf.read(ref_path)
                pred_audio, pred_sr = sf.read(pred_path)
                
                duration = max(len(ref_audio)/ref_sr, len(pred_audio)/pred_sr)
                
                if duration > args.max_duration:
                    print(f"Audio too long ({duration:.1f}s) - skipping {ref_path}")
                    continue
                
                if ref_sr != pred_sr or ref_sr != 16000:
                    ref_audio = resample_audio(ref_audio, ref_sr, 16000)
                    pred_audio = resample_audio(pred_audio, pred_sr, 16000)
                    sr = 16000
                else:
                    sr = ref_sr
                
                min_length = min(len(ref_audio), len(pred_audio))
                ref_audio = ref_audio[:min_length]
                pred_audio = pred_audio[:min_length]
                
                ref_segments = segment_audio(ref_audio, sr, args.segment_length, args.stride)
                pred_segments = segment_audio(pred_audio, sr, args.segment_length, args.stride)
                
                num_segments = min(len(ref_segments), len(pred_segments))
                
                file_scores = []
                for i in range(num_segments):
                    try:
                        score = pesq(sr, ref_segments[i], pred_segments[i], 'wb')
                        file_scores.append(score)
                        all_segment_scores.append(score)
                    except Exception as e:
                        segment_path = Path(tmpdir) / f"{ref_path.stem}_segment_{i}.wav"
                        sf.write(segment_path, ref_segments[i], sr)
                        print(f"Segment error: {str(e)}. Saved as {segment_path}")
                
                if file_scores:
                    avg_score = np.mean(file_scores)
                    print(f"{ref_path.name}: {len(file_scores)} segments, avg PESQ = {avg_score:.3f}")
                else:
                    print(f"No valid segments for {ref_path.name}")
                    
            except Exception as e:
                print(f"Error processing {ref_path}:")
                print(f"{type(e).__name__}: {str(e)}")
                print(traceback.format_exc())
                continue

    if not all_segment_scores:
        print("No valid segments processed")
        exit(1)
    
    all_segment_scores = np.array(all_segment_scores)
    print("\n" + "="*50)
    print(f"Processed {len(all_segment_scores)} audio segments")
    print(f"PESQ mean: {np.mean(all_segment_scores):.4f}")
    print(f"PESQ std: {np.std(all_segment_scores):.4f}")
    print(f"PESQ min: {np.min(all_segment_scores):.4f}")
    print(f"PESQ max: {np.max(all_segment_scores):.4f}")
    print("="*50)


