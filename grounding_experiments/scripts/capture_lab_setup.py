import os, time
import numpy as np
import pyrealsense2 as rs
from PIL import Image

OUT = "/home/summer2026/projects/lerobotprocessing/outputs/lab_setup"
os.makedirs(OUT, exist_ok=True)
CAMS = {"scene_D435": "346522072484", "wrist_D405": "335122272701"}

# clear any stuck state
ctx = rs.context()
present = {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}
print("devices present:", present, flush=True)
for d in rs.context().query_devices():
    try:
        d.hardware_reset()
    except Exception as e:
        print("reset err", repr(e)[:80], flush=True)
print("waiting 8s for re-enumeration...", flush=True)
time.sleep(8)

for name, serial in CAMS.items():
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    try:
        pipe.start(cfg)
        frames = None
        for _ in range(40):              # warm up so auto-exposure settles
            frames = pipe.wait_for_frames(6000)
        img = np.asanyarray(frames.get_color_frame().get_data())  # RGB
        Image.fromarray(img).save(f"{OUT}/{name}.png")
        print(f"SAVED {name}.png  serial={serial}  mean={img.mean():.1f}", flush=True)
        pipe.stop()
    except Exception as e:
        print(f"{name} FAILED: {repr(e)[:140]}", flush=True)
    time.sleep(0.5)
print("DONE ->", OUT, flush=True)
