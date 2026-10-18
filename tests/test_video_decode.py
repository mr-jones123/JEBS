"""Decode actual four-frame media and preserve its observed temporal order."""

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mdprep.verify_video import center, color_points, extract_frames
from mdprep.video_media import _render_clip, _render_frame
from mdprep.videos import _collision_trajectories


class EncodedTemporalOrderTests(unittest.TestCase):
    def test_actual_encoded_colors_retain_all_four_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            colors=((201,79,70),(76,149,108),(57,120,184),(224,184,61))
            for index, color in enumerate(colors):
                Image.new("RGB",(256,192),color).save(root/f"frame-{index:02d}.png")
            path=root/"ordered.mp4"
            subprocess.run(["ffmpeg","-v","error","-y","-framerate","2","-start_number","0",
                            "-i",str(root/"frame-%02d.png"),"-frames:v","4","-c:v","libx264",
                            "-crf","0","-pix_fmt","yuv444p",str(path)],check=True,capture_output=True)
            self.assertEqual(extract_frames(path,"color_sequence"),
                             [{"focus_color":"red"},{"focus_color":"green"},
                              {"focus_color":"blue"},{"focus_color":"yellow"}])

    def test_training_encoding_preserves_exact_gate_arrivals(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"gate.mp4"
            render={"kind":"gate_order","decorations":[],
                    "amber_xs":[160,189,164,128],"violet_xs":[223,171,128,102]}
            _render_clip({"absolute_path":path,"relative_path":"gate.mp4","scene":{"render":render}},
                         "ffmpeg","ffprobe")
            self.assertEqual(extract_frames(path,"gate_order"),
                             [{"amber_circle_center":(160,70),"violet_square_center":(223,126)},
                              {"amber_circle_center":(189,70),"violet_square_center":(171,126)},
                              {"amber_circle_center":(164,70),"violet_square_center":(128,126)},
                              {"amber_circle_center":(128,70),"violet_square_center":(102,126)}])

    def test_generated_contact_keeps_both_centers_observable(self):
        amber,violet=next((a,v) for a,v in _collision_trajectories()
                          if any(math.dist(p,q)<=32 for p,q in zip(a,v)))
        index=min(range(4),key=lambda i:math.dist(amber[i],violet[i]))
        image=_render_frame({"kind":"collision","decorations":[],
                             "amber_points":amber,"violet_points":violet},index)
        self.assertGreaterEqual(len(color_points(image,"amber")),12)
        observed_violet=center(image,"violet")
        self.assertEqual(observed_violet,violet[index])
        self.assertEqual(center(image,"amber",occluder=observed_violet),amber[index])


if __name__=="__main__":
    unittest.main()
