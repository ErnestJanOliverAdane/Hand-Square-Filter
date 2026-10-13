"""Hand-tracking square filter.

Make a square with your hand(s); a filter shows inside it. Touch your thumb to your little finger to change the filter.
Everything runs on this computer; video is not saved or sent anywhere.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

import cv2

import hud
import square
from filters import LandmarkSmoother

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model(path: str) -> str:
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    print("Downloading the hand model (about 8 MB, one time)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not download the hand model: {exc}\nDownload it from\n  {MODEL_URL}\nand save it as\n  {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--no-hud", action="store_true", help="start with the HUD hidden (press H to show)")
    ap.add_argument("--no-face", action="store_true", help="do not draw the face box")
    ap.add_argument("--smooth", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--filter", default=None, help="start filter (e.g. neon)")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(args.model)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Close other apps using it (Zoom, Teams) or try --camera 1.")

    sf = square.SquareFilter()
    if args.filter:
        names = sf.switcher.names
        if args.filter not in names:
            sys.exit("Unknown filter. Choose from: " + ", ".join(names))
        sf.switcher.index = names.index(args.filter)
    smooth = [LandmarkSmoother(True, args.smooth), LandmarkSmoother(True, args.smooth)]
    face = hud.FaceBox()
    show_hud = not args.no_hud
    t0 = last = time.monotonic()
    fps = 0.0
    print("Running. Make a square with your hands. Touch thumb to little finger to change the filter. Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int((now - t0) * 1000))
            raw = sorted(([(p.x, p.y) for p in hd] for hd in res.hand_landmarks), key=lambda hd: sum(x for x, _ in hd)) if res.hand_landmarks else []
            hands = [smooth[i](hd, now) for i, hd in enumerate(raw[:2])]
            if not hands:
                for s in smooth:
                    s.reset()
            box = None if args.no_face else face.update(frame, now)
            out = sf.render(frame, hands, now)
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            if show_hud:
                hud.draw_hud(out, hands, box, sf, fps, now, not args.no_face)
            cv2.imshow("Konektado hand square filter", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("n"):
                sf.switcher.next(now)
            if k == ord("h"):
                show_hud = not show_hud
            if k == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                path = os.path.join("screenshots", time.strftime("shot-%Y%m%d-%H%M%S.png"))
                cv2.imwrite(path, out)
                print("Saved", path)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
