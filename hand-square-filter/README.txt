HAND SQUARE FILTER (Python)

A filter shows inside a square made by your hand(s). Touch your thumb to your little finger to change it.
Only the hand-tracking filter: it does not touch your mouse, and no face is recognised or saved.
Runs on your computer; video is not saved or sent anywhere.

Start
  1. Install Python 3.10, 3.11 or 3.12 (python.org, tick "Add Python to PATH").
  2. Double-click run-square-filter.bat. First run installs libraries and downloads an 8 MB hand model
     (internet needed once). Allow camera access if Windows asks.

Make the square
  Two hands ..... thumb + index finger out on both hands. The 4 fingertips are the corners of the square.
  One hand ...... make an "L" (thumb + index out, other fingers folded). The square sits between your
                  thumb tip and index tip and turns with your hand. Open the L wider to make it bigger.
Change the filter
  Touch the tip of your THUMB to the tip of your LITTLE FINGER (either hand). One touch = next filter.
  Release and touch again for the next one. The square holds still while you touch, and flashes when it changes.
  Filters: sketch, neon, thermal, night vision, x-ray, cartoon, glitch, pixel, mono, invert, blur.

HUD (like the reference picture)
  Cyan glowing hand skeleton, a red face-detection box (it only finds a face, it does not identify anyone),
  an animated neural-network graphic on the left, a SYSTEM STATUS panel (hand tracking, filter, FPS),
  and GESTURE / BIOMETRIC lines at the bottom.
Keys: N next filter   H hide/show HUD   S save a screenshot (screenshots folder)   Q quit
Options (add after the .bat name):
  --no-hud    --no-face    --filter neon    --smooth low|medium|high    --camera 1

Tips
  Good light, plain background. If it will not start: close Zoom/Teams (camera busy) or try --camera 1.
  If pip fails to install mediapipe, use Python 3.11 or 3.12.

Developers: python -m unittest discover -s tests    (no camera needed)
