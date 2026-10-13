import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import effects  # noqa: E402
import hud  # noqa: E402
import square  # noqa: E402
from helpers import make_hand  # noqa: E402

W, H = 640, 480


def photo():
    rnd = np.random.default_rng(3)
    base = np.linspace(20, 235, W, dtype=np.uint8)[None, :, None].repeat(H, 0).repeat(3, 2)
    return np.clip(base + rnd.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)


def two_hands():
    return [make_hand(("index",), cx=0.22, cy=0.8, scale=1.4, thumb="out", mirror=True), make_hand(("index",), cx=0.78, cy=0.8, scale=1.4, thumb="out")]


class SwitchTests(unittest.TestCase):
    def test_ratio_detects_touch(self):
        self.assertLess(square.thumb_pinky_ratio(make_hand(thumb="pinky")), 0.3)
        self.assertGreater(square.thumb_pinky_ratio(make_hand(thumb="out")), 0.6)
        self.assertGreater(square.thumb_pinky_ratio(make_hand(thumb="in")), 0.35)

    def test_touch_changes_filter_once(self):
        sw = square.FilterSwitcher()
        start = sw.index
        hands_open, hands_touch = [make_hand(thumb="out")], [make_hand(thumb="pinky")]
        t = 0.0
        for _ in range(5):
            sw.update(hands_open, t); t += 0.033
        for _ in range(20):  # holding the touch for 0.66 s must not skip several filters
            sw.update(hands_touch, t); t += 0.033
        self.assertEqual(sw.index, (start + 1) % len(sw.names))

    def test_second_touch_after_release_changes_again(self):
        sw = square.FilterSwitcher()
        t = 0.0
        for _ in range(2):
            for _ in range(5):
                sw.update([make_hand(thumb="out")], t); t += 0.1
            for _ in range(5):
                sw.update([make_hand(thumb="pinky")], t); t += 0.1
        self.assertEqual(sw.index, 2)

    def test_works_with_either_hand(self):
        sw = square.FilterSwitcher()
        sw.update([make_hand(thumb="out"), make_hand(thumb="pinky", mirror=True)], 0.0)
        self.assertEqual(sw.index, 1)

    def test_cooldown_blocks_flutter(self):
        sw = square.FilterSwitcher(cooldown=1.0)
        sw.update([make_hand(thumb="pinky")], 0.0)
        sw.update([make_hand(thumb="out")], 0.1)
        sw.update([make_hand(thumb="pinky")], 0.2)
        self.assertEqual(sw.index, 1)

    def test_no_hands_is_safe(self):
        sw = square.FilterSwitcher()
        self.assertFalse(sw.update([], 0.0))

    def test_wraps_around(self):
        sw = square.FilterSwitcher(["a", "b"])
        sw.next(0); sw.next(0)
        self.assertEqual(sw.name, "a")


class SquareTests(unittest.TestCase):
    def test_two_hands_make_a_rectangle(self):
        poly = square.square_polygon(two_hands(), W, H)
        self.assertIsNotNone(poly)
        self.assertEqual(poly.shape, (4, 2))

    def test_one_open_hand_makes_a_square(self):
        hd = make_hand(("index",), thumb="out", scale=1.6)
        poly = square.square_polygon([hd], W, H)
        self.assertIsNotNone(poly)
        sides = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
        self.assertLess(max(sides) / min(sides), 1.05)

    def test_closed_thumb_makes_no_square(self):
        self.assertIsNone(square.square_polygon([make_hand(("index",), thumb="in")], W, H))

    def test_one_open_palm_does_not_make_a_square(self):
        self.assertIsNone(square.square_polygon([make_hand(thumb="out")], W, H))

    def test_two_fists_do_not_make_a_square(self):
        self.assertIsNone(square.square_polygon([make_hand((), thumb="in"), make_hand((), thumb="in")], W, H))

    def test_open_hands_with_thumb_and_index_out_do_make_a_square(self):
        hands = [make_hand(thumb="out", cx=0.25, cy=0.8, scale=1.4, mirror=True), make_hand(thumb="out", cx=0.75, cy=0.8, scale=1.4)]
        self.assertIsNotNone(square.square_polygon(hands, W, H))

    def test_no_hands_no_square(self):
        self.assertIsNone(square.square_polygon([], W, H))

    def test_filter_applies_inside_only(self):
        img = photo()
        sf = square.SquareFilter(square.FilterSwitcher(["invert"]))
        out = sf.render(img, two_hands(), 5.0)
        poly = sf.poly
        self.assertIsNotNone(poly)
        cx, cy = poly[:, 0].mean().astype(int), poly[:, 1].mean().astype(int)
        self.assertTrue((out[cy, cx] == 255 - img[cy, cx]).all())
        self.assertTrue((out[5:15, 5:15] == img[5:15, 5:15]).all())

    def test_square_is_held_still_while_thumb_touches_pinky(self):
        img = photo()
        sf = square.SquareFilter()
        hands = two_hands()
        sf.render(img, hands, 0.0)
        before = sf.poly.copy()
        hands[1] = make_hand(("index",), cx=0.78, cy=0.8, scale=1.4, thumb="pinky")
        sf.render(img, hands, 0.1)
        self.assertTrue((sf.poly == before).all())

    def test_gesture_names(self):
        self.assertEqual(square.gesture_name(make_hand(thumb="pinky")), "THUMB_PINKY")
        self.assertEqual(square.gesture_name(make_hand(thumb="out")), "OPEN_PALM")
        self.assertEqual(square.gesture_name(make_hand(("index",), thumb="out")), "POINT")
        self.assertEqual(square.gesture_name(make_hand((), thumb="in")), "FIST")


class EffectTests(unittest.TestCase):
    def test_every_filter_keeps_shape(self):
        img = photo()
        for n in effects.NAMES:
            out = effects.apply(img, n)
            self.assertEqual(out.shape, img.shape, n)
            self.assertEqual(out.dtype, np.uint8, n)

    def test_hud_draws_without_error(self):
        img = photo()
        sf = square.SquareFilter()
        out = sf.render(img, two_hands(), 1.0)
        before = out.copy()
        hud.draw_hud(out, two_hands(), (200, 100, 120, 140), sf, 30.0, 1.0)
        self.assertFalse((out == before).all())
        hud.draw_hud(out, [], None, sf, 30.0, 1.0)

    def test_face_box_finds_a_real_face(self):
        try:
            from skimage import data
        except ImportError:
            self.skipTest("scikit-image not installed")
        import cv2
        img = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (640, 480))
        fb = hud.FaceBox(every=1)
        if not fb.available:
            self.skipTest("this OpenCV build has no bundled face cascade")
        box = fb.update(img, 0.0)
        self.assertIsNotNone(box)
        x, y, w, h = box
        self.assertTrue(150 < x + w / 2 < 330 and y < 200)  # the astronaut's face


if __name__ == "__main__":
    unittest.main()
