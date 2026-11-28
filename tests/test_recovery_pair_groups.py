"""The clip-group readings (studies/recovery_pair/groups.py) on hand-written reference rows, records and readings:
low rows and spans, get-up windows in a trial's controls, the windows a trial enters and ends in, the groups of the
clips and the pooled counts. The library is not built here (reference_rows needs the LAFAN1 CSVs)."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "studies", "recovery_pair"))
import groups as G  # noqa: E402


class TestRows(unittest.TestCase):
    def test_low_rows_by_height_or_up_z(self):
        self.assertEqual(G.low_rows([0.49, 0.5, 0.8, 0.8], [0.9, 0.9, 0.49, 0.5]).tolist(), [True, False, True, False])

    def test_spans_of_at_least_25_rows(self):
        low = np.zeros(200, bool)
        low[10:34] = True  # 24 rows: not a span
        low[50:75] = True  # 25 rows
        low[150:200] = True  # to the end
        self.assertEqual(G.low_spans(low), [(50, 74), (150, 199)])
        self.assertEqual(G.low_spans(np.ones(30, bool)), [(0, 29)])
        self.assertEqual(G.low_spans(np.zeros(30, bool)), [])

    def test_windows_in_controls_clipped_at_the_clip(self):
        first, last = 1000, 1199  # 200 rows, controls 0 .. 198
        low = np.zeros(200, bool)
        low[0:30] = True  # span rows 1000..1029, window rows 979..1079 -> controls 0 .. 78
        low[100:130] = True  # span last row 1129, window rows 1079..1179 -> controls 78 .. 178
        low[170:200] = True  # span last row 1199, window rows 1149..1249 -> controls 148 .. 198
        w = G.get_up_windows(low, first, last)
        self.assertEqual([x["window_controls"] for x in w], [[0, 78], [78, 178], [148, 198]])
        self.assertEqual([x["span_last_row"] for x in w], [1029, 1129, 1199])
        self.assertEqual(w[0]["span_first_row"], 1000)


class TestEnds(unittest.TestCase):
    W = [dict(window_controls=[10, 20]), dict(window_controls=[15, 30]), dict(window_controls=[50, 60])]

    def test_entered_and_ended(self):
        self.assertEqual(G.window_ends(None, self.W), (3, 0))
        self.assertEqual(G.window_ends(9, self.W), (0, 0))  # ended before every window
        self.assertEqual(G.window_ends(10, self.W), (1, 1))  # first control of a window
        self.assertEqual(G.window_ends(17, self.W), (2, 2))  # overlapping windows count separately
        self.assertEqual(G.window_ends(30, self.W), (2, 1))  # last control of the second
        self.assertEqual(G.window_ends(40, self.W), (2, 0))
        self.assertEqual(G.window_ends(60, self.W), (3, 1))

    def test_first_end(self):
        self.assertIsNone(G.first_end([False] * 4))
        self.assertEqual(G.first_end([False, True, True]), 1)


def readings_of(clips, errors, completed):
    rc = [dict(clip=n, aligned_body_error_m=e, completed=c) for n, e, c in zip(clips, errors, completed)]
    return dict(save=dict(checkpoint="x"), R=dict(numerator=90), J=dict(numerator=88), F=dict(numerator=1),
                first_rows=dict(C=dict(completed=sum(completed)), E_m=dict(value=None), clips=rc))


class TestSet(unittest.TestCase):
    def setUp(self):
        # four clips of 100 rows: hidden h (one span), excluded x (one span), commanded c1 (one span), c2 (none)
        self.names = ["h", "x", "c1", "c2"]
        self.clips = [dict(clip=n, first_row=100 * i, last_row=100 * i + 99, controls=99) for i, n in enumerate(self.names)]
        self.scene = dict(clips=self.clips, settings=dict(hidden_clips=["h"], command_excluded_clips=["h", "x"]))
        height = np.full(400, 0.8)
        up = np.ones(400)
        height[20:60] = 0.2  # h: span rows 20..59 -> window controls max(9 - 1, 0) .. 59 + 50 - 1 = 8 .. 98
        up[130:160] = 0.1  # x: span rows 130..159 -> controls 8 .. 98
        height[200:230] = 0.3  # c1: span rows 200..229 -> controls 0 .. 78
        self.reference = (height, up)

    def test_groups_ends_and_errors(self):
        done = np.zeros((150, 4), bool)
        done[40, 0] = True  # h ends inside its window
        done[5, 1] = True  # x ends before its window: not entered
        done[120, 2] = True  # c1 ends after its window (in the hold): entered, not ended in
        readings = readings_of(self.names, [0.06, 0.05, 0.03, 0.02], [False, False, True, True])
        out, windows = G.read_set(self.scene, self.reference, readings, done, [0, 100, 200, 300])
        self.assertEqual(windows["h"][0]["window_controls"], [8, 98])
        self.assertEqual(windows["c1"][0]["window_controls"], [0, 78])
        self.assertEqual(windows["c2"], [])
        self.assertEqual(out["ends_per_reference_get_up_window"],
                         {"hidden_clips": dict(ended_in_window=1, windows_entered=1),
                          "excluded_clips_not_hidden": dict(ended_in_window=0, windows_entered=0),
                          "commanded_clips_with_get_ups": dict(ended_in_window=0, windows_entered=1)})
        self.assertEqual(out["commanded_clips"], dict(clips=2, clips_completed=2, tracking_error_m=0.025))
        self.assertEqual(out["excluded_clips"]["clips_completed"], 0)
        self.assertAlmostEqual(out["excluded_clips"]["tracking_error_m"], 0.055)
        self.assertEqual([t["end_control"] for t in out["trials"]], [40, 5, 120, None])
        self.assertEqual(out["trials"][3]["group"], None)
        pooled = G.pooled([out, out])
        self.assertEqual(pooled["hidden_clips"], dict(ended_in_window=2, windows_entered=2))

    def test_refusals_and_a_missing_error(self):
        readings = readings_of(self.names, [0.06, None, 0.03, 0.02], [False, False, True, True])
        out, _ = G.read_set(self.scene, self.reference, readings, np.zeros((10, 4), bool), [0, 100, 200, 300])
        self.assertIsNone(out["excluded_clips"]["tracking_error_m"])
        with self.assertRaises(ValueError):
            G.read_set(self.scene, self.reference, readings, np.zeros((10, 4), bool), [0, 100, 200, 301])
        with self.assertRaises(ValueError):
            G.read_set(self.scene, self.reference, readings_of(self.names[::-1], [0.1] * 4, [True] * 4),
                       np.zeros((10, 4), bool), [0, 100, 200, 300])

    def test_compare_reference(self):
        C, T, B = 3, 2, 2
        rows = np.array([[5, 105], [6, 106], [7, 107]])
        pos = np.zeros((C, T, B, 3), np.float32)
        quat = np.zeros((C, T, B, 4), np.float32)
        quat[..., 0] = 1
        pos[..., 0, 2] = self.reference[0][rows]
        rec = dict(available=np.ones((C, T), bool), command_row_after=rows, command_body_pos_w=pos,
                   command_body_quat_w=quat)
        c = G.compare_reference(self.reference, rec, 0)
        self.assertLess(c["height_max_abs_m"], 1e-6)
        self.assertEqual(c["low_flag_mismatches"], 0)
        self.assertEqual(c["controls"], 6)


if __name__ == "__main__":
    unittest.main()
