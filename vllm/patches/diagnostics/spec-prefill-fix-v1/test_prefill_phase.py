"""Exercise the real installed scheduler without allocating model/GPU memory.

Run unchanged against baseline (race cases must fail), then patched image.
"""
import pickle
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput


def scheduler(length, width):
    obj = AsyncScheduler.__new__(AsyncScheduler)
    req = SimpleNamespace(
        num_tokens=length, num_prompt_tokens=length,
        num_computed_tokens=length - 16384, is_prefill_chunk=True,
        num_output_placeholders=0, num_output_tokens=0,
        use_structured_output=False, max_tokens=192, spec_token_ids=[],
        is_finished=lambda: False, request_id='r',
    )
    obj.requests = {'r': req}
    obj.finished_req_ids = set()
    obj.num_sampled_tokens_per_step = 1
    obj.num_spec_tokens = width
    obj._spec_token_placeholders = [-1] * width
    obj._v50_ph_limit = 64
    obj._v51_f8_mode = 'v52'
    obj._v52_f8_grace = 60
    obj._v52_parked = {}
    obj._v51_f8_logged = set()
    obj._v52m_strikes_n = 3
    obj._v52m_watch = {}
    obj._v52b_strikes = {}
    obj._v52b_logged = set()
    obj._v52_term_pending = {}
    obj._v52m_strike_out = Mock()
    return obj, req


def step(tokens=8192, width=None):
    out = SchedulerOutput.make_empty()
    if tokens:
        out.num_scheduled_tokens = {'r': tokens}
        out.total_num_scheduled_tokens = tokens
    if width:
        out.scheduled_spec_decode_tokens = {'r': [-1] * width}
    return out


def resolve(obj, out, ids):
    # Only downstream token bookkeeping is stubbed. The guard under test
    # and both actual _update_after_schedule methods run unchanged.
    result = SimpleNamespace(sampled_token_ids=[ids], req_id_to_index={'r': 0})
    with patch.object(Scheduler, 'update_from_output', return_value={}):
        obj.update_from_output(out, result)


class PrefillPhaseTests(unittest.TestCase):
    def test_delayed_prefill_never_arms_decode_watchdog(self):
        for length in (65536, 131072, 261888):
            for width in (0, 1, 4, 7):
                with self.subTest(length=length, width=width):
                    obj, req = scheduler(length, width)
                    early, final = step(), step()
                    obj._update_after_schedule(early)
                    self.assertTrue(req.is_prefill_chunk)
                    obj._update_after_schedule(final)
                    self.assertFalse(req.is_prefill_chunk)
                    # The first result is still pending when the final
                    # chunk advances the mutable request to decode.
                    with patch('vllm.v1.core.sched.async_scheduler.time.monotonic', return_value=100):
                        resolve(obj, early, [])
                    with patch('vllm.v1.core.sched.async_scheduler.time.monotonic', return_value=106):
                        obj._update_after_schedule(step(0))
                    self.assertEqual(obj._v52b_strikes, {})
                    self.assertEqual(obj._v52m_watch, {})
                    obj._v52m_strike_out.assert_not_called()

    def test_snapshot_is_per_step_and_survives_transport(self):
        obj, req = scheduler(65536, 7)
        early, final = step(), step()
        obj._update_after_schedule(early)
        obj._update_after_schedule(final)
        self.assertEqual(early.scheduled_prefill_chunk_req_ids, frozenset({'r'}))
        self.assertEqual(final.scheduled_prefill_chunk_req_ids, frozenset())
        restored = pickle.loads(pickle.dumps(early))
        self.assertEqual(restored.scheduled_prefill_chunk_req_ids, frozenset({'r'}))

    def test_genuine_empty_decode_still_reaches_guard(self):
        obj, req = scheduler(65536, 4)
        req.num_computed_tokens = req.num_tokens
        req.is_prefill_chunk = False
        for _ in range(3):
            out = step(5, 4)
            obj._update_after_schedule(out)
            resolve(obj, out, [])
        obj._v52m_strike_out.assert_called_with('r', 'strike-out')

    def test_valid_final_prefill_and_decode_clear_strikes(self):
        obj, req = scheduler(65536, 7)
        obj._update_after_schedule(step())
        out = step()
        obj._update_after_schedule(out)
        obj._v52b_strikes['r'] = 1
        obj._v52m_watch['r'] = 100
        resolve(obj, out, [17])
        self.assertEqual(obj._v52b_strikes, {})
        self.assertEqual(obj._v52m_watch, {})

    def test_unknown_step_phase_cannot_justify_abort(self):
        obj, req = scheduler(65536, 7)
        req.is_prefill_chunk = False
        resolve(obj, step(), [])  # No scheduling snapshot: unknown origin.
        self.assertEqual(obj._v52b_strikes, {})

    def test_cancelled_and_grammar_rows_are_not_struck(self):
        for grammar, finished in ((True, False), (False, True)):
            obj, req = scheduler(65536, 4)
            req.num_computed_tokens = req.num_tokens
            out = step(5, 4)
            obj._update_after_schedule(out)
            req.use_structured_output = grammar
            req.is_finished = lambda: finished
            resolve(obj, out, [])
            self.assertEqual(obj._v52b_strikes, {})

    def test_mixed_batch_uses_each_rows_scheduled_phase(self):
        obj, req = scheduler(131072, 7)
        decoder = copy.copy(req)
        decoder.request_id = 'd'
        decoder.num_computed_tokens = decoder.num_tokens
        decoder.is_prefill_chunk = False
        obj.requests['d'] = decoder
        early = step()
        early.num_scheduled_tokens['d'] = 8
        early.scheduled_spec_decode_tokens['d'] = [-1] * 7
        early.total_num_scheduled_tokens += 8
        obj._update_after_schedule(early)
        obj._update_after_schedule(step())
        result = SimpleNamespace(sampled_token_ids=[[], []],
                                 req_id_to_index={'r': 0, 'd': 1})
        with patch.object(Scheduler, 'update_from_output', return_value={}):
            obj.update_from_output(early, result)
        self.assertEqual(obj._v52b_strikes, {'d': 1})
        self.assertEqual(set(obj._v52m_watch), {'d'})

    def test_snapshot_does_not_change_scheduled_width_or_placeholders(self):
        for width in (0, 1, 4, 7):
            obj, req = scheduler(65536, width)
            early, final = step(), step()
            obj._update_after_schedule(early)
            self.assertEqual(req.num_output_placeholders, 0)
            self.assertEqual(req.spec_token_ids, [])
            obj._update_after_schedule(final)
            self.assertEqual(req.num_output_placeholders, 1)
            self.assertEqual(req.spec_token_ids, [-1] * width)
            decode = step(1 + width, width)
            obj._update_after_schedule(decode)
            self.assertEqual(req.num_output_placeholders, 2 + width)
            self.assertEqual(decode.num_scheduled_tokens, {'r': 1 + width})


if __name__ == '__main__':
    unittest.main(verbosity=2)
