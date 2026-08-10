import threading
import time
import unittest

from mt5.safe_api import mt5_session, serialized_mt5


class MT5SerializationTests(unittest.TestCase):
    def test_native_transactions_never_overlap(self):
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        @serialized_mt5
        def transaction():
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

        workers = [threading.Thread(target=transaction) for _ in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=1.0)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(maximum_active, 1)

    def test_session_gate_is_reentrant_for_native_workers(self):
        completed = []

        @serialized_mt5
        def nested_transaction():
            with mt5_session():
                completed.append(True)

        worker = threading.Thread(target=nested_transaction)
        worker.start()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(completed, [True])


if __name__ == "__main__":
    unittest.main()
