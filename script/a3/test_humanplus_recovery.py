"""Pure metadata controls for the explicitly scoped historical Xid resolution."""

import copy
import unittest

from script.a3.humanplus_recovered_health import EXPECTED_MESSAGE, validate_resolution
from script.a3.humanplus_health import TelemetryError


class ResolutionControls(unittest.TestCase):
    def pair(self):
        event = dict(cursor='retained-cursor', message=EXPECTED_MESSAGE)
        record = dict(status='EXPLICIT_APPLICATION_RESTART_AUTHORIZED',
            authorization='修，修好了继续优化', boot_id='retained-boot', resolved_events=[event])
        evidence = dict(boot_id='retained-boot', journal=dict(fatal=[copy.deepcopy(event)]), new_oom_events=[])
        return record, evidence

    def test_exact_explicit_history_only(self):
        validate_resolution(*self.pair())

    def test_new_missing_changed_events_and_oom_rejected(self):
        record, before = self.pair()
        for events in ([], before['journal']['fatal'] * 2,
                       [dict(cursor='different', message=EXPECTED_MESSAGE)],
                       [dict(cursor='retained-cursor', message=EXPECTED_MESSAGE.replace('43,', '79,'))]):
            with self.subTest(events=events), self.assertRaises(TelemetryError):
                validate_resolution(record, dict(before, journal=dict(fatal=events)))
        with self.assertRaises(TelemetryError):
            validate_resolution(record, dict(before, new_oom_events=[dict(increase=1)]))

    def test_boot_and_authorization_cannot_change(self):
        record, evidence = self.pair()
        with self.assertRaises(TelemetryError):
            validate_resolution(record, dict(evidence, boot_id='new-boot'))
        with self.assertRaises(TelemetryError):
            validate_resolution(dict(record, authorization='automatic'), evidence)


if __name__ == '__main__':
    unittest.main()
