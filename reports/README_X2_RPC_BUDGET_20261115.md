# X2 RPC deadline: independent 20/25/30ms experiment

The user asked whether the20ms gate can be relaxed. It is a software acceptance deadline tied to the nominal50Hz period, not a physical maximum. The16:53 powered run successfully handed over but latched fault at RPC24.346ms while feedback age was about1.72ms.

Saved first600 samples:587 reply-ready; RPC above20/25/30ms was13/2/1 respectively, maximum40.029274ms. This is only a retrospective timing screen: post-fault samples cannot be reclassified as successful physical control.

Independent source generator validates original native controller SHA ca3f80b4b8a77613b243e1c903ba856475e1fb0b4618aeb0d3cbb7a46a18ee47, changes only two RPC comparisons and adds explicit budget range20..30ms/status metadata. Existing source, R6 binary/profile, model/gains/physics,50Hz requests/500Hz writer,80ms target and120ms feedback limits are preserved. Generates new20/25/30ms profiles and a new binary under an independent data root. No automatic powered adoption or MC changes.

Preparation: two local generator tests passed; shell syntax passed. Board build and read-only observation pending. Bounds: build180s/two compile workers, at least4GiB available/<80C; observation at most20s per candidate, no--control, no new training or BLE reconnection. Inspect existing writer absence before observation. Resources and actual results must be recorded separately. A30ms single-RPC deadline can still fault on40ms; this does not certify stable50Hz or propose indefinite holding of stale commands.

Next: build fresh candidate, evaluate live30ms in observe mode with the same current suit stream. Keep the20ms reference as default, do not replace R6 automatically. Recovery has already been independently confirmed atMC STAND_DEFAULT and original damping writer exited.
