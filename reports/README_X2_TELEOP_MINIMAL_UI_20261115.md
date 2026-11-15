# X2 teleoperation: remove auxiliary work from pose publication

2026-11-15. User authorized removing optional web work if it affects teleoperation.

Done: run_live_orin.sh now supplies --no-live-mesh --no-smpl-log. This bypasses live SMPL mesh/FK generation completely, while preserving the same pose matrices and native root translation for the HumanFrame bridge. Per-frame SMPL file output is disabled. The bridge still sends every available sample, but logs only every 30 requests. BLE status, 3+3 second calibration, 5 degree gate and reconnect remain available. The browser hides the 3D body and stops fetching mesh frames in this mode. Standalone run_orin.sh smpl retains optional visualization.

Verification: 16 Mac and 16 Orin preview tests passed; a separate bridge startup test passed. The new regression produces an SMPL publication with a mesh renderer that deliberately raises if called, and verifies no frame file is created. Shell/Python/JavaScript syntax passed. No model or native physics changes, no motor commands, no MC changes, no Bluetooth connection parameter mutation.

Deployment: no native controller or control_session was running. Old user-owned bridge13554 was stopped with SIGINT using sudo after explicit command validation; its parent cleaned up old web13469. New T1 parent108189 started the combination in data/analysis/X2_FGP_LIVE_MORPH_20261115_163912. API confirmed live_mesh_enabled=false, smpl_logging_enabled=false, sample_log_bytes=0; both garments receiving, error null, waiting_calibration. User needs to refresh and calibrate again. Do not claim measured live improvement before that.

Baseline: a saved bridge tail covering about10 seconds gave 272 outputs, 27.202Hz, input-proxy gap median34.435ms/p95 55.681ms/max71.201ms, none over100ms in that particular window; inference median32.835ms/p95 36.860ms. Other previous windows had >100ms gaps. This variable behavior does not establish continuous admission or end-to-end latency.

BLE audit: lspci showed Intel8086:2725; USB8087:0032 explicitly AX210 Bluetooth; hci0 B8:F7:75:26:10:FF served both garment LE links, upper handle2048 and lower2049. A bounded12-second passive btmon trace is stored only on robot at data/analysis/X2_BLE_CONTINUITY_20261115/passive_hci.txt. Observed upper144-byte complete HCI data and many lower28-byte fragments with4-byte tails. These are HCI-layer observations, NOT proof of ATT MTU, on-air PHY/data length, RF packet loss or radio contention. Monitor decoder unexpected-continuation messages must not be called garment checksum errors. No radio changes or reconnections were made by the passive monitor.

Next: after fresh wearer calibration, measure actual output cadence and >100ms gaps, then bounded native observe RPC/rejections with no writer. Investigate lower-link negotiation/connection timing separately; do not assert shared AX210 is saturated from differing average Hz alone. Keep120ms freshness/100ms interpolation and20ms RPC gate while comparing. Real powered control remains unverified.

References: Bluetooth LE connection parameter overview https://www.bluetooth.com/bluetooth-le-primer/ ; Intel Wi-Fi/Bluetooth coexistence https://www.intel.com/content/www/us/en/support/articles/000005821.html .
