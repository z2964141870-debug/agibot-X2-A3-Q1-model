"""Protocol fault controls; synthetic bytes are not mocap/training evidence."""
import struct
import unittest

from script.headband.nori_offline import decode_bmp, decode_payload, forward_us


def packet(axes=(0, 8192, 0, 3276, 0, 0)):
    return bytes.fromhex('1011000000000000') + struct.pack('>II', 100, 200) + struct.pack('>I6h', 150, *axes)


def bmp(payload, top_down=False, offset=70):
    width, height = 155, len(payload) // 2 * 8 + 4
    stride = ((width * 24 + 31) // 32) * 4
    pixels = bytearray([255] * (stride * height))
    for n in range(len(payload) // 2):
        y = 2 + 8 * n
        row = y if top_down else height - 1 - y
        for bit in range(16):
            value = 128 if payload[n * 2 + bit // 8] & (1 << (bit % 8)) else 0
            for x in range(8 + bit * 8, 16 + bit * 8):
                pixels[row * stride + x * 3:row * stride + x * 3 + 3] = bytes([value] * 3)
    header = bytearray(offset)
    struct.pack_into('<2sIHHI', header, 0, b'BM', offset + len(pixels), 0, 0, offset)
    struct.pack_into('<IiiHHII', header, 14, 40, width, -height if top_down else height, 1, 24, 0, len(pixels))
    return bytes(header + pixels)


class ProtocolTests(unittest.TestCase):
    def test_scaling_and_endianness(self):
        r = decode_payload(packet(), accel_range_g=4, gyro_range_dps=1000)
        self.assertEqual(r['exposure_start_us32'], 100)
        self.assertEqual(r['imu_samples'][0]['timestamp_us32'], 150)
        self.assertAlmostEqual(r['imu_samples'][0]['accel_m_s2'][1], 9.80665)
        self.assertAlmostEqual(r['imu_samples'][0]['gyro_rad_s'][0], 1.744903146)

    def test_no_implicit_physical_units(self):
        self.assertNotIn('accel_m_s2', decode_payload(packet())['imu_samples'][0])

    def test_sentinel_is_invalid_not_zero(self):
        for axes in [(-1, -1, -1, 0, 0, 0), (0, 0, 8192, -32768, 0, 0)]:
            r = decode_payload(packet(axes), accel_range_g=4)
            self.assertFalse(r['imu_samples'][0]['valid'])
            self.assertNotIn('accel_m_s2', r['imu_samples'][0])

    def test_truncated_payload(self):
        with self.assertRaises(ValueError): decode_payload(packet()[:-1])

    def test_unsupported_protocol(self):
        p = bytearray(packet()); p[0] = 0x20
        with self.assertRaises(ValueError): decode_payload(p)

    def test_excessive_group_count(self):
        p = bytearray(packet()); p[1] = 0x1f
        with self.assertRaises(ValueError): decode_payload(p)

    def test_unknown_populated_sensor(self):
        p = bytearray(packet()); p[1] = 0x21
        with self.assertRaises(ValueError): decode_payload(p)

    def test_stride_offset_orientation(self):
        for down in (False, True):
            r = decode_bmp(bmp(packet(), down))
            self.assertEqual(r['payload_hex'], packet().hex())
            self.assertEqual(r['bmp']['stride'], 468)
            self.assertEqual(r['bmp']['offset'], 70)

    def test_truncated_bmp(self):
        with self.assertRaises(ValueError): decode_bmp(bmp(packet())[:-1])

    def test_missing_prefix(self):
        d = bytearray(bmp(packet(), True)); stride = 468
        d[70 + 2 * stride] = 0
        with self.assertRaises(ValueError): decode_bmp(d)

    def test_rollover(self):
        self.assertEqual(forward_us(2**32 - 10, 15, 100), 25)

    def test_backward_duplicate_and_long_gap(self):
        for a, b in [(100, 99), (100, 100), (100, 1000)]:
            with self.assertRaises(ValueError): forward_us(a, b, 100)

    def test_demo_golden_header(self):
        # Independently transcribed from guide p2/p3 screenshot, not an encoder.
        from script.headband.nori_offline import parse_header
        r = parse_header(bytes.fromhex('101b02000000000001ad096901ad26af'))
        self.assertEqual(r['slots'], [(1, 11), (2, 0), (0, 0), (0, 0), (0, 0)])
        self.assertEqual(r['exposure_end_us32'] - r['exposure_start_us32'], 7494)


if __name__ == '__main__': unittest.main()
