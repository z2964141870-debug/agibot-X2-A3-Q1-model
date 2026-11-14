"""Counter correctness matters more than nominal monitor execution."""
import unittest

from .monitor import cpu_delta, descendants, parse_cpu, parse_stat, parse_tegra, process_delta


class MonitorContract(unittest.TestCase):
    def test_stat_name_with_spaces_and_parentheses(self):
        fields = ["S", "9"]+["0"]*36
        fields[11], fields[12], fields[19] = "12", "8", "456"
        row = parse_stat("123 (a (worker)) "+" ".join(fields))
        self.assertEqual((row["pid"],row["name"],row["ppid"],row["ticks"],row["starttick"]),
                         (123,"a (worker)",9,20,456))

    def test_guest_ticks_not_counted_twice(self):
        a = parse_cpu("cpu0 20 0 10 40 5 0 0 0 8 0\nctxt 123")
        b = parse_cpu("cpu0 40 0 10 60 5 0 0 0 16 0")
        self.assertEqual(cpu_delta(a,b),{"cpu0":50})

    def test_counter_reset_not_reported_as_usage(self):
        self.assertEqual(cpu_delta({"cpu":(100,50)},{"cpu":(20,10)}),{})

    def test_pid_reuse_does_not_inherit_cpu(self):
        old = {4:dict(pid=4,starttick=1,ticks=10)}
        new = {4:dict(pid=4,starttick=2,ticks=100)}
        self.assertEqual(process_delta(old,new,1,100),[])

    def test_multicore_process_not_clamped_to_100(self):
        old = {4:dict(pid=4,starttick=1,ticks=10)}
        new = {4:dict(pid=4,starttick=1,ticks=260)}
        self.assertEqual(process_delta(old,new,1,100)[0]["cpu_percent_one_core"],250)

    def test_descendant_requires_original_root(self):
        rows = {1:dict(ppid=0,starttick=8),2:dict(ppid=1,starttick=9),3:dict(ppid=2,starttick=10)}
        self.assertEqual(descendants(rows,1,8),{1,2,3})
        self.assertEqual(descendants(rows,1,7),set())

    def test_tegra_and_missing_fields(self):
        row = parse_tegra("RAM 2745/15656MB GR3D_FREQ 13%@306 EMC_FREQ 5%@3199 cpu@55.718C gpu@51C")
        self.assertEqual(row["gpu_busy_percent"],13)
        self.assertEqual(row["ram_used_mib"],2745)
        self.assertEqual(row["temperature_c"]["cpu"],55.718)
        self.assertIsNone(parse_tegra("unrecognized status")["gpu_busy_percent"])


if __name__ == "__main__":
    unittest.main()
