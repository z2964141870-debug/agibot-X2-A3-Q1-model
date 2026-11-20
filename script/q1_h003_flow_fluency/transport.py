"""5 ms simulated transport with sequence numbers and acknowledgements at application."""

import numpy as np


class Transport:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.micro = 0
        self.seq = -1
        self.pending = []
        self.applied = None
        self.ack = None
        self.ages = []
        self.command = None
        self.previous = None
        self.delay = 2
        self.phase = self.rng.integers(0, 31)

    def observation(self, target):
        valid = self.ack is not None and self.micro - self.ack[0] <= 12
        age = self.ages[-1] if valid else 0.0
        history = [
            age,
            np.mean(self.ages[-4:]) if valid else 0.0,
            (self.micro - self.ack[0]) / 4 if valid else 0.0,
            float(valid),
        ]
        prev = self.previous if self.previous is not None else target
        confirmed = self.ack[2] if valid else prev
        return np.r_[history, prev - target, confirmed - prev].astype(np.float32), valid, age

    def submit(self, command, initial):
        if self.command is None:
            self.command = initial.copy()
        t = self.micro // 4
        if self.mode == "train_var":
            if t % 5 == 0:
                self.delay = int(self.rng.integers(0, 7))
        elif self.mode == "unseen":
            self.delay = (0, 2, 5, 6, 4, 1, 3)[((t + self.phase) // 9) % 7]
        else:
            self.delay = int(self.mode) // 5
        self.events = []
        self.seq += 1
        self.pending.append((self.micro + self.delay, self.seq, self.micro, command.copy()))
        self.previous = command.copy()
        return self.delay

    def next(self):
        ready = [p for p in self.pending if p[0] <= self.micro]
        self.pending = [p for p in self.pending if p[0] > self.micro]
        if ready:
            _, seq, sent, value = max(ready, key=lambda p: p[1])
            if self.applied is None or seq > self.applied:
                self.command = value
                self.applied = seq
                # Fixed time windows of dropped acknowledgements on variable links.
                if self.mode not in ("train_var", "unseen") or (self.micro // 4) % 83 not in range(
                    40, 48
                ):
                    self.ack = (self.micro, seq, value)
                    self.events.append((seq, self.micro))
                    self.ages.append((self.micro - sent) / 4)
        self.micro += 1
        return self.command
