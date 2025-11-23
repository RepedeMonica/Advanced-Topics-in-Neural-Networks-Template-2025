class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # min = loss, max = acc
        self.counter = 0
        self.best = float("inf") if mode == "min" else float("-inf")
        self.stop = False

    def __call__(self, value):
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
