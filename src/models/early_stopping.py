class EarlyStopping:

    def __init__(self, patience=3, delta=1e-3):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.min_loss = float('inf')


    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
            
        return False
            

        