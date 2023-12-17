import torch
class EarlyStopping:

    def __init__(self, save_path, patience=3, delta=1e-3,):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.min_score = float('inf')

        self.save_path = save_path

    def early_stop(self, score, model):
        if score < self.min_score:
            self.min_score = score
            self.counter = 0
            torch.save(model, self.save_path)
        elif score - self.min_score > self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
       
        return False
            
    def __str__(self):
        return  'EarlyStopping(\n' \
               f'    patience : {self.patience}\n' \
               f'    delta : {self.delta}\n'\
                ')'\
            

        