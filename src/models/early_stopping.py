import torch
class EarlyStopping:

    def __init__(self, save_path, patience=3, delta=1e-3, direction="minimize"):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_score = float('inf')
        self.direction = direction

        if self.direction == "maximize":
            self.best_score = -float('inf')

        self.save_path = save_path


    def early_stop(self, score, model):
        if self.direction == "minimize" : 
            if score < self.best_score:
                self.best_score = score
                self.counter = 0
                torch.save(model, self.save_path)
            elif score - self.best_score > self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        elif self.direction == "maximize" : 
            if score > self.best_score:
                self.best_score = score
                self.counter = 0
                torch.save(model, self.save_path)
            elif self.best_score - score > self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
                
        else :
            raise Exception("Unknown direction, must be minimize or maximize")
       
        return False
            
    def __str__(self):
        return  'EarlyStopping(\n' \
               f'    patience : {self.patience}\n' \
               f'    delta : {self.delta}\n'\
                ')'\
            

        