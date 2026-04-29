import pandas as pd
import numpy as np

train_loss_list, total_valid_loss_list, obj_valid_loss_list, step = [0], [0], [0], [0]

def figure(path):
    df_data = {
        "step" : step,
        "train_loss" : train_loss_list,
        "total_valid_loss" : total_valid_loss_list,
        "obj_valid_loss" : obj_valid_loss_list
    }
    df = pd.DataFrame(df_data)
    df.set_index("step", inplace=True)
    df_transposed = df.T
    df_transposed.to_csv(path, sep = '\t')

def create_list(train_loss, total_valid_loss, obj_valid_loss, batch_number):
    step.append(batch_number)
    train_loss_list.append(train_loss)
    total_valid_loss_list.append(total_valid_loss)
    obj_valid_loss_list.append(obj_valid_loss)
