#!/usr/bin/env python3

import pandas as pd
from utils import get_model

if __name__ == '__main__':
    df = pd.read_csv('data.csv')

    x = df['km']
    y = df['price']
    m = len(y)

    try:
        intercept, slope = get_model()

    except FileNotFoundError:
        print('Please run train.py first.')
        exit(1)

    except Exception:
        print('Invalid model file. Please run train.py again.')
        exit(1)

    y_pred = intercept + (slope * x)

    mse = (1/m) * sum((y_pred - y) ** 2)
    mae = (1/m) * sum(abs(y_pred - y))

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
