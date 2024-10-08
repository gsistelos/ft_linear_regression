#!/usr/bin/env python3

from utils import get_model


def get_input():
    x = int(input("Mileage (km): "))
    if x < 0:
        raise ValueError

    return x


if __name__ == "__main__":
    try:
        intercept, slope = get_model()

    except FileNotFoundError:
        print('Please run train.py first.')
        exit(1)

    except Exception:
        print('Invalid model file. Please run train.py again.')
        exit(1)

    try:
        x = get_input()

    except ValueError:
        print("Please enter a positive number.")
        exit(1)

    pred = intercept + (slope * x)

    print(f"The estimated price is {pred:.2f}€")
