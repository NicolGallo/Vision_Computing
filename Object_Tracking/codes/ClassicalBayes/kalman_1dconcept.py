#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def kalman_filter_demo():

    # Define the state transition matrix
    dt = 0.1  # time step
    A = np.array([[1, dt], [0, 1]])

    # Define the measurement matrix
    C = np.array([[1, 0]])

    # Define the process and measurement noise covariance matrices
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[10]])

    # Define the initial state estimate and covariance matrix
    x0 = np.array([0, 0])
    P0 = np.array([[1, 0], [0, 1]])

    # Generate some noisy sensor data
    t = np.arange(0, 10, dt)
    n = len(t)
    x = np.zeros((n, 2))
    y = np.zeros((n, 1))
    for i in range(n):
        if i == 0:
            x[i] = x0
        else:
            x[i] = np.dot(A, x[i-1]) + np.random.multivariate_normal([0, 0], Q)
        y[i] = np.dot(C, x[i]) + np.random.normal(0, np.sqrt(R))

    # Run the Kalman filter
    xhat = np.zeros((n, 2))
    Phat = np.zeros((n, 2, 2))
    xhat[0] = x0
    Phat[0] = P0
    for i in range(1, n):
        xhat[i] = np.dot(A, xhat[i-1])
        Phat[i] = np.dot(np.dot(A, Phat[i-1]), A.T) + Q
        K = np.dot(np.dot(Phat[i], C.T), np.linalg.inv(np.dot(np.dot(C, Phat[i]), C.T) + R))
        xhat[i] = xhat[i] + np.dot(K, y[i] - np.dot(C, xhat[i]))
        Phat[i] = np.dot(np.eye(2) - np.dot(K, C), Phat[i])

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x[:, 0], 'b-', label='true')
    plt.plot(t, y[:, 0], 'r.', label='measured')
    plt.plot(t, xhat[:, 0], 'g-', label='estimated')
    plt.legend()
    plt.ylabel('Position (m)')
    plt.title('Kalman filter example')
    plt.subplot(2, 1, 2)
    plt.plot(t, x[:, 1], 'b-', label='true')
    plt.plot(t, xhat[:, 1], 'g-', label='estimated')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.show()