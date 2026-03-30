import numpy as np
import random
import multiprocessing
from solver import montecarlo_stocastic_accn
import pickle



def run_montecarlo_simulations(num_simulations=100):
    # Define constants (replace with actual values as needed)
    length = 50
    modulus = 200e9
    linearMass = 500
    modalDampingRatio = 0.005
    numbers = 3
    pedmass = 70
    peddamp = 0.3
    pedBodyF = 2
    pedvelocity = 1.25
    numped = 1
    hht = 0.01
    x_interested = length/2  # Adjust according to your needs

    # Initialize the final response matrix
    acceleration_responses = np.zeros((num_simulations, len(x_interested)))

    # Use multiprocessing to run the simulations in parallel
    with multiprocessing.Pool(processes=4) as pool:  # Adjust 'processes' based on your CPU
        # Prepare arguments for each simulation
        tasks = [(length, modulus, linearMass, modalDampingRatio, numbers, pedmass, peddamp, pedBodyF, pedvelocity, numped, hht, x_interested, i) for i in range(num_simulations)]

        # Run simulations in parallel
        results = pool.starmap(montecarlo_stocastic_accn, tasks)

    # Collect the results
    for i, result in enumerate(results):
        acceleration_responses[i, :] = result

    return acceleration_responses

if __name__ == "__main__":
    # Run 100 simulations
    num_simulations = 10
    acceleration_responses = run_montecarlo_simulations(num_simulations)
   
    # Save the matrix using pickle
    with open('acceleration_responses.pkl', 'wb') as f:
        pickle.dump(acceleration_responses, f)

    # Output or process the results
    print(acceleration_responses)

