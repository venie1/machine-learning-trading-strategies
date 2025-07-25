		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
	  	   		 	   		  		  		    	 		 		   		 		  
    return "pvenieris3"  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
		  	   		 	   		  		  		    	 		 		   		 		  
    return 903960654 
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    		  	   		 	   		  		  		    	 		 		   		 		  
    result = False  		  	   		 	   		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	   		  		  		    	 		 		   		 		  
        result = True  		  	   		 	   		  		  		    	 		 		   		 		  
    return result  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def exp1(win_prob,spins,episodes):
    winnings = np.zeros((episodes, spins + 1))  # array holding winnings
    for episode in range(episodes):

        winnings[episode, 0] = 0
        episode_winnings = 0
        bet_amount = 1
        for spin in range(1, spins + 1):
            if episode_winnings>= 80:
                # if we reach 80 , the next bets are always 80
                winnings[episode, spin:] = 80
                break
            won =get_spin_result(win_prob) # Determine if the spin is a win based on the win probability
            if won:
                episode_winnings += bet_amount
                bet_amount = 1  # Reset the bet amount after a win
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2  # Double the bet amount after a loss

            winnings[episode, spin] = episode_winnings  # Record the winnings after each spin


    return winnings




def exp2(num_episodes, num_spins, win_prob, initial_bankroll, target_winnings):
    winnings = np.zeros((num_episodes, num_spins + 1))  # +1 for the initial zero winnings

    for episode in range(num_episodes):
        bankroll = initial_bankroll
        current_winnings = 0
        bet_amount = 1

        for spin in range(1, num_spins + 1):
            if current_winnings >= target_winnings:
                # If winnings reach or equals $80, fill the rest with the last value
                winnings[episode, spin:] = current_winnings
                break
            if current_winnings <= -initial_bankroll:
                # If winnings reach -$256, fill the rest with -$256
                winnings[episode, spin:] = -initial_bankroll
                break

            # Ensure that we cannot bet more than what we have left
            if bet_amount > bankroll:
                bet_amount = bankroll

            # Simulate the spin
            won = get_spin_result(win_prob)

            if won:
                current_winnings += bet_amount
                bankroll += bet_amount
                bet_amount = 1  # Reset bet amount after a win
            else:
                current_winnings -= bet_amount
                bankroll -= bet_amount
                bet_amount *= 2  # Double the bet amount after a loss


            winnings[episode, spin] = current_winnings

    return winnings


def simulate(win_prob, initial_bankroll, target_winnings, max_bets=1000):
    bankroll = initial_bankroll
    winnings = 0
    bet_amount = 1

    for _ in range(max_bets):
        if winnings >= target_winnings:
            return True  # Player won $80 or more
        if bankroll <= 0:
            return False  # Player go bankrupt

        if bet_amount > bankroll:
            bet_amount = bankroll  # Bet only what you have left

        if get_spin_result(win_prob):
            winnings += bet_amount
            bankroll += bet_amount
            bet_amount = 1  # Reset bet after win
        else:
            winnings -= bet_amount
            bankroll -= bet_amount
            bet_amount *= 2  # Double the bet after loss

    return winnings >= target_winnings
def estimate_probability(win_prob, initial_bankroll, target_winnings, num_simulations=10000):
    successes = 0

    for _ in range(num_simulations):
        if simulate(win_prob, initial_bankroll, target_winnings):
            successes += 1

    return successes / num_simulations
def test_code():
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """



    np.random.seed(gtid())  # do this only once
    episodes = 10
    spins = 1000
    win_prob = 18 / 38
    win= exp1(win_prob,spins,episodes)
    plt.figure(1)
    plt.axis([0, 300, -256, 100])  # Set axis limits for X (0-300) and Y (-256 to +100)
    for episode in range(episodes):
        plt.plot(win[episode, :301])

    plt.legend(["Episode " + str(a) for a in range(1, episodes + 1)])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("winnings per episode")
    plt.savefig("./images/Figure1.png")  # Save plot
    plt.close()

    episodes = 1000
    win = exp1(win_prob, spins, episodes)
    mean_winnings = np.mean(win, axis=0)
    std_dev_winnings = np.std(win,axis=0)
    mean_plus = mean_winnings + std_dev_winnings
    mean_minus = mean_winnings - std_dev_winnings

    plt.axis([0, 300, -256, 100])  # Set axis limits

    # Plot mean winnings
    plt.plot(mean_winnings[:301], label='Mean Winnings', color='blue')

    # Plot mean + std deviation
    plt.plot(mean_plus, label='Mean + 1 SD', color='green')

    # Plot mean - std deviation
    plt.plot(mean_minus, label='Mean - 1 SD', color='red')

    # Add labels and title
    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.title('Mean Winnings with Standard Deviation Over 1000 Episodes')
    plt.legend(loc='lower left')
    plt.savefig("./images/Figure2.png")  # Save plot
    plt.close()


    median_winnings = np.median(win, axis=0)
    std_dev_winnings = np.std(win, axis=0)
    med_plus=median_winnings+std_dev_winnings
    med_minus=median_winnings-std_dev_winnings


    # Plotting the results

    plt.axis([0, 300, -256, 100])  # Set axis limits

    # Plot median winnings
    plt.plot(median_winnings[:301], label='Median Winnings', color='blue')

    # Plot median + std deviation
    plt.plot(med_plus, label='Median + 1 SD', color='green')

    # Plot median - std deviation
    plt.plot(med_minus, label='Median - 1 SD', color='red')


    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.title('Median Winnings with Standard Deviation Over 1000 Episodes')
    plt.legend(loc='lower right')
    plt.savefig("./images/Figure3.png")  # Save plot
    plt.close()

    num_episodes = 1000
    num_spins = 1000
    win_prob = 18 / 38
    initial_bankroll = 256
    target_winnings = 80
    winnings = exp2(num_episodes, num_spins, win_prob, initial_bankroll, target_winnings)
    # Calculate mean and standard deviation for each spin
    mean_winnings = np.mean(winnings, axis=0)
    std_dev_winnings = np.std(winnings, axis=0)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.axis([0, 300, -256, 100])  # Set axis limits

    # Plot mean winnings
    plt.plot(mean_winnings[:301], label='Mean Winnings', color='blue')

    # Plot mean + std deviation
    plt.plot(mean_winnings[:301] + std_dev_winnings[:301], '--', label='Mean + 1 SD', color='green')

    # Plot mean - std deviation
    plt.plot(mean_winnings[:301] - std_dev_winnings[:301], '--', label='Mean - 1 SD', color='red')


    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.title('Mean Winnings with Standard Deviation Over 1000 Episodes with $256 Bankroll')
    plt.legend(loc='upper left')
    plt.savefig("./images/Figure4.png")  # Save plot
    plt.close()

    # Calculate median and standard deviation for each spin
    median_winnings = np.median(winnings, axis=0)
    std_dev_winnings = np.std(winnings, axis=0)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.axis([0, 300, -256, 100])  # Set axis limits

    # Plot median winnings
    plt.plot(median_winnings[:301], label='Median Winnings', color='blue')

    # Plot median + std deviation
    plt.plot(median_winnings[:301] + std_dev_winnings[:301], '--', label='Median + 1 SD', color='green')

    # Plot median - std deviation
    plt.plot(median_winnings[:301] - std_dev_winnings[:301], '--', label='Median - 1 SD', color='red')


    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.title('Median Winnings with Standard Deviation Over 1000 Episodes with $256 Bankroll')
    plt.legend(loc='upper left')
    plt.savefig("./images/Figure5.png")  # Save plot
    plt.close()

if __name__ == "__main__":
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
