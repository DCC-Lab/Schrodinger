import numpy as np


def model_epr():
	"""
	The "States" in EPR and QM are very different.

	Here, we have possible states according to EPR that are selected upon the
	creation of the two-particle system.

	I am assuming all identical population of N1 to N8 states.
	Another version of the code may explore that possibility.
	"""
	states = [
		([+1,+1,+1],[-1,-1,-1]),
		([+1,+1,-1],[-1,-1,+1]),
		([+1,-1,+1],[-1,+1,-1]),
		([+1,-1,-1],[-1,+1,+1]),
		([-1,+1,+1],[+1,-1,-1]),
		([-1,+1,-1],[+1,-1,+1]),
		([-1,-1,+1],[+1,+1,-1]),
		([-1,-1,-1],[+1,+1,+1])
	]
	stats_axes_selected = [0,0,0]
	stats_states_selected = [0,0,0,0,0,0,0,0]

	def validate_states():
		for state_a, state_b in states:
			for axis in [0,1,2]:
				assert state_a[axis] * state_b[axis] < 0

	def choose_random_populations():
		return np.random.randint(100, size=8)

	def choose_axes():
		axis_a = np.random.randint(0, 3)
		axis_b = np.random.randint(0, 3)
		stats_axes_selected[axis_a] += 1
		stats_axes_selected[axis_b] += 1
		return axis_a, axis_b

	def choose_initial_states(states_prob):
		"""
		I am assuming all identical population of N1 to N8 states.
		Another version of the code may explore the possibility
		of different populations
		"""
		index = np.random.choice(range(8), p=states_prob)
		stats_states_selected[index] += 1

		return states[index]

	validate_states() # Just to make sure we didn't make a mistake (I did the first time)

	N = 100000
	same = 0

	"""
	We can try any populations, the most correlation will always be 4/9
	"""
	# states_Ni = np.array([1,1,1,1,1,1,1,1]) # equal populations
	# states_Ni = np.array([0,1,0,0,0,0,0,0]) 
	states_Ni = choose_random_populations()

	states_prob = states_Ni / np.sum(states_Ni)
	for _ in range(N):
		state_a, state_b = choose_initial_states(states_prob)

		axis_a, axis_b = choose_axes()

		measurement = state_a[axis_a],state_b[axis_b]

		if measurement[0]*measurement[1] > 0: # ++ or -- gives > 0
			same += 1

	same_stat = same/N*100

	print(f"EPR Model")
	print(f"=========")
	print(f"After performing the experiment {N} times, with the 8 states having probabilities {[f"{p:0.1f}%" for p in states_prob*100]} we obtained:")

	if same_stat <= 4/9*100:
		print(f"Same {same_stat:.2f}% ≤ 4/9 of the time, as expected if hidden variables model is correct")
	else:
		print(f"Error!? Same : {same_stat:.2f}% > 4/9") # We are replicating the EPR strategy, this should not happen

	print("\nQuick validation of code:")
	print(f"Validation axis selection stats: {stats_axes_selected/np.sum(stats_axes_selected)*100} [expect 33.3% each]")
	print(f"States selection stats: {stats_states_selected/np.sum(stats_states_selected)*100}")
	print(f"States selection probs: {states_prob*100}\n\n")


def model_qm():
	stats_axes_selected = [0,0,0]

	"""
	The "States" in EPR and QM are very different.

	Here, we have possible states according to QM which are expressed 
	in a given basis. The states "| + + >" and "| - - >" are not 
	possible here, but they are part of the basis.
	"""

	basis_states = ["| + + >","| + - >","| - + >","| - - >"]

	def choose_initial_state():
		return np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])

	def choose_axes():
		axis_a = np.random.randint(0, 3)
		axis_b = np.random.randint(0, 3)
		stats_axes_selected[axis_a] += 1
		stats_axes_selected[axis_b] += 1
		return axis_a, axis_b

	N = 1000000
	same = 0

	for _ in range(N):

		"""
		We refer to Alice and Bob for measurements a and b.

		We start with an equal superposition of | + - > and | - + >. We use
		the basis we call "z-basis", but we could choose any basis.
		"""
		state_coefficients = choose_initial_state()
		state_prob = state_coefficients**2


		"""
		Alice measures whatever axis she wants, but we define it as z so the
		original state is already in that basis. We could also convert it
		like we will do for the second particle, but for now, let's do the
		same thing as Mcintyre.

		Theta is the angle difference between Alice's and Bob's axes.
		"""
		axis_a, axis_b = choose_axes()
		theta = (axis_b - axis_a) * 2*np.pi/3

		"""
		The probability of Alice measuring + or - is obtained from the
		coefficients in that basis.  Since we have defined the chosen axis
		as z, it is
		"""

		measurement_a = np.random.choice(basis_states, p=state_prob)

		"""
		The measurement by Alice collapses (i.e. projects) the wavefunction
		onto either "| + - >" or "| - + >". We determine what the measurement was,
		and from there calculate the probabiblity of Bob's particle to be in + or - 
		along Bob's axis.

		If Bob has selected exactly the same axis as Alice (i.e. theta=0, see
		above), then the state will be the opposite of Alice because it
		has to. This is obvious from Eq. 2.42 because:

		if measurement_a == "| + - >":
			measurement_a = +1 # Alice measured +, if Bob measures along same axis, then 100% certain to get -
			probs_along_axis_b = [0, 1] # If theta = 0
		elif measurement_a == "| - + >":
			measurement_a = -1 # Alice measured -, if Bob measures along same axis, then 100% certain to get +
			probs_along_axis_b = [1, 0] # If theta = 0

		But Bob may have selected a different axis. Therefore, two ways of doing this:

		1. I prefer to express Bob's state in the basis of his chosen axis. This is Eq. 2.42
		2. McIntyre expresses the projection bra in Bob's axis and then calculates for both 
		possible states.  I think option 1. reflects a measurement better here.
		"""
		if measurement_a == "| + - >":
			measurement_a = +1 # Alice measured +, Bob's state *IN THAT BASIS* is -
			probs_along_axis_b = [np.sin(theta/2)**2, np.cos(theta/2)**2] # Eq. 2.42 McIntyre
		elif measurement_a == "| - + >":
			measurement_a = -1 # Alice measured -, Bob's state *IN THAT BASIS* is +
			probs_along_axis_b = [np.cos(theta/2)**2, np.sin(theta/2)**2] # Eq. 2.42 McIntyre
		else:
			raise RuntimeError("Wtf?")

		"""
		According to the probabilities form the coefficient, we "measure" + or -
		"""
		measurement_b = np.random.choice([+1, -1], p=probs_along_axis_b)

		if measurement_a*measurement_b > 0:
			same += 1

	same_stat = same/N*100

	print(f"Quantum Mechanics Model")
	print(f"=======================")
	print(f"After performing the experiment {N} times, we obtained:")

	if same_stat > 4/9*100:
		print(f"QM measurement : {same_stat:.2f}% > {4/9*100:.2f} of the time, as expected if hidden variables model is incorrect")
	else:
		print(f"QM measurement : {same_stat:.2f}% > {4/9*100:.2f}")

	print("\nQuick validation of code:")
	print(f"Validation axis selection stats: {stats_axes_selected/np.sum(stats_axes_selected)*100} [expect 33.3% each]")


if __name__ == "__main__":
	model_epr()
	model_qm()

	print("\n\n")
	print(f"Experimental validation")
	print(f"=========================")
	print("Only an experiment can discrimiate the two possiblities.")
	print("Lab measurements show that the correlation (i.e. measuring the same spin for both) is close to 50%), therefore QM is correct, EPR is incorrect.")
