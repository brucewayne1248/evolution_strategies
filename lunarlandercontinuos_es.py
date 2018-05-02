import gym
import numpy as np
import pickle
import sys

env = gym.make("LunarLanderContinuous-v2")
np.random.seed(1337)
env.seed(1337)

h1_size = 100
version = 1 # naming the model
# Hyperparams of Evolution Strategies https://blog.openai.com/evolution-strategies/
npop = 50 # population size, generating npop different parameter vectors w_1, ..., w_npop
sigma = 0.1 # gaussian std used to generate different parameter vectors
alpha = 0.03 # learning rate, indicating how strongly parameter vector w is pushed towards better solution
#iter_num = 300 # max steps used in one episode
aver_reward = None
allow_writing = True # saving model in pickle file
reload = True # reload previously saved model
save_size = 10 # every how many iterations sould the model be saved

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


if reload:
   model = pickle.load(open("model-lunar%d.p" % version, "rb"))
else:
   # initialize weights of feed foward model
   model = {}
   model["W1"] = np.random.randn(state_dim, h1_size) / np.sqrt(state_dim)
   model["W2"] = np.random.randn(h1_size, action_dim) / np.sqrt(h1_size)

def get_action(state, model):
   h1 = np.matmul(state, model["W1"])
   h1 = np.tanh(h1) # squash values between -1 and 1
   action = np.matmul(h1, model["W2"])
   action = np.tanh(action)
   return action

state = env.reset()

def f(model, render=False):
   """function that plays one episode and returns total reward"""
   state = env.reset()
   total_reward = 0
   while True:
      if render: env.render()

      action = get_action(state, model)
      state, reward, done, info = env.step(action)
      total_reward += reward

      if done:
         break
   return total_reward

# run the environment with a trained agent
if reload:
   iter_num = 10000
   for episode in range(5):
      print(f(model, render=True))
   env.close() # close render window
   sys.exit("demo finished")

total_iterations = 1000
average_reward = 0
for i in range(total_iterations):
   N = {}
   for key, value in model.items():
      # create npop random N matrices of shape W1 (state_dim, h1_size) and W2 (h1_size, action_dim)
      N[key] = np.random.randn(npop, value.shape[0], value.shape[1])
   R = np.zeros(npop) # vector containing the total rewards of jittered trajectories

   for traj in range(npop):
      model_jitter = {}
      for key, value in model.items():
         model_jitter[key] = value + sigma*N[key][traj]
      R[traj] = f(model_jitter)

   A = (R - np.mean(R)) / np.std(R) # normalize rewards
   # update the model parameters
   for key in model:
      # np.dot is the sum product over the last axis of N[key].transpose(1,2,0) and A
      # e.g. np.dot(shape(24,100,50), shape(50,)) -> shape(24,100)
      # it moves the weights into the direction which gave a big reward
      model[key] = model[key] + alpha/(npop*sigma) * np.dot(N[key].transpose(1, 2, 0), A)

   cur_reward = f(model)
   # new_average = 1/n * (old_average * (n-1)+ new_value) , below i+1 because i starts at 0
   average_reward = 1/(i+1) *(average_reward*i + cur_reward)
   print("iteration {:4d}, reward: {:4.1f}, average reward: {:4.1f}".format(i, cur_reward, average_reward))

   if i % 10 == 0 and allow_writing:
      pickle.dump(model, open("model-lunar%d.p" % version, "wb"))