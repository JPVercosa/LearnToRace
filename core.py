# -*- coding: utf-8 -*-

import pyglet

import numpy as np
from numpy import cos,sin,radians,sqrt,ones
from neural_network import NeuralNetwork, NeuralNetworkDDPG, CQ, Mu
from graphics import CarInfo, CarLabel

from objects import Result, ResultDDPG, ResultTD3


# finds intersection between two LINE SEGMENTS
def find_intersection( p0, p1, p2, p3 ) :
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0 : return False # parallel
    denom_is_positive = denom > 0
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive : return False # no collision
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive : return False # no collision
    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision
    # collision detected
    t = t_numer / denom
    intersection_point = [ p0[0] + (t * s10_x), p0[1] + (t * s10_y) ]
    return intersection_point

def dist_between(p0,p1):
    """Distance between two points """
    diffx = abs(p0[0] - p1[0])
    diffy = abs(p0[1] - p1[1])
    dist = sqrt(diffx**2+diffy**2)
    return dist

def angle_between(p0,p1):
    """Angle between two points (in degrees)"""
    diffx = p0[0] - p1[0]
    diffy = p0[1] - p1[1]
    return 270 - np.degrees(np.arctan2(diffx, diffy))

def index_loop(ind, len):
    return ind % len if ind >= len else ind

class ReplayMemory:
    def __init__(self, state_dims, action_dims, size=3000000, alpha=0.6):
        self.alpha = alpha  # Prioritization exponent
        self.epsilon = 1e-6  # Small constant to ensure no zero priorities

        self.s = np.ndarray([size, state_dims])
        self.a = np.ndarray([size, action_dims])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, state_dims])
        self.terminal = np.ndarray([size, 1])
        self.priorities = np.zeros(size, dtype=np.float32)

        self.i = 0
        self.n = 0
        self.size = size

    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""
        return self.n
    
    def add(self, s, a, r, sp, terminal):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, terminal) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `terminal`
           specifies whether the episode finished at terminal absorbing
           state `sp`.

           Memory.add(s, a, r, sp, terminal, v, logp) additionally records
           the value of state s and log-probability of taking action a.
        """

        self.s[self.i, :] = s
        self.a[self.i, :] = a
        self.r[self.i, :] = r
        self.sp[self.i, :] = sp
        self.terminal[self.i, :] = terminal
        
         # Assign maximum priority to new transition
        max_priority = self.priorities.max() if self.n > 0 else 1.0
        self.priorities[self.i] = max_priority        

        self.i = (self.i + 1) % self.size
        if self.n < self.size:
            self.n += 1

    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.terminal[idx]
    
    def per_sample(self, batch_size, beta=0.4):
        if self.n == self.size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.n]
        
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.n, batch_size, p=probs)
        weights = (self.n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (self.s[indices], self.a[indices], self.r[indices], self.sp[indices], self.terminal[indices], indices, weights)
    
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + self.epsilon
        
    def reset(self):
        """Reset memory."""

        self.i = 0
        self.n = 0

"""
Core of simulation.
"""
class Simulation:

    def __init__(self, track=None):
        self.cars = []
        self.track = track

        self.checkpoint_range = [-3, -2, -1, 0, 1]
        self.reward_history = []

    def generate_cars_from_nns(self, nns, parameters, images, batch, labels_batch=None):
        self.cars = []
        for i in range(len(nns)):
            name, image = images[index_loop(i, len(images))]
            sprite = pyglet.sprite.Sprite(image, batch=batch)
            label = CarLabel(name="TST", batch=labels_batch)
            pos = (*self.track.cps_arr[self.track.spawn_index], self.track.spawn_angle)
            # get mutated version of one of best NNs
            self.cars.append(Car(
                nn=nns[i],
                pos=pos,
                parameters=parameters,
                sprite=sprite,
                label=label
            ))

    def get_closest_car_to(self, x, y):
        closest_car, dist = None, float("inf")
        for car in self.cars:
            new_dist = dist_between((x,y), (car.xpos, car.ypos))
            if new_dist < dist:
                dist = new_dist
                closest_car = car
        return closest_car, dist

    # return list  [nn, cp_score, dist_to_next_cp]
    def get_nns_results(self):
        nns_score = []
        for car in self.cars:
            nns_score.append(Result(
                nn=car.nn,
                score=car.score,
                dist_to_next_cp=dist_between(
                    (car.xpos, car.ypos),
                    self.track.cps_arr[index_loop(
                            car.score + self.track.spawn_index + 1,
                            len(self.track.cps_arr)
                    )]
                )
            ))
        return nns_score

    # returns current leader
    def get_leader(self):
        if self.cars:
            return max(self.cars, key=lambda car:car.score)
        else:
            return None

    def get_cars_sorted(self):
        return sorted(self.cars, key= lambda car:(car.score, -car.dist_to_cp), reverse=True)

    # debug
    def get_car_cp_lines(self, car):
        lines_arr = []
        check_ind = index_loop(car.score + self.track.spawn_index, len(self.track.cps_arr))
        for plus in self.checkpoint_range:
            current_check_ind = index_loop(check_ind + plus, len(self.track.cps_arr))
            lines = self.track.lines_arr[current_check_ind]
            lines_arr.append(lines)
        return lines_arr

    # get car inputs (sensors and velocity)
    # because of track subdivision
    # only track segments near current cp are tested if sensors intersect with it
    def get_car_input(self, car):
        inp = ones(car.sensors.shape[0]+1)  # input array

        # index of cp on which car currently is
        cps_length = len(self.track.cps_arr)
        check_ind = index_loop(car.score + self.track.spawn_index, cps_length)

        # sensors index loop
        for sen_ind in range(car.sensors.shape[0]):
            new_pos = car.translate_point(car.sensors_shape[sen_ind])
            # update position of sensor
            car.sensors[sen_ind] = new_pos
            sen_pos = [(car.xpos, car.ypos), new_pos]  # sensor coordinates
            min_dist = 1  # smallest dist

            # goes through nearby cps
            # -1 previous cp
            # 0 = current cp
            # 1 = next cp
            for plus in self.checkpoint_range:
                # current cp
                cp_ind = index_loop(check_ind + plus, cps_length)
                if cp_ind < 0: cp_ind += cps_length
                # line segment which belongs to current cp
                lines = self.track.lines_arr[cp_ind]
                for line in lines:
                    # intersection point (if it exists)
                    intersection = find_intersection(
                        *sen_pos,
                        *line
                    )
                    if intersection:
                        dist = dist_between(intersection, (car.xpos, car.ypos))
                        # did the car crah?
                        if dist < 30:
                            car.speed = 0
                            car.active = False
                            car.sprite.opacity = 100
                        # normalized distance (0,1)
                        dist = dist / car.sensors_lengths[sen_ind]
                        if min_dist > dist: min_dist = dist
            inp[sen_ind] = min_dist
        # append normalized speed (0,1)
        inp[-1] = car.speed / car.param["max_speed"]
        return inp

    # check if car is on next checkpoint
    # checkpoint are sorted in a loop so it is only looking for the next cp
    def update_car_checkpoint(self, car):
        index = index_loop(car.score + self.track.spawn_index, len(self.track.cps_arr))
        checkpoint = self.track.cps_arr[index]  # next checkpoint
        dist = dist_between((car.xpos, car.ypos), checkpoint)
        if (dist < 120):
            car.score += 1
            car.dist_to_cp = float('inf')
        else:
            car.dist_to_cp = dist

    def calculate_reward(self, car, inp):  
        # # print(reward, car.dist_to_next_cp)
        if inp[-1] < 0:
            car.reward -= abs(car.reward) * 100
            car.active = False
        # print(inp[-1])
        if inp[-1] > 0.01:
            car.reward += 0.1
        elif inp[-1] > 0.1:
            car.reward += 0.5
        elif inp[-1] > 0.2:
            car.reward += 1
        elif inp[-1] > 0.3:
            car.reward += 1.5
        elif inp[-1] > 0.4:
            car.reward += 2
        
        # print(inp)
        if inp[0] > 0.4 and inp[1] > 0.4 and inp[2] > 0.4 and inp[3] > 0.4 and inp[4] > 0.4 and inp[5] > 0.2:
            car.reward += 5
            #print("Near center")
        if inp[0] < 0.4 or inp[1] < 0.4 or inp[2] < 0.4 or inp[3] < 0.4 or inp[4] < 0.4:
            car.reward -= 0.5
            #print("Not near center")

        return car.reward

    # behaviour of cars (acceleration, steering)
    # run nn
    def behave(self, dt):
        inactive = True
        for car in self.cars:
            if car.active == True:
                inactive = False
                # get nn input
                inp = self.get_car_input(car)
                self.update_car_checkpoint(car)
                # get nn output
                out = car.nn.forward(inp)
                # move the car!
                car.move(out[1])  # number between 0 and 1
                car.turn((out[0]-.5)*2)  # number between -1 and 1

                if car.active == False:
                    self.reward_history.append(car.score)
                    # print(car.reward)

        # if no car is active :(
        if inactive: return False
        else: return True

    # tick
    def update(self, dt):
        for car in self.cars:
            car.update()

class SimulationDDPG(Simulation):
    def __init__(self, track=None,
                 gamma=0.99, 
                 batch_size=256):
        super().__init__(track)

        self.memory = ReplayMemory(6, 2, 1000000)
        self.gamma = gamma
        self.batch_size = 512
        self.noise_stddev = 0.25
        self.update_counter = 0
        self.update_target = 250

    def generate_cars_from_nns(self, cq, mu, parameters, images, batch, labels_batch=None):
        self.cars = []
        assert len(cq) == len(mu), "cq and mu must have the same length"
        for i in range(len(cq)):
            name, image = images[index_loop(i, len(images))]
            sprite = pyglet.sprite.Sprite(image, batch=batch)
            label = CarLabel(name="TST", batch=labels_batch)
            pos = (*self.track.cps_arr[self.track.spawn_index], self.track.spawn_angle)
            
            # get mutated version of one of best NNs
            self.cars.append(CarDDPG(
                cq=cq[i],
                mu=mu[i],
                pos=pos,
                parameters=parameters,
                sprite=sprite,
                label=label
            ))

    def get_nns_results(self):
        nns_score = []
        for car in self.cars:
            car.dist_to_next_cp = dist_between(
            (car.xpos, car.ypos),
            self.track.cps_arr[index_loop(
                    car.score + self.track.spawn_index + 1,
                    len(self.track.cps_arr)
                )]
            )
            nns_score.append(ResultDDPG(
                cq=car.cq,
                mu=car.mu,
                score=car.score,
                reward=car.reward,
                dist_to_next_cp= car.dist_to_next_cp
            ))
        return nns_score

    def behave(self, dt):
        inactive = True
        for car in self.cars:
            if car.active:
                
                inactive = False

                theta = 0.15

                # Get the current state input
                inp = self.get_car_input(car)
                if car.state is None:
                    next_inp = inp
                    car.state = inp     
                    action = car.actor(car.state)
                    #Add Ornstein-Uhlenbeck noise to the action
                    car.noise = (1-theta)*car.noise + np.random.normal(loc=0, scale=self.noise_stddev)
                    action = action + car.noise
                           

                else:
                    action = car.actor(car.state)
                    #Add Ornstein-Uhlenbeck noise to the action
                    car.noise = (1-theta)*car.noise + np.random.normal(loc=0, scale=self.noise_stddev)
                    action = action + car.noise
                    next_inp = inp

                self.update_car_checkpoint(car)
                # Get the action output from the actor network
                
                # Move the car
                car.turn(action[0])  # number between -1 and 1
                car.move((action[1] + 1) / 2)  # number between 0 and 1
                
                reward = self.calculate_reward(car, inp)              

                terminal = 0 if car.active else 1
                # if terminal: reward = -20
                if terminal:
                    # print(car.reward)
                    self.reward_history.append(car.score)
                self.memory.add(car.state, action, reward, next_inp, terminal)
                
                if len(self.memory) % 1000 == 0:
                    print(len(self.memory))

                car.state = inp
                
                # Update the networks if there are enough experiences
                if len(self.memory) >= self.batch_size*2: #and len(self.memory) % 50 == 0:
                    minibatch = self.memory.per_sample(self.batch_size)
                    states, actions, rewards, next_states, terminals, indices, weights = minibatch

                    next_actions = car.a_copy(next_states)
                    # Adicionar ruido Gaussiano ao next_actions (menor que o anterior)
                    next_q_values = car.c_copy(next_states, next_actions)
                    # Vão existir dois criticos semelhantes
                    # Pegar a o mínimo entre eles
                    targets = rewards + self.gamma * next_q_values * (1 - terminals)

                    # Compute current Q-values
                    current_q_values = car.critic(states, actions)

                    # Compute TD error
                    td_errors = np.abs(targets - current_q_values).flatten()

                    car.critic.update(states, actions, targets)
                    # Atualizar o segundo crítico
                    car.actor.update(states, car.critic)
                
                    self.memory.update_priorities(indices, td_errors)

                    self.update_counter += 1
                    if self.update_counter >= self.update_target:
                        # print("Updating target networks")
                        car.c_copy.copyfrom(car.critic)
                        car.a_copy.copyfrom(car.actor)
                        # Atualizar o segundo crítico
                        self.update_counter = 0
                

        # if no car is active :(
        # if episode ended

        if inactive:
            return False
        else:
            return True
        
class SimulationTD3(SimulationDDPG):
    def __init__(self, track=None,
                    gamma=0.99, 
                    batch_size=256):
        super().__init__(track)

    def generate_cars_from_nns(self, cq, cq2, mu, parameters, images, batch, labels_batch=None):
        self.cars = []
        assert len(cq) == len(mu) == len(cq2), "cq, cq2 and mu must have the same length"
        for i in range(len(cq)):
            name, image = images[index_loop(i, len(images))]
            sprite = pyglet.sprite.Sprite(image, batch=batch)
            label = CarLabel(name="TST", batch=labels_batch)
            pos = (*self.track.cps_arr[self.track.spawn_index], self.track.spawn_angle)
            
            # get mutated version of one of best NNs
            self.cars.append(CarTD3(
                cq=cq[i],
                cq2=cq2[i],
                mu=mu[i],
                pos=pos,
                parameters=parameters,
                sprite=sprite,
                label=label
            ))

    def get_nns_results(self):
        nns_score = []
        for car in self.cars:
            car.dist_to_next_cp = dist_between(
            (car.xpos, car.ypos),
            self.track.cps_arr[index_loop(
                    car.score + self.track.spawn_index + 1,
                    len(self.track.cps_arr)
                )]
            )
            nns_score.append(ResultTD3(
                cq=car.cq,
                cq2=car.cq2,
                mu=car.mu,
                score=car.score,
                reward=car.reward,
                dist_to_next_cp= car.dist_to_next_cp
            ))
        return nns_score

    def behave(self, dt):
        inactive = True
        for car in self.cars:
            if car.active:
                
                inactive = False

                theta = 0.15

                # Get the current state input
                # Get the action output from the actor network
                inp = self.get_car_input(car)
                if car.state is None:
                    next_inp = inp
                    car.state = inp     
                    action = car.actor(car.state)
                    #Add Ornstein-Uhlenbeck noise to the action
                    car.noise = (1-theta)*car.noise + np.random.normal(loc=0, scale=self.noise_stddev)
                    action = action + car.noise
                           

                else:
                    action = car.actor(car.state)
                    #Add Ornstein-Uhlenbeck noise to the action
                    car.noise = (1-theta)*car.noise + np.random.normal(loc=0, scale=self.noise_stddev)
                    action = action + car.noise
                    next_inp = inp

                self.update_car_checkpoint(car)
                
                # Move the car
                car.turn(action[0])  # number between -1 and 1
                car.move((action[1] + 1) / 2)  # number between 0 and 1
                
                reward = self.calculate_reward(car, inp)              

                terminal = 0 if car.active else 1
                # if terminal: reward = -20
                if terminal:
                    # print(car.reward)
                    self.reward_history.append(car.score)
                self.memory.add(car.state, action, reward, next_inp, terminal)
                
                if len(self.memory) % 1000 == 0:
                    print(len(self.memory))

                car.state = inp
                
                # Update the networks if there are enough experiences
                if len(self.memory) >= self.batch_size*2: #and len(self.memory) % 50 == 0:
                    minibatch = self.memory.per_sample(self.batch_size)
                    states, actions, rewards, next_states, terminals, indices, weights = minibatch

                    next_actions = car.a_copy(next_states)
                    # Adicionar ruido Gaussiano ao next_actions (menor que o anterior)
                    noise = np.random.normal(0, 0.2, size=next_actions.shape)
                    next_actions = next_actions + noise
                    next_actions[0] = np.clip(next_actions[0], -1, 1)
                    next_actions[1] = np.clip(next_actions[1], 0, 1)

                    next_q_values_1 = car.c1_copy(next_states, next_actions)
                    next_q_values_2 = car.c2_copy(next_states, next_actions)
                    # Vão existir dois criticos semelhantes
                    # Pegar a o mínimo entre eles
                    next_q_values = np.minimum(next_q_values_1, next_q_values_2)
                    targets = rewards + self.gamma * next_q_values * (1 - terminals)

                    # Compute current Q-values
                    current_q_values_1 = car.critic1(states, actions)
                    current_q_values_2 = car.critic2(states, actions)

                    # Compute TD errors
                    td_errors_1 = np.abs(targets - current_q_values_1).flatten()
                    td_errors_2 = np.abs(targets - current_q_values_2).flatten()

                    # Atualizar o segundo crítico
                    car.critic1.update(states, actions, targets)
                    car.critic2.update(states, actions, targets)
                    car.actor.update(states, car.critic)
                
                    td_errors = td_errors_1 + td_errors_2
                    self.memory.update_priorities(indices, td_errors)

                    self.update_counter += 1
                    if self.update_counter >= self.update_target:
                        # print("Updating target networks")
                        car.c1_copy.copyfrom(car.critic)
                        car.c2_copy.copyfrom(car.critic)
                        car.a_copy.copyfrom(car.actor)
                        # Atualizar o segundo crítico
                        self.update_counter = 0
                

        # if no car is active :(
        # if episode ended

        if inactive:
            return False
        else:
            return True




"""
Track object.

The way this works is the best I have come up with so far. 

Track consists of several 2-point "gates" in a loop. These points connect to form the edge of a track.
    
Checkpoints are in the middle of every "gate". So to each checkpoint belongs only 2 line segments connected to the next "gate".

When car is at the CP it checks for intersection only with nearby lines belonging to nearby CPs.
And dont have to find intersection on every line.
"""
class Track:
    def __init__(self, nodes, shape, spawn_index=0, spawn_angle=None, bg=False):
        self.nodes = nodes
        self.shape = shape  # tile shape - eg (5, 3)
        self.vertex_lists = (
            pyglet.graphics.vertex_list(len(self.nodes[0]), ('v2i', (self.nodes[0].flatten()).astype(int))),
            pyglet.graphics.vertex_list(len(self.nodes[1]), ('v2i', (self.nodes[1].flatten()).astype(int)))
        )
        self.bg = bg

        # (n, left/right, prev/next, x/y)
        self.lines_arr = self.nodes_to_lines(self.nodes)
        self.cps_arr = self.nodes_to_cps(self.nodes)
        self.spawn_index = spawn_index

        self.spawn_angle = spawn_angle
        if self.spawn_angle == None:
            self.spawn_angle = angle_between(self.cps_arr[self.spawn_index], self.cps_arr[self.spawn_index + 1])

    def nodes_to_lines(self, nodes):
        """
        :param nodes: shape (left/right, n, x/y)
        :return: lines: shape (n, left/right, prev/next, x/y)
        """
        lines = np.swapaxes(np.stack((nodes, np.roll(nodes, -1, 1)), axis=2), 0, 1)
        return lines

    def nodes_to_cps(self, nodes):
        """
        :param nodes: shape (left/right, n, x/y)
        :return: cps: point in the center with shape (n, xy)
        """
        center_point = lambda gate: [(gate[0,0] + gate[1,0]) // 2, (gate[0,1] + gate[1,1]) // 2]
        return np.array([center_point(nodes[:,i,:]) for i in range(nodes.shape[1])])

    def change_scale(self, scale):
        pass
        """for i in range(len(self.vertex_lists)):
            self.vertex_lists[i].vertices = (self.nodes[i].flatten() * scale).astype(int)
        if self.bg: self.bg.scale = scale"""

"""
Car object. 
Each one has its own neural network and sprite.
"""

class Car:
    def __init__(self, nn: NeuralNetwork, pos: tuple, sprite, parameters: dict, label: CarLabel = None):
        self.nn = nn

        """
        param = {
            accerelation,
            max_speed,
            rotation_speed
            friction
        }
        """
        self.param = parameters

        self.xpos = pos[0]
        self.ypos = pos[1]
        self.angle = pos[2]

        self.active = True
        self.speed = 0
        # number of cps
        self.score = 0
        self.dist_to_cp = 0

        # sprite
        self.sprite = sprite

        # sensors - from point [0, 0]
        #    \  |  /
        #     \ | /
        # -----[ ]-----
        self.sensors_shape = np.array([
            [25, -140],
            [190, -100],
            [300, 0],
            [190, 100],
            [25, 140],
        ])
        self.sensors = np.copy(self.sensors_shape)
        self.sensors_lengths = [dist_between((0,0), pos) for pos in self.sensors]

        self.info = CarInfo()

        self.label = label

    # returns translated point (coordinates from perspective of car -> coordinates on screen)
    def translate_point(self, p):
        x, y = p
        _cos = np.cos(radians(self.angle))
        _sin = np.sin(radians(self.angle))
        new_x = x * _cos + y * _sin + self.xpos
        new_y = -(-x * _sin + y * _cos) + self.ypos
        return (new_x, new_y)

    def update_label(self):
        pass

    def update_info(self):
        self.info.labels["active"].text = str(self.active)
        self.info.labels["score"].text = str(self.score)
        self.info.labels["speed"].text = str(round(self.speed, 2))

    # apply translation to every sensor
    def update_sensors(self):
        self.translate_point(self.sensors)

    # tick
    def update(self):
        if self.active == True:
            self.xpos += cos(radians(self.angle)) * self.speed
            self.ypos += sin(radians(self.angle)) * self.speed

    # tick
    def move(self, direction=1):
        self.speed += self.param["acceleration"] * direction
        if self.speed > self.param["max_speed"]:
            self.speed = self.param["max_speed"]
        self.speed *= self.param["friction"]

    # tick
    def turn(self, direction=1):
        # direction = 1 or -1
        self.angle += self.param["rotation_speed"] * direction
        if self.angle > 360: self.angle -= 360
        if self.angle < 0: self.angle += 360

class CarDDPG(Car):
    def __init__(self, cq: CQ, mu: Mu, pos: tuple, sprite, parameters: dict, label: CarLabel = None):
        super().__init__(None, pos, sprite, parameters, label)

        # self.critic, self.c_copy = CQ(6, 2), CQ(6, 2)
        # self.actor, self.a_copy = Mu(6, 2), Mu(6, 2)
        self.cq = cq
        self.mu = mu

        self.critic = cq
        self.c_copy = cq
        self.actor = mu 
        self.a_copy = mu

        self.dist_to_next_cp = 0
        self.reward = 0

        self.noise = 0
        self.state = None

class CarTD3(CarDDPG):
    def __init__(self, cq: CQ, mu: Mu, pos: tuple, sprite, parameters: dict, label: CarLabel = None, cq2: CQ = None):
        super().__init__(cq, mu, pos, sprite, parameters, label)

        self.cq2 = cq2

        self.critic1 = cq
        self.critic2 = cq2
        self.c1_copy = cq
        self.c2_copy = cq2
        
