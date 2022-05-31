import numpy as np
import random as rnd    

max_decel = 2

class Vehicle:
    marker = None
    
    def __init__(self, Road, pos, lane, vel, p_slow, **kwargs):
        self.pos = pos
        self.lane = lane
        self.prev_lane = 0
        self.p_slow = p_slow
        self.vel = vel
        self.vmax = Road.vmax
        self.road = Road.road
        self.Road = Road
        self.id = Road.generate_id()

    def accelerate(self):
        self.prev_vel = self.vel
        self.vel = min(self.vel+1, self.vmax)

    def decelerate(self):
        self.vel = min(self.vel, self.headway(self.lane))

    def random_slow(self):
        self.rng = np.random.random()
        if (self.rng < self.p_slow):
            self.vel = max(self.vel-1, 0)

    def movement(self):
        self.remove()
        self.pos += self.vel
        if self.Road.is_periodic:
            self.pos %= self.Road.roadlength
        self.place()

    def headway(self, lane):
        headwaycount = 0
        _pos = self.pos+1
        if self.Road.is_periodic:
            _pos %= self.Road.roadlength
        while (self.road[lane, _pos]==0) and (headwaycount<(self.vmax*2)):
            _pos += 1
            if self.Road.is_periodic:
                _pos %= self.Road.roadlength
            headwaycount += 1
        return headwaycount


    def place(self):
        self.road[self.lane, self.pos] = self.id

    def remove(self):
        self.road[self.lane, self.pos] = 0

class Car(Vehicle):
    def __init__(self, Road, pos, lane, vel, p_slow, **kwargs):
        super().__init__(Road, pos, lane, vel, p_slow, **kwargs)
        self.marker = 1

class Bus(Vehicle):
    def __init__(self, Road, pos, lane, vel, p_slow, **kwargs):
        super().__init__(Road, pos, lane, vel, p_slow, **kwargs)
        self.marker = 2
        self.vmax = Road.vmax
        self.road = Road.road
        self.Road = Road
        self.pedestrian = Road.pedestrian
        self.num_passengers = 0
        self.wait_counter = Road.bus_wait_time
        self.id = Road.generate_id()

    def accelerate(self):
        if self.wait_counter==self.Road.bus_wait_time:
            self.prev_vel = self.vel
            self.vel = min(self.vel+1, self.vmax)
        else:
            self.wait_counter += 1

    def decelerate(self):
        hw_pass = self.passenger_headway()
        hw = self.headway(self.lane)

        # skip when too fast
        if (2*self.prev_vel-3*max_decel)>hw_pass:# and (self.prev_vel-max_decel)>(max_decel):
            self.vel = min([self.vel, hw])
        else:
            # anticipate the stop
            if (self.prev_vel+max_decel)>=hw_pass>=max_decel:
                self.vel = min([self.vel, hw, max(self.prev_vel-max_decel, max_decel)]) 
            else:
                self.vel = min([self.vel, hw, hw_pass])
        # print(self.prev_vel, self.vel, hw, hw_pass, c)

    def passenger_headway(self):
        headwaycount = 0
        _pos = self.pos+1
        if self.Road.is_periodic:
            _pos %= self.Road.roadlength
        while (self.pedestrian[self.lane, _pos]==0) and (headwaycount<(self.vmax*2)):
            _pos += 1
            if self.Road.is_periodic:
                _pos %= self.Road.roadlength
            headwaycount += 1
        return headwaycount + 1

    def load(self):
        if self.pedestrian[self.lane, self.pos] != 0:
            self.Road.waiting_times.append(self.pedestrian[self.lane, self.pos])
            self.pedestrian[self.lane, self.pos] = 0
            self.vel = 0
            self.num_passengers += 1
            self.wait_counter = 0


class Model:
    id_counter = 0
    def __init__(self, roadlength, num_lanes, vmax, alpha, 
                    frac_bus, density, p_slow, is_periodic=True,
                    station_period=1, max_passengers=2147483647, bus_wait_time=0):
        self.vehicle_array = []
        self.waiting_times = []
        self.roadlength = roadlength
        self.road = np.zeros((num_lanes, roadlength), dtype=np.int)
        self.pedestrian = np.zeros((num_lanes, roadlength), dtype=np.int)
        self.vmax = vmax
        self.num_lanes = num_lanes
        self.alpha = alpha
        self.is_periodic = is_periodic
        self.p_slow = p_slow
        self.station_period = station_period
        self.max_passengers = max_passengers
        self.bus_wait_time = bus_wait_time
        if frac_bus <= 1./num_lanes:
            self.frac_bus = frac_bus
        else:
            raise ValueError("Invalid Bus Fraction")

        if self.is_periodic:
            num_vehicles = int(density*roadlength*num_lanes)
            num_buses = int(num_vehicles*frac_bus)
            num_cars = num_vehicles - num_buses
            self.place_vehicle_type(Bus, num_buses)
            self.place_vehicle_type(Car, num_cars)

    def generate_id(self):
        self.id_counter = self.id_counter + 1
        return self.id_counter

    def place_vehicle_type(self, veh_type, number):
        for i in range(number):
            pos=0; lane=self.num_lanes-1
            while not self.place_check(pos, lane):
                pos = np.random.randint(self.roadlength)
                if veh_type != Bus: # type checking
                    lane = np.random.randint(self.num_lanes)
                else:
                    lane = self.num_lanes-1
            vehicle = veh_type(Road=self, pos=pos, lane=lane, vel=self.vmax, p_slow=self.p_slow, p_lambda=0 if veh_type == Bus else 1)
            self.vehicle_array.append(vehicle)
            vehicle.place()

    def place_check(self, pos, lane):
        return False if self.road[lane, pos] else True
            
    def timestep_parallel(self):
        if not self.is_periodic:
            self.populate()
        if self.frac_bus>0:
            self.spawn_pedestrian(self.station_period)
        np.random.shuffle(self.vehicle_array)
        reached_end = []
        self.waiting_times = []
        lcs = np.zeros_like(self.vehicle_array)
        for i, vehicle in enumerate(self.vehicle_array):
            vehicle.accelerate()
            if type(vehicle) == Bus:
                if vehicle.num_passengers<self.max_passengers:
                    vehicle.load()

        for i, vehicle in enumerate(self.vehicle_array):

            vehicle.decelerate()
            vehicle.random_slow()

        for i, vehicle in enumerate(self.vehicle_array):

            vehicle.movement()

            if vehicle.pos >= (self.roadlength-self.vmax-1):
                reached_end.append(i)
        if not self.is_periodic:
            self.clear(reached_end)
            
    def timestep(self):
        if not self.is_periodic:
            self.populate()
        if self.frac_bus>0:
            self.spawn_pedestrian()
        np.random.shuffle(self.vehicle_array)
        reached_end = []

        for i, vehicle in enumerate(self.vehicle_array):
            ## core rules
            vehicle.accelerate()
            if type(vehicle) == Bus:
                vehicle.load()
            vehicle.decelerate()
            vehicle.random_slow()
            vehicle.movement()

            if vehicle.pos >= (self.roadlength-self.vmax-1):
                reached_end.append(i)
        if not self.is_periodic:
            self.clear(reached_end)

    def populate(self):
        for i, lane in enumerate(self.road):
            if lane[0] == 0:  # First cell empty
                if (np.random.random() < self.frac_bus_converter()) and (i == (self.num_lanes-1)):
                    vehicle = Bus(Road=self, pos=0, lane=i, vel=self.vmax,
                                  p_slow=self.p_slow, p_lambda=0)
                else:
                    vehicle = Vehicle(
                        Road=self, pos=0, lane=i, vel=self.vmax, p_slow=self.p_slow, p_lambda=1)
                self.vehicle_array.append(vehicle)
                vehicle.place()

    def clear(self, reached_end):
        for i in reached_end:
            self.vehicle_array[i].remove()
        self.vehicle_array = [veh for i, veh in enumerate(
            self.vehicle_array) if i not in reached_end]

    def spawn_pedestrian(self, period=1):
        for i in range(0, len(self.pedestrian[self.num_lanes-1]), period):
            if self.pedestrian[self.num_lanes-1][i] == 0:
                self.pedestrian[self.num_lanes-1][i] += (self.road[self.num_lanes-1,i] == 0) * (np.random.random() < self.alpha)*1
            else:
                # increment waiting time
                self.pedestrian[self.num_lanes-1][i] += 1


    def frac_bus_converter(self):
        return self.num_lanes*self.frac_bus

    def throughput(self):
        return 1.*sum([i.vel for i in self.vehicle_array])/self.roadlength/self.num_lanes


    def get_density(self):
        count = 0
        for i in range(len(self.road)):
            for j in range(len(self.road[0])):
                if self.road[i,j] != 0:
                    count += 1
        return count/self.road.size

    def get_num_full_buses(self):
        count = 0
        for veh in self.vehicle_array:
            if type(veh) == Bus:
                if veh.num_passengers == self.max_passengers:
                    count += 1
        return count

    def get_road(self):
        road = np.zeros((self.num_lanes, self.roadlength), dtype=np.int)
        for veh in self.vehicle_array:
            road[veh.lane, veh.pos] = veh.marker
        return road

    def get_vehicle(self, veh_id):
        for veh in self.vehicle_array:
            if veh.id == veh_id:
                return veh