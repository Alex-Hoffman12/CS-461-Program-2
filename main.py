import random
import numpy as np
from dataclasses import dataclass
from typing import List, Set
import copy


@dataclass
class Activity:
    name: str
    expected_enrollment: int
    preferred_facilitators: Set[str]
    other_facilitators: Set[str]
    section: str = None  # For activities with multiple sections (A or B)


@dataclass
class Room:
    name: str
    capacity: int
    building: str


@dataclass
class TimeSlot:
    hour: int
    minute: int

    def __str__(self):
        return f"{self.hour:02d}:{self.minute:02d}"


@dataclass
class ScheduleItem:
    activity: Activity
    room: Room
    time_slot: TimeSlot
    facilitator: str

# Create the Schedule and has all the functions needed to get info about said schedule
class Schedule:
    def __init__(self, items: List[ScheduleItem]):
        self.items = items
        self._fitness = None

    def get_fitness(self) -> float:
        if self._fitness is None:
            self._fitness = self.calculate_fitness()
        return self._fitness

    def calculate_fitness(self) -> float:
        total_fitness = 0
        facilitator_loads = {}  # Count of activities per facilitator
        time_slot_assignments = {}  # Track room usage and facilitator assignments per time slot

        # Initialize facilitator loads
        for item in self.items:
            facilitator_loads[item.facilitator] = facilitator_loads.get(item.facilitator, 0) + 1
            slot_key = str(item.time_slot)
            if slot_key not in time_slot_assignments:
                time_slot_assignments[slot_key] = []
            time_slot_assignments[slot_key].append(item)

        for item in self.items:
            fitness = 0

            # Check room conflicts
            slot_key = str(item.time_slot)
            for other_item in time_slot_assignments[slot_key]:
                if other_item != item and other_item.room == item.room:
                    fitness -= 0.5

            # Room size checks
            if item.room.capacity < item.activity.expected_enrollment:
                fitness -= 0.5
            elif item.room.capacity > 6 * item.activity.expected_enrollment:
                fitness -= 0.4
            elif item.room.capacity > 3 * item.activity.expected_enrollment:
                fitness -= 0.2
            else:
                fitness += 0.3

            # Facilitator preference checks
            if item.facilitator in item.activity.preferred_facilitators:
                fitness += 0.5
            elif item.facilitator in item.activity.other_facilitators:
                fitness += 0.2
            else:
                fitness -= 0.1

            # Facilitator load checks
            facilitator_count_in_slot = sum(1 for x in time_slot_assignments[slot_key]
                                            if x.facilitator == item.facilitator)
            if facilitator_count_in_slot == 1:
                fitness += 0.2
            elif facilitator_count_in_slot > 1:
                fitness -= 0.2

            if facilitator_loads[item.facilitator] > 4:
                fitness -= 0.5
            # Dr. Tyler can't have more than two actives, so checking that separately
            elif facilitator_loads[item.facilitator] == 3 and item.facilitator == "Dr. Tyler":
                fitness -= 0.4
            elif facilitator_loads[item.facilitator] < 3 and item.facilitator != "Dr. Tyler":
                fitness -= 0.4


            # Special rules for SLA 101 and 191
            if item.activity.name in ["SLA 101", "SLA 191"]:
                other_section = next((x for x in self.items
                                      if x.activity.name == item.activity.name
                                      and x.activity.section != item.activity.section), None)
                if other_section:
                    time_diff = abs(other_section.time_slot.hour - item.time_slot.hour)
                    if time_diff > 4:
                        fitness += 0.5
                    elif time_diff == 0:
                        fitness -= 0.5

            # Check consecutive slots for SLA 101 and 191
            if item.activity.name in ["SLA 101", "SLA 191"]:
                for other_item in self.items:
                    if other_item.activity.name in ["SLA 101", "SLA 191"] and other_item != item:
                        time_diff = abs(other_item.time_slot.hour - item.time_slot.hour)
                        if time_diff == 1:
                            fitness += 0.5
                            if ((item.room.building in ["Roman", "Beach"]) !=
                                    (other_item.room.building in ["Roman", "Beach"])):
                                fitness -= 0.4
                        # In the doc it said separated by 1 hour, but the times were from 10AM-12PM, so I just had it do two hours instead
                        elif time_diff == 2:
                            fitness += 0.25
                        elif time_diff == 0:
                            fitness -= 0.25

            total_fitness += fitness

        return total_fitness

# All of the functions needed for the genetic algorithm to work
class GeneticAlgorithm:
    def __init__(self,
                 activities: List[Activity],
                 rooms: List[Room],
                 time_slots: List[TimeSlot],
                 facilitators: List[str],
                 population_size: int = 500,
                 mutation_rate: float = 0.01):
        self.activities = activities
        self.rooms = rooms
        self.time_slots = time_slots
        self.facilitators = facilitators
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self) -> List[Schedule]:
        population = []
        for _ in range(self.population_size):
            schedule_items = []
            for activity in self.activities:
                schedule_items.append(ScheduleItem(
                    activity=activity,
                    room=random.choice(self.rooms),
                    time_slot=random.choice(self.time_slots),
                    facilitator=random.choice(self.facilitators)
                ))
            population.append(Schedule(schedule_items))
        return population

    def select_parent(self, population: List[Schedule]) -> Schedule:
        # Tournament selection
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.get_fitness())

    def crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        # Single-point crossover
        crossover_point = random.randint(1, len(self.activities) - 1)
        child_items = (parent1.items[:crossover_point] +
                       parent2.items[crossover_point:])
        return Schedule(child_items)

    def mutate(self, schedule: Schedule) -> Schedule:
        new_items = copy.deepcopy(schedule.items)
        for item in new_items:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['room', 'time', 'facilitator'])
                if mutation_type == 'room':
                    item.room = random.choice(self.rooms)
                elif mutation_type == 'time':
                    item.time_slot = random.choice(self.time_slots)
                else:
                    item.facilitator = random.choice(self.facilitators)
        return Schedule(new_items)

    @staticmethod
    def softmax(fitnesses: List[float]) -> List[float]:
        exp_f = np.exp(fitnesses - np.max(fitnesses))
        return exp_f / exp_f.sum()

    def evolve(self, generations: int = 100, improvement_threshold: float = 0.01):
        avg_fitness_history = []
        best_fitness_history = []

        for gen in range(generations):
            # Calculate fitness for all schedules
            fitnesses = [schedule.get_fitness() for schedule in self.population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)

            avg_fitness_history.append(avg_fitness)
            best_fitness_history.append(best_fitness)

            # Check for convergence after 100 generations
            if gen >= 100:
                improvement = ((avg_fitness_history[-1] - avg_fitness_history[-100]) /
                               abs(avg_fitness_history[-100]))
                if improvement < improvement_threshold:
                    break

            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(self.population)
                parent2 = self.select_parent(self.population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        return max(self.population, key=lambda x: x.get_fitness())

# Creates the data set (teachers, activities, rooms, times) so the main function doesn't look as messy
def create_actual_data():
    # Facilitators
    facilitators = [
        "Lock", "Glen", "Banks", "Richards", "Shaw", "Singer",
        "Uther", "Tyler", "Numen", "Zeldin"
    ]

    # Activities
    activities = [
        Activity("SLA100", 50,
                 {"Glen", "Lock", "Banks", "Zeldin"},
                 {"Numen", "Richards"}, "A"),
        Activity("SLA100", 50,
                 {"Glen", "Lock", "Banks", "Zeldin"},
                 {"Numen", "Richards"}, "B"),
        Activity("SLA191", 50,
                 {"Glen", "Lock", "Banks", "Zeldin"},
                 {"Numen", "Richards"}, "A"),
        Activity("SLA191", 50,
                 {"Glen", "Lock", "Banks", "Zeldin"},
                 {"Numen", "Richards"}, "B"),
        Activity("SLA201", 50,
                 {"Glen", "Banks", "Zeldin", "Shaw"},
                 {"Numen", "Richards", "Singer"}),
        Activity("SLA291", 50,
                 {"Lock", "Banks", "Zeldin", "Singer"},
                 {"Numen", "Richards", "Shaw", "Tyler"}),
        Activity("SLA303", 60,
                 {"Glen", "Zeldin", "Banks"},
                 {"Numen", "Singer", "Shaw"}),
        Activity("SLA304", 25,
                 {"Glen", "Banks", "Tyler"},
                 {"Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"}),
        Activity("SLA394", 20,
                 {"Tyler", "Singer"},
                 {"Richards", "Zeldin"}),
        Activity("SLA449", 60,
                 {"Tyler", "Singer", "Shaw"},
                 {"Zeldin", "Uther"}),
        Activity("SLA451", 100,
                 {"Tyler", "Singer", "Shaw"},
                 {"Zeldin", "Uther", "Richards", "Banks"})
    ]

    # Rooms
    rooms = [
        Room("Slater 003", 45, "Slater"),
        Room("Roman 216", 30, "Roman"),
        Room("Loft 206", 75, "Loft"),
        Room("Roman 201", 50, "Roman"),
        Room("Loft 310", 108, "Loft"),
        Room("Beach 201", 60, "Beach"),
        Room("Beach 301", 75, "Beach"),
        Room("Logos 325", 450, "Logos"),
        Room("Frank 119", 60, "Frank")
    ]

    # Time slots (10 AM to 3 PM)
    time_slots = [
        TimeSlot(10, 0),  # 10 AM
        TimeSlot(11, 0),  # 11 AM
        TimeSlot(12, 0),  # 12 PM
        TimeSlot(13, 0),  # 1 PM
        TimeSlot(14, 0),  # 2 PM
        TimeSlot(15, 0)  # 3 PM
    ]

    return activities, rooms, time_slots, facilitators


def main():
    # Create actual data
    activities, rooms, time_slots, facilitators = create_actual_data()

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        activities=activities,
        rooms=rooms,
        time_slots=time_slots,
        facilitators=facilitators,
        population_size=500,
        mutation_rate=0.01
    )

    # Runs genetic algorithm for said generation and improvement threshold
    best_schedule = ga.evolve(generations=500, improvement_threshold=0.01)

    # Print schedule so it doesn't look like shit
    print("\n=== BEST SCHEDULE ===")
    print(f"Overall Fitness Score: {best_schedule.get_fitness():.2f}\n")

    # Sort items by time slot for better readability
    sorted_items = sorted(best_schedule.items,
                          key=lambda x: (x.time_slot.hour, x.activity.name))

    # Prints what is happening at each time
    current_time = None
    for item in sorted_items:
        if current_time != item.time_slot:
            current_time = item.time_slot
            print(f"\n=== {current_time} ===")

        print(f"\nActivity: {item.activity.name} "
              f"{item.activity.section or ''}")
        print(f"Room: {item.room.name} "
              f"(Capacity: {item.room.capacity})")
        print(f"Facilitator: {item.facilitator}")
        print(f"Expected Enrollment: {item.activity.expected_enrollment}")
        print("-" * 40)

    # Print facilitator workload summary
    print("\n=== FACILITATOR WORKLOAD ===")
    facilitator_load = {}
    for item in best_schedule.items:
        facilitator_load[item.facilitator] = facilitator_load.get(item.facilitator, 0) + 1

    for facilitator, load in sorted(facilitator_load.items()):
        print(f"{facilitator}: {load} activities")


if __name__ == "__main__":
    main()

