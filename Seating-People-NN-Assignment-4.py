#!/usr/bin/env python
# coding: utf-8

# In[17]:


import random
import torch

# Utility function to print a 2D array with indices
def print_2d_array_with_indices_table(array):
    print("   ", end=" ")
    for j in range(len(array[0])):
        print(f"{j:4}", end=" ")
    print()  
    for i, row in enumerate(array):
        print(f"{i:2}", end=" ")
        for val in row:
            print(f"{val:4}", end=" ")
        print()

# Generate a list of unique full names
def get_names():
    first_names = ["Ali", "Zahra", "Reza", "Sara", "Mohammad", "Fatemeh", 
                   "Hossein", "Maryam", "Mehdi", "Narges", "Hamed", "Roya"]
    last_names = ["Ahmadi", "Hosseini", "Karimi", "Rahimi", "Hashemi", "Ebrahimi", 
                  "Moradi", "Mohammadi", "Rostami", "Fazeli", "Hosseinzadeh", "Niknam"]
    random.seed(0)
    full_names = set()
    while len(full_names) < 24:
        full_name = random.choice(first_names) + " " + random.choice(last_names)
        full_names.add(full_name)
    return list(full_names)

# Generate a conflict matrix
def get_conflicts(count_conflicts=80):
    conflicts = torch.zeros([24, 24], dtype=torch.int64)
    while sum(sum(conflicts)) < count_conflicts:
        x = random.randint(0, 23)
        y = random.randint(0, 23)
        conflicts[x, y] = 1
        conflicts[y, x] = 1
    return conflicts

# Get the neighbors of a given position in the seating arrangement
def get_neighbors(i, j):
    neighbors = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i < 5:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < 3:
        neighbors.append((i, j + 1))
    return neighbors

# Compute the loss based on the conflicts and the seating arrangement
def get_loss(conflicts, chairs):
    loss = 0
    for i in range(6):
        for j in range(4):
            neighbors = get_neighbors(i, j)
            person_one = chairs[i, j]
            for x, y in neighbors:
                person_two = chairs[x, y]
                loss += conflicts[person_one, person_two]
    return loss

# Randomly seat people in the chairs
def people_sitting_on_chairs(chairs):
    persons = set(range(24))
    for i in range(chairs.size(0)):
        for j in range(chairs.size(1)):
            person = random.choice(list(persons))
            persons.remove(person)
            chairs[i, j] = person
    return chairs

# Swap two seats in the seating arrangement
def swap_seats(chairs, i1, j1, i2, j2):
    temp = chairs[i1, j1].item()
    chairs[i1, j1] = chairs[i2, j2].item()
    chairs[i2, j2] = temp
    return chairs

# Main function to initialize and optimize the seating arrangement
def main():
    names = get_names()
    conflicts = get_conflicts()
    chairs = torch.zeros(6, 4, dtype=torch.int64)
    chairs = people_sitting_on_chairs(chairs)

    print("Initial Conflict Matrix:")
    print_2d_array_with_indices_table(conflicts)

    print("\nInitial Chairs Arrangement:")
    print_2d_array_with_indices_table(chairs)

    initial_loss = get_loss(conflicts, chairs)
    print(f"\nInitial Loss: {initial_loss}")

    num_epochs = 1000
    for epoch in range(num_epochs):
        current_loss = get_loss(conflicts, chairs)
        best_loss = current_loss
        best_chairs = chairs.clone()

        for i1 in range(6):
            for j1 in range(4):
                for i2 in range(6):
                    for j2 in range(4):
                        if i1 == i2 and j1 == j2:
                            continue
                        chairs = swap_seats(chairs, i1, j1, i2, j2)
                        new_loss = get_loss(conflicts, chairs)
                        if new_loss < best_loss:
                            best_loss = new_loss
                            best_chairs = chairs.clone()
                        chairs = swap_seats(chairs, i1, j1, i2, j2)  # Swap back

        chairs = best_chairs.clone()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {best_loss}")

    print("\nFinal Chairs Arrangement:")
    print_2d_array_with_indices_table(chairs)

    final_loss = get_loss(conflicts, chairs)
    print(f"\nFinal Loss: {final_loss}")

main()


# In[ ]:




