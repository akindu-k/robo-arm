def get_angles_for_digit(digit):
    if not (0 <= digit <= 9):
        raise ValueError("Input must be an integer between 0 and 9.")

    angle_map = {
        0: [90, 100, 180, 0],
        1: [90, 100, 180, 20],
        2: [90, 100, 180, 40],
        3: [90, 100, 180, 60],
        4: [90, 100, 180, 80],
        5: [90, 100, 180, 100],
        6: [90, 100, 180, 120],
        7: [90, 100, 180, 140],
        8: [90, 100, 180, 160],
        9: [90, 100, 180, 180],
    }

    return angle_map[digit]

digit = int(input("Enter a digit (0-9): "))
angles = get_angles_for_digit(digit)
print(angles)
