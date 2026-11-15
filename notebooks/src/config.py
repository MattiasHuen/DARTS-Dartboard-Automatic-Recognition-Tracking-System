N_ANGLES = 720
N_ROWS = 720           # one row per 0.5 degrees around the board
N_COLUMNS = 800        # columns from the bullseye (0.0) to the inner ring (1.0)
N_SECTORS = 20
ROWS_PER_SECTOR = N_ROWS // N_SECTORS

# The sector numbers clockwise, starting from the top of a dartboard
SECTOR_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
RING_GUESSES = dict(bull=0.04, outer_bull=0.10, treble_inner=0.56, treble_outer=0.62, double_inner=0.90, double_outer = 0.97)
