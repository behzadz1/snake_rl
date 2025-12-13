"""
Headless Snake Game for RL Training
No pygame display - significantly faster training
"""
import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

class SnakeGameHeadless:
    """Headless Snake game for fast RL training without display overhead."""
    
    def __init__(self, w=640, h=480, with_wall=False):
        self.w = w
        self.h = h
        self.with_wall = with_wall
        self.wall_segments = []
        self.reset()

    def reset(self):
        """Reset game state for new episode."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        # Setup wall if enabled
        if self.with_wall:
            self._setup_wall()

    def _setup_wall(self):
        """Create a static wall obstacle in the middle of the play area."""
        # Horizontal wall in the middle-upper area
        wall_y = self.h // 3
        wall_start_x = self.w // 4
        wall_end_x = 3 * self.w // 4
        
        self.wall_segments = []
        x = wall_start_x
        while x < wall_end_x:
            self.wall_segments.append(Point(x, wall_y))
            x += BLOCK_SIZE
        
        # Add a vertical section
        wall_x = self.w // 2
        wall_start_y = self.h // 3
        wall_end_y = 2 * self.h // 3
        y = wall_start_y
        while y < wall_end_y:
            segment = Point(wall_x, y)
            if segment not in self.wall_segments:
                self.wall_segments.append(segment)
            y += BLOCK_SIZE

    def _place_food(self):
        """Place food at random valid position."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # Ensure food doesn't spawn on snake or wall
        if self.food in self.snake:
            self._place_food()
        elif self.with_wall and self.food in self.wall_segments:
            self._place_food()

    def play_step(self, action):
        """Execute one game step with given action."""
        self.frame_iteration += 1
        
        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check game over conditions
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check food consumption
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Check if point collides with boundary, snake body, or wall."""
        if pt is None:
            pt = self.head
            
        # Boundary collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # Self collision
        if pt in self.snake[1:]:
            return True
        
        # Wall collision (if enabled)
        if self.with_wall and pt in self.wall_segments:
            return True

        return False

    def _move(self, action):
        """Update head position based on action [straight, right, left]."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
