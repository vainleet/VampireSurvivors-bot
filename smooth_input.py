# smooth_input.py

from pynput.keyboard import Controller

class SmoothInput:
    def __init__(self):
        self.keyboard = Controller()
        self.current = set()
    def update(self, vec):
        wanted = set()
        if vec[0] > 0.3: wanted.add('d')
        elif vec[0] < -0.3: wanted.add('a')
        if vec[1] > 0.3: wanted.add('s')
        elif vec[1] < -0.3: wanted.add('w')
        for k in self.current - wanted:
            self.keyboard.release(k)
        for k in wanted - self.current:
            self.keyboard.press(k)
        self.current = wanted
    def release_all(self):
        for k in self.current: self.keyboard.release(k)
        self.current.clear()